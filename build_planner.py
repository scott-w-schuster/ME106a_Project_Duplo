"""
build_planner.py

Main planning node for the Duplo building task.

Workflow:
  1. Load the target build spec from a YAML file
  2. Wait for brick_detector to publish baseplate_frame TF
  3. For each layer (bottom to top):
     a. For each brick needed in this layer:
        - Find a matching brick on the table via /detected_bricks
        - Compute target placement pose from baseplate_frame TF + grid offset
        - Pick the brick from the table
        - Place the brick on the baseplate
        - Mark the slot as placed in software state
     b. Verify the completed layer with the camera before proceeding
  4. Build complete

ROS2 Topics:
  Subscribed:
    /detected_bricks        (geometry_msgs/PoseArray)
    /detected_bricks_meta   (std_msgs/String)  [JSON color+shape per pose]
  TF:
    Reads: base_link -> baseplate_frame  (published by brick_detector)

Build Spec YAML format:
  bricks:
    - color:       red
      shape:       [2, 4]    # [rows, cols]
      position:    [0, 0]    # [row, col] on baseplate stud grid
      orientation: 0         # degrees relative to baseplate x-axis (0 or 90)
      layer:       0         # 0 = directly on baseplate
"""

import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import yaml
import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import String

import tf2_ros
import tf2_geometry_msgs


# ---------------------------------------------------------------------------
# Duplo physical constants (meters)
# ---------------------------------------------------------------------------
STUD_PITCH_M    = 0.016    # 16 mm between stud centers
BRICK_HEIGHT_M  = 0.0192   # 19.2 mm per layer (bottom of plate to bottom of next plate)

# ---------------------------------------------------------------------------
# Arm motion clearances
# ---------------------------------------------------------------------------
APPROACH_CLEARANCE_M = 0.08   # height above target for pre-grasp / pre-place hover
RETRACT_CLEARANCE_M  = 0.10   # height to retract to after pick or place


class BuildPlannerNode(Node):

    def __init__(self):
        super().__init__('build_planner')

        # --- Build spec ---
        self.declare_parameter('build_spec', 'build.yaml')
        spec_path = self.get_parameter('build_spec').get_parameter_value().string_value
        self.build_layers = self._load_build_spec(spec_path)
        total = sum(len(v) for v in self.build_layers.values())
        self.get_logger().info(
            f'Loaded {total} bricks across {len(self.build_layers)} layers from {spec_path}'
        )

        # Software state: set of (row, col, layer) slots confirmed placed
        self.placed = set()

        # Latest detections from brick_detector
        self.latest_poses = []   # list of Pose in base_link
        self.latest_meta  = []   # list of {color, shape}
        self._detections_lock = threading.Lock()

        # Callbacks fire concurrently with run() via MultiThreadedExecutor
        cb_group = ReentrantCallbackGroup()
        self.create_subscription(
            PoseArray, '/detected_bricks',      self._bricks_callback, 10,
            callback_group=cb_group
        )
        self.create_subscription(
            String,    '/detected_bricks_meta', self._meta_callback,   10,
            callback_group=cb_group
        )

        # TF — reads baseplate_frame broadcast by brick_detector each cycle
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Arm interface
        self._init_arm()

        self.get_logger().info('BuildPlannerNode initialized.')

    # -----------------------------------------------------------------------
    # Build spec
    # -----------------------------------------------------------------------

    def _load_build_spec(self, path: str) -> dict:
        """
        Load YAML build spec.
        Returns {layer_index: [brick_dict, ...]} sorted from lowest to highest layer.

        Each brick_dict keys: color, shape, position, orientation, layer
        """
        with open(path, 'r') as f:
            spec = yaml.safe_load(f)

        layers = {}
        for brick in spec['bricks']:
            layer = brick['layer']
            # Default orientation: 0 degrees (long axis along baseplate x)
            brick.setdefault('orientation', 0)
            layers.setdefault(layer, []).append(brick)

        return dict(sorted(layers.items()))

    # -----------------------------------------------------------------------
    # Subscriber callbacks
    # -----------------------------------------------------------------------

    def _bricks_callback(self, msg: PoseArray):
        with self._detections_lock:
            self.latest_poses = list(msg.poses)

    def _meta_callback(self, msg: String):
        with self._detections_lock:
            self.latest_meta = json.loads(msg.data)

    # -----------------------------------------------------------------------
    # Main planning loop
    # -----------------------------------------------------------------------

    def run(self):
        """
        Execute the full build plan layer by layer.
        Runs in the main thread while the executor spins in a background thread.
        """
        self.get_logger().info('Waiting for baseplate_frame TF...')
        if not self._wait_for_baseplate(timeout_sec=30.0):
            self.get_logger().error('Aborting: baseplate not detected.')
            return

        for layer_idx, bricks_in_layer in self.build_layers.items():
            self.get_logger().info(f'=== Layer {layer_idx} — {len(bricks_in_layer)} bricks ===')

            for brick_spec in bricks_in_layer:
                color       = brick_spec['color']
                shape       = tuple(brick_spec['shape'])        # (rows, cols)
                row, col    = brick_spec['position']
                orientation = brick_spec['orientation']         # degrees, 0 or 90

                slot = (row, col, layer_idx)
                if slot in self.placed:
                    self.get_logger().info(f'  Slot {slot} already placed, skipping.')
                    continue

                self.get_logger().info(
                    f'  Placing {color} {shape[0]}x{shape[1]} → grid ({row},{col}) '
                    f'layer {layer_idx} orientation {orientation}°'
                )

                success = False
                for attempt in range(3):
                    # Find a matching brick on the table
                    source_pose = self._find_brick_on_table(color, shape)
                    if source_pose is None:
                        self.get_logger().warn(
                            f'  No {color} {shape} on table (attempt {attempt+1}/3) — waiting...'
                        )
                        time.sleep(1.5)
                        continue

                    # Compute where to put it on the baseplate
                    target = self._compute_placement_pose(row, col, layer_idx, orientation)
                    if target is None:
                        self.get_logger().warn('  baseplate_frame unavailable, retrying...')
                        time.sleep(1.0)
                        continue

                    # Pick and place
                    if self._pick(source_pose) and self._place(target):
                        self.placed.add(slot)
                        success = True
                        break
                    else:
                        self.get_logger().warn(f'  Pick-place failed (attempt {attempt+1}/3)')
                        time.sleep(1.0)

                if not success:
                    self.get_logger().error(
                        f'  FAILED to place {color} {shape} at {slot} after 3 attempts. Continuing.'
                    )

            # Verify this layer with the camera before moving on
            self._verify_layer(layer_idx, bricks_in_layer)

        self.get_logger().info('Build complete!')

    # -----------------------------------------------------------------------
    # Detection helpers
    # -----------------------------------------------------------------------

    def _find_brick_on_table(self, color: str, shape: tuple) -> Pose:
        """
        Search the latest detections for a brick matching color and shape.
        Returns its Pose in base_link, or None if not found.
        """
        with self._detections_lock:
            pairs = list(zip(self.latest_poses, self.latest_meta))

        for pose, meta in pairs:
            if meta['color'] == color and tuple(meta['shape']) == shape:
                return pose

        return None

    def _verify_layer(self, layer_idx: int, bricks_in_layer: list):
        """
        Compare software state against camera-detected baseplate state for this layer.

        Currently checks software state only. For full camera verification, move
        the arm to an observation pose, wait for fresh /detected_bricks data with
        the camera aimed at the baseplate, then compare positions against expected
        baseplate slots.
        """
        self.get_logger().info(f'  Verifying layer {layer_idx}...')

        missing = []
        for brick_spec in bricks_in_layer:
            row, col = brick_spec['position']
            slot     = (row, col, layer_idx)
            if slot not in self.placed:
                missing.append(slot)

        if missing:
            self.get_logger().warn(f'  Layer {layer_idx}: {len(missing)} slot(s) not confirmed: {missing}')
        else:
            self.get_logger().info(f'  Layer {layer_idx}: all slots confirmed placed.')

    # -----------------------------------------------------------------------
    # Pose computation
    # -----------------------------------------------------------------------

    def _compute_placement_pose(self, row: int, col: int, layer: int,
                                 orientation_deg: float) -> PoseStamped:
        """
        Compute the target placement pose in base_link for a brick at (row, col, layer).

        baseplate_frame origin = stud (0,0) corner of the baseplate.
            x = col * STUD_PITCH_M   (along baseplate columns)
            y = row * STUD_PITCH_M   (along baseplate rows)
            z = layer * BRICK_HEIGHT_M

        orientation_deg: yaw of the brick's long axis relative to baseplate x-axis.
        """
        try:
            tf_bp = self.tf_buffer.lookup_transform(
                'base_link', 'baseplate_frame', rclpy.time.Time()
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'baseplate_frame lookup failed: {e}')
            return None

        # Position in baseplate frame
        pose_in_bp = Pose()
        pose_in_bp.position.x = col * STUD_PITCH_M
        pose_in_bp.position.y = row * STUD_PITCH_M
        pose_in_bp.position.z = layer * BRICK_HEIGHT_M

        # Orientation: yaw about baseplate z-axis
        rot = Rotation.from_euler('z', orientation_deg, degrees=True)
        q   = rot.as_quat()   # [x, y, z, w]
        pose_in_bp.orientation.x = q[0]
        pose_in_bp.orientation.y = q[1]
        pose_in_bp.orientation.z = q[2]
        pose_in_bp.orientation.w = q[3]

        # Transform to base_link
        pose_base = tf2_geometry_msgs.do_transform_pose(pose_in_bp, tf_bp)

        target = PoseStamped()
        target.header.stamp    = self.get_clock().now().to_msg()
        target.header.frame_id = 'base_link'
        target.pose            = pose_base
        return target

    def _approach_pose(self, pose: PoseStamped, clearance_m: float) -> PoseStamped:
        """Return a copy of pose shifted upward by clearance_m along base_link z."""
        approach = PoseStamped()
        approach.header = pose.header
        approach.pose   = Pose()
        approach.pose.position.x    = pose.pose.position.x
        approach.pose.position.y    = pose.pose.position.y
        approach.pose.position.z    = pose.pose.position.z + clearance_m
        approach.pose.orientation   = pose.pose.orientation
        return approach

    # -----------------------------------------------------------------------
    # Arm interface
    # -----------------------------------------------------------------------

    def _init_arm(self):
        """
        Initialize the arm controller.

        Replace the placeholder below with your MoveIt2 setup, e.g.:
            from moveit.planning import MoveItPy
            self.moveit = MoveItPy(node_name='build_planner')
            self.arm    = self.moveit.get_planning_component('manipulator')
            self.gripper = self.moveit.get_planning_component('gripper')
        """
        self.get_logger().warn(
            '_init_arm() is a placeholder — wire up MoveIt2 or your arm controller here.'
        )

    def _move_to(self, target: PoseStamped) -> bool:
        """
        Command the arm to move its tool center point to target.

        Replace with your MoveIt2 call, e.g.:
            self.arm.set_goal_state(pose_stamped_msg=target, pose_link='tool0')
            plan = self.arm.plan()
            return self.arm.execute(plan)
        """
        self.get_logger().info(
            f'    [ARM] move_to ({target.pose.position.x:.3f}, '
            f'{target.pose.position.y:.3f}, {target.pose.position.z:.3f})'
        )
        return True   # placeholder

    def _open_gripper(self) -> bool:
        """Open the gripper. Replace with your gripper controller call."""
        self.get_logger().info('    [GRIPPER] open')
        return True   # placeholder

    def _close_gripper(self) -> bool:
        """Close the gripper. Replace with your gripper controller call."""
        self.get_logger().info('    [GRIPPER] close')
        return True   # placeholder

    def _pick(self, source_pose: Pose) -> bool:
        """
        Pick the brick at source_pose (Pose in base_link).

        Sequence:
          1. Pre-grasp hover (above the brick)
          2. Open gripper
          3. Descend to grasp pose
          4. Close gripper
          5. Retract upward
        """
        # Wrap bare Pose in a PoseStamped so _approach_pose / _move_to work uniformly
        stamped = PoseStamped()
        stamped.header.stamp    = self.get_clock().now().to_msg()
        stamped.header.frame_id = 'base_link'
        stamped.pose            = source_pose

        pre_grasp = self._approach_pose(stamped, APPROACH_CLEARANCE_M)
        retract   = self._approach_pose(stamped, RETRACT_CLEARANCE_M)

        return (
            self._move_to(pre_grasp)   and
            self._open_gripper()       and
            self._move_to(stamped)     and
            self._close_gripper()      and
            self._move_to(retract)
        )

    def _place(self, target: PoseStamped) -> bool:
        """
        Place the held brick at target (PoseStamped in base_link).

        Sequence:
          1. Pre-place hover (above the target slot)
          2. Descend to place pose
          3. Open gripper (release brick)
          4. Retract upward
        """
        pre_place = self._approach_pose(target, APPROACH_CLEARANCE_M)
        retract   = self._approach_pose(target, RETRACT_CLEARANCE_M)

        return (
            self._move_to(pre_place)  and
            self._move_to(target)     and
            self._open_gripper()      and
            self._move_to(retract)
        )

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def _wait_for_baseplate(self, timeout_sec: float = 30.0) -> bool:
        """
        Block until baseplate_frame TF is available.
        Returns True on success, False on timeout.
        """
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            try:
                self.tf_buffer.lookup_transform(
                    'base_link', 'baseplate_frame', rclpy.time.Time()
                )
                self.get_logger().info('baseplate_frame detected.')
                return True
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                time.sleep(0.5)

        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = BuildPlannerNode()

    # Spin in a background thread so detection callbacks fire while run() blocks
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
