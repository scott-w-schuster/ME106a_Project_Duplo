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
      shape:       [2, 4]    # [rows, cols] in brick's local frame
      position:    [0, 0]    # [row, col] of brick's corner stud on baseplate grid
      orientation: 0         # degrees relative to baseplate x-axis (0 or 90)
      layer:       0         # 0 = directly on baseplate
      height_type: normal    # half | normal | tall  (default: normal)

Gripper geometry:
  1×2 stud footprint — each jaw has one half-circle slot matching a Duplo stud.
  The TCP is commanded to the centroid of the brick's placed footprint so the
  gripper naturally centres on the nearest 1×2 pair of studs.
"""

import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration

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
STUD_PITCH_M   = 0.016   # 16 mm between stud centres

BRICK_HEIGHTS = {
    'half':   0.0096,    #  9.6 mm — Duplo flat / plate
    'normal': 0.0192,    # 19.2 mm — standard Duplo brick
    'tall':   0.0384,    # 38.4 mm — 2× standard (confirmed)
}

# ---------------------------------------------------------------------------
# Gripper geometry
# ---------------------------------------------------------------------------
# 1×2 stud footprint: one half-circle jaw slot per stud, spaced 1 stud apart.
# The TCP is positioned at the centroid of the brick's placed footprint.
GRIPPER_ROWS = 1
GRIPPER_COLS = 2

# ---------------------------------------------------------------------------
# TF lookup timeout
# ---------------------------------------------------------------------------
# Using rclpy.time.Time() (== time 0) requests the latest available transform,
# which works for both static and dynamic TFs.  Adding a timeout lets the call
# block and wait if the transform hasn't arrived yet — critical on startup when
# a static_transform_publisher may not have sent its first message yet.
TF_TIMEOUT_SEC = 1.0

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
        self.build_layers, self.baseplate_cfg, self.layer_base_z = \
            self._load_build_spec(spec_path)
        total = sum(len(v) for v in self.build_layers.values())
        self.get_logger().info(
            f'Loaded {total} bricks across {len(self.build_layers)} layers from {spec_path}'
        )

        # Validate all placements against the baseplate shape before we start
        errors = self._validate_build()
        if errors:
            for e in errors:
                self.get_logger().error(f'Build spec error: {e}')
            raise ValueError(f'{len(errors)} invalid placement(s) in build spec — fix before running.')

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

    def _load_build_spec(self, path: str):
        """
        Load YAML build spec.
        Returns:
            layers:        {layer_index: [brick_dict, ...]} sorted lowest to highest
            baseplate_cfg: dict with 'rows', 'cols', and 'invalid_studs' set of (r, c)
            layer_base_z:  {layer_index: base_z_m} — baseplate-frame Z at bottom of
                           each layer, computed from the tallest brick in each layer below
        """
        with open(path, 'r') as f:
            spec = yaml.safe_load(f)

        # Parse baseplate shape
        bp = spec.get('baseplate', {})
        bp_rows = bp.get('rows', 16)
        bp_cols = bp.get('cols', 16)
        invalid = {tuple(c) for c in bp.get('missing_corners', [])}
        baseplate_cfg = {
            'rows':          bp_rows,
            'cols':          bp_cols,
            'invalid_studs': invalid,
        }

        # Group bricks by layer; fill in defaults
        layers = {}
        for brick in spec['bricks']:
            layer = brick['layer']
            brick.setdefault('orientation',  0)
            brick.setdefault('height_type', 'normal')
            layers.setdefault(layer, []).append(brick)
        layers = dict(sorted(layers.items()))

        # Cumulative base Z for each layer (bottom of that layer in baseplate frame)
        layer_base_z = {}
        cumulative_z = 0.0
        for layer_idx in sorted(layers.keys()):
            layer_base_z[layer_idx] = cumulative_z
            layer_height = max(
                BRICK_HEIGHTS.get(b['height_type'], BRICK_HEIGHTS['normal'])
                for b in layers[layer_idx]
            )
            cumulative_z += layer_height

        return layers, baseplate_cfg, layer_base_z

    def _validate_build(self) -> list:
        """
        Check every brick placement against the baseplate boundaries and invalid studs.
        Returns a list of error strings (empty means the spec is valid).
        """
        bp       = self.baseplate_cfg
        bp_rows  = bp['rows']
        bp_cols  = bp['cols']
        invalid  = bp['invalid_studs']
        errors   = []

        for layer_idx, bricks in self.build_layers.items():
            for brick in bricks:
                color     = brick['color']
                br, bc    = brick['shape']       # brick rows, brick cols
                row, col  = brick['position']

                # Check every stud the brick will occupy
                for r in range(row, row + br):
                    for c in range(col, col + bc):
                        if r < 0 or r >= bp_rows or c < 0 or c >= bp_cols:
                            errors.append(
                                f'Layer {layer_idx} {color} {br}x{bc} at ({row},{col}): '
                                f'stud ({r},{c}) is outside the baseplate ({bp_rows}x{bp_cols}).'
                            )
                        elif (r, c) in invalid:
                            errors.append(
                                f'Layer {layer_idx} {color} {br}x{bc} at ({row},{col}): '
                                f'stud ({r},{c}) is a missing corner on this baseplate.'
                            )
        return errors

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
                height_type = brick_spec['height_type']         # half | normal | tall

                slot = (row, col, layer_idx)
                if slot in self.placed:
                    self.get_logger().info(f'  Slot {slot} already placed, skipping.')
                    continue

                self.get_logger().info(
                    f'  Placing {color} {shape[0]}x{shape[1]} [{height_type}] '
                    f'→ grid ({row},{col}) layer {layer_idx} orientation {orientation}°'
                )

                success = False
                for attempt in range(3):
                    # Find a matching brick on the table
                    source_pose = self._find_brick_on_table(color, shape, height_type)
                    if source_pose is None:
                        self.get_logger().warn(
                            f'  No {color} {shape} on table (attempt {attempt+1}/3) — waiting...'
                        )
                        time.sleep(1.5)
                        continue

                    # Compute where to put it on the baseplate
                    target = self._compute_placement_pose(
                        row, col, layer_idx, orientation, shape, height_type
                    )
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

    def _find_brick_on_table(self, color: str, shape: tuple,
                              height_type: str) -> Pose:
        """
        Search the latest detections for a brick matching color, shape, AND
        height_type.  All three must agree so the planner doesn't pick a normal-
        height brick when a tall one is specified (or vice-versa).

        Returns the centroid Pose in base_link, or None if not found.
        """
        with self._detections_lock:
            pairs = list(zip(self.latest_poses, self.latest_meta))

        for pose, meta in pairs:
            if (meta['color']                    == color       and
                    tuple(meta['shape'])         == shape       and
                    meta.get('height_type', 'normal') == height_type):
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

    def _grasp_offset_m(self, shape: tuple, orientation_deg: float) -> tuple:
        """
        Returns (dx, dy) offset in meters from the brick's corner stud to the
        gripper TCP, expressed in the baseplate frame (x = col, y = row).

        The gripper is 1×2 studs.  We command the TCP to the centroid of the
        brick's placed footprint so the jaw slots naturally centre on the nearest
        pair of studs along the gripper's closing axis.

        shape:           (brick_rows, brick_cols) in the brick's local frame
        orientation_deg: yaw of brick's long axis relative to baseplate x-axis
        """
        br, bc = shape
        # After rotation, the brick's row/col dimensions may swap on the baseplate
        if int(orientation_deg) % 180 == 90:
            placed_rows, placed_cols = bc, br
        else:
            placed_rows, placed_cols = br, bc

        dy = (placed_rows - 1) / 2.0 * STUD_PITCH_M   # baseplate y (row direction)
        dx = (placed_cols - 1) / 2.0 * STUD_PITCH_M   # baseplate x (col direction)
        return dx, dy

    def _compute_placement_pose(self, row: int, col: int, layer: int,
                                 orientation_deg: float, shape: tuple,
                                 height_type: str) -> PoseStamped:
        """
        Compute the TCP target pose in base_link for placing a brick.

        baseplate_frame origin = corner of stud (0,0).
          TCP x = col * STUD_PITCH + grasp_dx   (centroid of brick footprint)
          TCP y = row * STUD_PITCH + grasp_dy
          TCP z = layer_base_z + brick_height    (top of brick = stud level)

        orientation_deg: yaw of the brick's long axis relative to baseplate x-axis.
        height_type:     'half' | 'normal' | 'tall'
        """
        try:
            tf_bp = self.tf_buffer.lookup_transform(
                'base_link', 'baseplate_frame',
                rclpy.time.Time(),
                timeout=Duration(seconds=TF_TIMEOUT_SEC),
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'baseplate_frame lookup failed: {e}')
            return None

        dx, dy   = self._grasp_offset_m(shape, orientation_deg)
        brick_h  = BRICK_HEIGHTS.get(height_type, BRICK_HEIGHTS['normal'])
        base_z   = self.layer_base_z.get(layer, 0.0)

        # Position in baseplate frame — TCP at stud level of placed brick
        pose_in_bp = Pose()
        pose_in_bp.position.x = col * STUD_PITCH_M + dx
        pose_in_bp.position.y = row * STUD_PITCH_M + dy
        pose_in_bp.position.z = base_z + brick_h

        # Orientation: gripper yaw matches brick orientation
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
                # Short per-attempt timeout so rclpy.ok() is checked regularly.
                # rclpy.time.Time() == time 0 → "latest available", correct for
                # both static and dynamic TFs (including StaticTransformBroadcaster).
                self.tf_buffer.lookup_transform(
                    'base_link', 'baseplate_frame',
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.5),
                )
                self.get_logger().info('baseplate_frame detected.')
                return True
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                pass   # keep waiting

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
