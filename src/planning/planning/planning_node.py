import json
import threading
import time

import numpy as np
import requests
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy

from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R

import tf2_ros
import tf2_geometry_msgs

STUD_PITCH_M = 0.016
BLOCK_DIMS   = {
    '1x2':                  (1, 2),
    '2x2':                  (2, 2),
    '2x4':                  (2, 4),
    '2x6':                  (2, 6),
    '2x4-fillet-top-both':  (2, 4),
    '2x4-fillet-bot-both':  (2, 4),
    '2x3-fillet-top':       (2, 3),
    '2x4-half':             (2, 4),
    '2x6-half':             (2, 6),
    '1x2-double':           (1, 2),
    '2x2-cylinder':         (2, 2),
}
BRICK_HEIGHTS = {'half': 0.0096, 'normal': 0.0192, 'tall': 0.0384}

TF_TIMEOUT_SEC       = 1.0
PICK_VERIFY_RADIUS_M = 0.05

LATCH = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def block_rot_to_quat(rotation_deg: float):
    q = R.from_euler('z', rotation_deg, degrees=True) * R.from_quat([0, 1, 0, 0])
    x, y, z, w = q.as_quat()
    return float(x), float(y), float(z), float(w)


class LEGOBuildPlanner(Node):

    def __init__(self):
        super().__init__('lego_build_planner')

        self.declare_parameter('jsonbin_id',  '69fa7a5d36566621a82c98c7')
        self.declare_parameter('jsonbin_key', '$2a$10$NCneQ9Ep5v3h/EScA.B7E../Xy0LcWBhFaGgSHjuoQ.NAoxjmZTJC')
        self.declare_parameter('build_plan_path', 'lego_build.json')
        self.build_sequence = self._fetch_build_plan()
        self.current_step = 0
        self._layer_base_z = self._compute_layer_heights()

        self._lock            = threading.Lock()
        self._latest_poses    = None
        self._latest_meta     = None
        self._last_pick_pose  = None
        self._last_pick_color = None

        self.pick_pub  = self.create_publisher(PoseStamped, '/pick_pose',  LATCH)
        self.place_pub = self.create_publisher(PoseStamped, '/place_pose', LATCH)

        self.pregrasp_cli = self.create_client(Trigger, '/move_to_pregrasp')
        self.grasp_cli    = self.create_client(Trigger, '/grasp')
        self.check_cli    = self.create_client(Trigger, '/move_to_check')
        self.place_cli    = self.create_client(Trigger, '/preplace_and_place')
        self.scan_cli     = self.create_client(Trigger, '/next_scan_pose')

        self.create_subscription(PoseArray, '/detected_bricks',      self._on_bricks,      10)
        self.create_subscription(String,    '/detected_bricks_meta', self._on_bricks_meta, 10)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._started = False
        self.create_timer(3.0, self._start)

    def _fetch_build_plan(self) -> list:
        bin_id = self.get_parameter('jsonbin_id').get_parameter_value().string_value
        key    = self.get_parameter('jsonbin_key').get_parameter_value().string_value
        try:
            resp = requests.get(
                f'https://api.jsonbin.io/v3/b/{bin_id}/latest',
                headers={'X-Master-Key': key},
                timeout=10.0,
            )
            resp.raise_for_status()
            sequence = resp.json()['record']['build_sequence']
            self.get_logger().info(f'Fetched {len(sequence)} block(s) from JSONBin.')
            return sequence
        except Exception as e:
            self.get_logger().warn(f'JSONBin fetch failed ({e}), falling back to local file.')
            path = self.get_parameter('build_plan_path').get_parameter_value().string_value
            try:
                with open(path) as f:
                    sequence = json.load(f)
                self.get_logger().info(f'Loaded {len(sequence)} block(s) from local file.')
                return sequence
            except FileNotFoundError:
                self.get_logger().error(f'Local file {path} not found. No build plan loaded.')
                return []

    def _compute_layer_heights(self) -> dict:
        layers = {}
        for step in self.build_sequence:
            idx = step.get('layer', 0)
            layers.setdefault(idx, []).append(step)
        base_z, result = 0.0, {}
        for idx in sorted(layers):
            result[idx] = base_z
            base_z += max(
                BRICK_HEIGHTS.get(s.get('height_type', 'normal'), BRICK_HEIGHTS['normal'])
                for s in layers[idx]
            )
        return result

    def _on_bricks(self, msg: PoseArray):
        with self._lock:
            self._latest_poses = msg

    def _on_bricks_meta(self, msg: String):
        with self._lock:
            self._latest_meta = json.loads(msg.data)

    def _start(self):
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._start_worker, daemon=True).start()

    def _start_worker(self):
        import traceback
        try:
            for cli in (self.pregrasp_cli, self.grasp_cli, self.check_cli, self.place_cli, self.scan_cli):
                cli.wait_for_service()
            self.get_logger().info('All services ready — scanning workspace...')
            if not self._scan_workspace():
                self.get_logger().error('Baseplate not detected after full scan — aborting.')
                return
            self._execute_step()
        except Exception as e:
            self.get_logger().error(f'_start_worker crashed: {e}\n{traceback.format_exc()}')

    def _scan_workspace(self) -> bool:
        for i in range(12):
            self.get_logger().info(f'Scan pose {i + 1}/12 — moving arm...')
            done = threading.Event()
            resp = [None]

            def _cb(future, _r=resp, _d=done):
                try:
                    _r[0] = future.result()
                except Exception as e:
                    self.get_logger().error(f'Scan service error: {e}')
                _d.set()

            self.scan_cli.call_async(Trigger.Request()).add_done_callback(_cb)
            if not done.wait(timeout=20.0):
                self.get_logger().warn(f'Scan pose {i + 1} timed out')
            elif resp[0] is not None and not resp[0].success:
                self.get_logger().warn(
                    f'Scan pose {i + 1} failed: {resp[0].message} '
                    f'(IK or controller issue)')

            time.sleep(1.0)
            try:
                self.tf_buffer.lookup_transform(
                    'base_link', 'baseplate_frame',
                    rclpy.time.Time(), timeout=Duration(seconds=0.3))
                self.get_logger().info('Baseplate found during scan.')
                return True
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                self.get_logger().info('Baseplate not yet visible — continuing scan...')
        return self._wait_for_baseplate(timeout_sec=10.0)

    def _wait_for_baseplate(self, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            try:
                self.tf_buffer.lookup_transform(
                    'base_link', 'baseplate_frame',
                    rclpy.time.Time(), timeout=Duration(seconds=0.5))
                self.get_logger().info('baseplate_frame detected.')
                return True
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                pass
        return False

    def _execute_step(self):
        if self.current_step >= len(self.build_sequence):
            self.get_logger().info('Build complete!')
            return

        step = self.build_sequence[self.current_step]
        self.get_logger().info(
            f'Step {self.current_step + 1}/{len(self.build_sequence)}: '
            f'{step["color"]} {step["type"]} layer={step.get("layer", 0)} '
            f'grid=({step["grid_x"]},{step["grid_z"]}) rot={step["rotation_deg"]}°'
        )

        detected = self._detect_bricks()
        pick_pose = self._find_brick(detected, step['type'], step['color'])
        if pick_pose is None:
            self.get_logger().error(f'No {step["color"]} {step["type"]} found — skipping')
            self.current_step += 1
            self._execute_step()
            return

        place_pose = self._grid_to_pose(step)
        if place_pose is None:
            self.get_logger().error('Cannot compute placement pose — aborting step')
            self.current_step += 1
            self._execute_step()
            return

        qx, qy, qz, qw = block_rot_to_quat(step['rotation_deg'])
        for ps in (pick_pose, place_pose):
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw

        self._last_pick_pose  = pick_pose
        self._last_pick_color = step['color']

        self.pick_pub.publish(pick_pose)
        self.place_pub.publish(place_pose)
        self.pregrasp_cli.call_async(Trigger.Request()).add_done_callback(self._on_pregrasp)

    def _on_pregrasp(self, future):
        if not self._ok(future, 'pregrasp'):
            return
        self.grasp_cli.call_async(Trigger.Request()).add_done_callback(self._on_grasp)

    def _on_grasp(self, future):
        if not self._ok(future, 'grasp'):
            return
        self.check_cli.call_async(Trigger.Request()).add_done_callback(self._on_check)

    def _on_check(self, future):
        if not self._ok(future, 'move_to_check'):
            return
        if not self._verify_pickup():
            self.get_logger().error('Pickup verification failed — retrying step')
            self._execute_step()
            return
        self.place_cli.call_async(Trigger.Request()).add_done_callback(self._on_place)

    def _on_place(self, future):
        if not self._ok(future, 'place'):
            return
        self.current_step += 1
        self._execute_step()

    def _detect_bricks(self) -> list:
        with self._lock:
            poses = self._latest_poses
            meta  = self._latest_meta
        if poses is None or meta is None:
            self.get_logger().warn('No brick detections received yet.')
            return []
        result = []
        for pose, m in zip(poses.poses, meta):
            rows, cols = m['shape']
            brick_type = f'{min(rows, cols)}x{max(rows, cols)}'
            if brick_type not in BLOCK_DIMS:
                continue
            ps = PoseStamped()
            ps.header = poses.header
            ps.pose   = pose
            result.append({'type': brick_type, 'color': m['color'], 'pose': ps})
        return result

    def _verify_pickup(self) -> bool:
        time.sleep(0.3)
        if self._last_pick_pose is None:
            return True
        with self._lock:
            poses = self._latest_poses
            meta  = self._latest_meta
        if poses is None or meta is None:
            return True
        pick_xy = np.array([
            self._last_pick_pose.pose.position.x,
            self._last_pick_pose.pose.position.y,
        ])
        for pose, m in zip(poses.poses, meta):
            if m.get('color') != self._last_pick_color:
                continue
            if np.linalg.norm(np.array([pose.position.x, pose.position.y]) - pick_xy) \
                    < PICK_VERIFY_RADIUS_M:
                self.get_logger().warn('Brick still detected at pick location.')
                return False
        return True

    def _find_brick(self, detected: list, block_type: str, color: str):
        for b in detected:
            if b['type'] == block_type and b['color'] == color:
                return b['pose']
        return None

    def _grid_to_pose(self, step: dict) -> PoseStamped:
        try:
            tf_bp = self.tf_buffer.lookup_transform(
                'base_link', 'baseplate_frame',
                rclpy.time.Time(), timeout=Duration(seconds=TF_TIMEOUT_SEC))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'baseplate_frame lookup failed: {e}')
            return None

        w, d = BLOCK_DIMS[step['type']]
        if int(step['rotation_deg']) % 180 == 90:
            w, d = d, w

        layer       = step.get('layer', 0)
        height_type = step.get('height_type', 'normal')
        base_z      = self._layer_base_z.get(layer, 0.0)
        brick_h     = BRICK_HEIGHTS.get(height_type, BRICK_HEIGHTS['normal'])

        pose_bp = Pose()
        pose_bp.position.x = (step['grid_x'] + (w - 1) / 2.0) * STUD_PITCH_M
        pose_bp.position.y = (step['grid_z'] + (d - 1) / 2.0) * STUD_PITCH_M
        pose_bp.position.z = base_z + brick_h
        pose_bp.orientation.w = 1.0

        ps = PoseStamped()
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.header.frame_id = 'base_link'
        ps.pose            = tf2_geometry_msgs.do_transform_pose(pose_bp, tf_bp)
        return ps

    def _ok(self, future, step_name: str) -> bool:
        try:
            resp = future.result()
            if resp.success:
                return True
            self.get_logger().error(f'{step_name} failed: {resp.message}')
        except Exception as e:
            self.get_logger().error(f'{step_name} service error: {e}')
        return False


def main(args=None):
    rclpy.init(args=args)
    node = LEGOBuildPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
