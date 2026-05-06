import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

STUD_PITCH_M   = 0.008   # 1 stud  = 8 mm
BLOCK_HEIGHT_M = 0.0096  # 1 layer = 9.6 mm
GRID_SIZE      = 20
BLOCK_DIMS     = {'2x4': (2, 4), '2x2': (2, 2)}

LATCH = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def block_rot_to_quat(rotation_deg: float):
    """Gripper quaternion for a given block rotation (pointing down + yaw)."""
    q = R.from_euler('z', rotation_deg, degrees=True) * R.from_quat([0, 1, 0, 0])
    x, y, z, w = q.as_quat()
    return float(x), float(y), float(z), float(w)


class LEGOBuildPlanner(Node):

    def __init__(self):
        super().__init__('lego_build_planner')

        self.declare_parameter('build_plan_path', 'lego_build.json')
        path = self.get_parameter('build_plan_path').get_parameter_value().string_value
        with open(path) as f:
            self.build_sequence = json.load(f)
        self.get_logger().info(f'Loaded {len(self.build_sequence)} block(s)')
        self.current_step = 0

        # ── Pose publishers → main.py ─────────────────────────────────────────
        self.pick_pub  = self.create_publisher(PoseStamped, '/pick_pose',  LATCH)
        self.place_pub = self.create_publisher(PoseStamped, '/place_pose', LATCH)

        # ── Service clients → main.py ─────────────────────────────────────────
        self.pregrasp_cli = self.create_client(Trigger, '/move_to_pregrasp')
        self.grasp_cli    = self.create_client(Trigger, '/grasp')
        self.check_cli    = self.create_client(Trigger, '/move_to_check')
        self.place_cli    = self.create_client(Trigger, '/preplace_and_place')

        # Brief delay so main.py can finish starting up
        self._started = False
        self.create_timer(3.0, self._start)

    # ── Startup ───────────────────────────────────────────────────────────────

    def _start(self):
        if self._started:
            return
        self._started = True
        for cli in (self.pregrasp_cli, self.grasp_cli, self.check_cli, self.place_cli):
            cli.wait_for_service()
        self.get_logger().info('All services ready — starting build')
        self._execute_step()

    # ── Step entry point ──────────────────────────────────────────────────────

    def _execute_step(self):
        if self.current_step >= len(self.build_sequence):
            self.get_logger().info('Build complete!')
            return

        step = self.build_sequence[self.current_step]
        self.get_logger().info(
            f'Step {self.current_step + 1}/{len(self.build_sequence)}: '
            f'{step["color"]} {step["type"]} layer={step["layer"]} '
            f'grid=({step["grid_x"]},{step["grid_z"]}) rot={step["rotation_deg"]}°')

        # Detection stubs
        detected_bricks = self._detect_bricks()
        plate_pose      = self._detect_plate()

        if plate_pose is None:
            self.get_logger().error('Build plate not detected — aborting')
            return

        pick_pose = self._find_brick(detected_bricks, step['type'], step['color'])
        if pick_pose is None:
            self.get_logger().error(f'No {step["color"]} {step["type"]} found — skipping')
            self.current_step += 1
            self._execute_step()
            return

        # Bake gripper rotation into both poses so main.py can read it directly
        qx, qy, qz, qw = block_rot_to_quat(step['rotation_deg'])
        for pose in (pick_pose, self._grid_to_pose(step, plate_pose)):
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

        place_pose = self._grid_to_pose(step, plate_pose)
        place_pose.pose.orientation.x = qx
        place_pose.pose.orientation.y = qy
        place_pose.pose.orientation.z = qz
        place_pose.pose.orientation.w = qw

        self.pick_pub.publish(pick_pose)
        self.place_pub.publish(place_pose)

        # Kick off the async service chain
        self.pregrasp_cli.call_async(Trigger.Request()).add_done_callback(self._on_pregrasp)

    # ── Async service chain ───────────────────────────────────────────────────

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

    # ── Detection stubs ───────────────────────────────────────────────────────

    def _detect_bricks(self) -> list:
        """
        STUB — call /detect_bricks service.
        Should return list of dicts: {type, color, pose: PoseStamped}
        """
        self.get_logger().warn('STUB: _detect_bricks — returning []')
        return []

    def _detect_plate(self):
        """
        STUB — call /detect_build_plate service.
        Should return PoseStamped of plate centre in base_link frame, or None.
        """
        self.get_logger().warn('STUB: _detect_plate — returning None')
        return None

    def _verify_pickup(self) -> bool:
        """
        STUB — read camera while arm is at check pose to confirm block is held.
        """
        self.get_logger().warn('STUB: _verify_pickup — assuming True')
        return True

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_brick(self, detected: list, block_type: str, color: str):
        for b in detected:
            if b['type'] == block_type and b['color'] == color:
                return b['pose']
        return None

    def _grid_to_pose(self, step: dict, plate_pose: PoseStamped) -> PoseStamped:
        """Convert grid position to world-frame PoseStamped using plate origin."""
        w, d = BLOCK_DIMS[step['type']]
        if step['rotation_deg'] % 180 != 0:
            w, d = d, w

        x_off = (step['grid_x'] + (w - 1) / 2 - GRID_SIZE / 2 + 0.5) * STUD_PITCH_M
        y_off = (step['grid_z'] + (d - 1) / 2 - GRID_SIZE / 2 + 0.5) * STUD_PITCH_M
        z_off = step['layer'] * BLOCK_HEIGHT_M

        pose = PoseStamped()
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = plate_pose.pose.position.x + x_off
        pose.pose.position.y = plate_pose.pose.position.y + y_off
        pose.pose.position.z = plate_pose.pose.position.z + z_off
        return pose

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
