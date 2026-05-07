import threading

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy

from std_srvs.srv import Trigger
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from planning.ik import IKPlanner

# Arm moves to this pose after grasping so the planning node can verify pickup
CHECK_X = 0.095
CHECK_Y = 0.418
CHECK_Z = 0.188

LATCH = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

SCAN_POSES = [
   (0.095,  0.408, 0.288),
   (0.195,  0.408, 0.288),
   (0.295,  0.408, 0.288),
   (-0.095,  0.408, 0.288),
   (-0.195,  0.408, 0.288),
   (-0.295,  0.408, 0.288),
   (-0.395,  0.408, 0.288),
]



class UR7e_CubeGrasp(Node):

    PRE_GRASP_OFFSET = 0.185
    GRASP_OFFSET     = 0.103
    PRE_PLACE_OFFSET = 0.185
    PLACE_OFFSET     = 0.1025

    def __init__(self):
        super().__init__('cube_grasp')

        cb = ReentrantCallbackGroup()

        # ── Pose inputs from planning node ────────────────────────────────────
        self.pick_pose  = None
        self.place_pose = None
        self.create_subscription(PoseStamped, '/pick_pose',  self._on_pick_pose,  LATCH, callback_group=cb)
        self.create_subscription(PoseStamped, '/place_pose', self._on_place_pose, LATCH, callback_group=cb)

        # ── Joint state ───────────────────────────────────────────────────────
        self.joint_state = None
        self._js_lock = threading.Lock()
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 1, callback_group=cb)

        # ── Service servers (planning node calls these in order) ──────────────
        self._scan_idx = 0
        self.create_service(Trigger, '/move_to_pregrasp',     self._handle_pregrasp,   callback_group=cb)
        self.create_service(Trigger, '/grasp',                self._handle_grasp,      callback_group=cb)
        self.create_service(Trigger, '/move_to_check',        self._handle_check,      callback_group=cb)
        self.create_service(Trigger, '/preplace_and_place',   self._handle_place,      callback_group=cb)
        self.create_service(Trigger, '/next_scan_pose',       self._handle_scan_pose,  callback_group=cb)

        # ── Hardware ──────────────────────────────────────────────────────────
        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
            callback_group=cb,
        )
        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper', callback_group=cb)

        self.ik_planner = IKPlanner()

    # ── Subscriptions ─────────────────────────────────────────────────────────

    def _on_pick_pose(self, msg):
        self.pick_pose = msg

    def _on_place_pose(self, msg):
        self.place_pose = msg

    def _on_joint_state(self, msg):
        with self._js_lock:
            self.joint_state = msg

    # ── Service handlers (block until motion complete) ────────────────────────

    def _handle_scan_pose(self, _request, response):
        x, y, z = SCAN_POSES[self._scan_idx % len(SCAN_POSES)]
        self._scan_idx += 1
        print(f'[SCAN] Moving to ({x:.3f}, {y:.3f}, {z:.3f})', flush=True)
        with self._js_lock:
            js = self.joint_state
        if js is None:
            print('[SCAN] FAIL — no joint state received yet', flush=True)
            return self._fail(response, 'no joint state')
        print(f'[SCAN] Joint state OK, calling IK...', flush=True)
        ok = self._move_to(x, y, z, vel=0.05, accel=0.05)
        print(f'[SCAN] Move result: {ok}', flush=True)
        return self._result(response, ok, 'scan_pose')

    def _handle_pregrasp(self, _request, response):
        if self.pick_pose is None:
            return self._fail(response, 'No pick pose received')
        p = self.pick_pose.pose
        ok = self._move_to(p.position.x, p.position.y, p.position.z + self.PRE_GRASP_OFFSET,
                           p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
                           vel=0.1, accel=0.1)
        return self._result(response, ok, 'pregrasp')

    def _handle_grasp(self, _request, response):
        if self.pick_pose is None:
            return self._fail(response, 'No pick pose received')
        p = self.pick_pose.pose
        ox, oy, oz, ow = p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w

        ok = (self._toggle_gripper()
              and self._move_to(p.position.x, p.position.y, p.position.z + self.GRASP_OFFSET,
                                ox, oy, oz, ow, vel=0.05, accel=0.05)
              and self._toggle_gripper()
              and self._move_to(p.position.x, p.position.y, p.position.z + self.PRE_GRASP_OFFSET,
                                ox, oy, oz, ow, vel=0.1, accel=0.1))
        return self._result(response, ok, 'grasp')

    def _handle_check(self, _request, response):
        ok = self._move_to(CHECK_X, CHECK_Y, CHECK_Z, vel=0.15, accel=0.15)
        return self._result(response, ok, 'move_to_check')

    def _handle_place(self, _request, response):
        if self.place_pose is None:
            return self._fail(response, 'No place pose received')
        p = self.place_pose.pose
        ox, oy, oz, ow = p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w

        ok = (self._move_to(p.position.x, p.position.y, p.position.z + self.PRE_PLACE_OFFSET,
                            ox, oy, oz, ow, vel=0.1, accel=0.1)
              and self._move_to(p.position.x, p.position.y, p.position.z + self.PLACE_OFFSET,
                                ox, oy, oz, ow, vel=0.05, accel=0.05)
              and self._toggle_gripper()
              and self._move_to(p.position.x, p.position.y, p.position.z + self.PRE_PLACE_OFFSET,
                                ox, oy, oz, ow, vel=0.1, accel=0.1))
        return self._result(response, ok, 'place')

    # ── Motion helpers ────────────────────────────────────────────────────────

    def _move_to(self, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0,
                 vel=0.1, accel=0.1) -> bool:
        with self._js_lock:
            js = self.joint_state
        if js is None:
            self.get_logger().error('No joint state available')
            return False

        joint_sol = self.ik_planner.compute_ik(js, x, y, z, qx, qy, qz, qw)
        if joint_sol is None:
            return False

        traj = self.ik_planner.plan_to_joints(joint_sol, vel, accel)
        if traj is None:
            return False

        return self._execute_traj(traj.joint_trajectory)

    def _execute_traj(self, joint_traj) -> bool:
        done    = threading.Event()
        success = [False]

        def on_result(future):
            try:
                future.result().result
                success[0] = True
            except Exception as e:
                self.get_logger().error(f'Trajectory failed: {e}')
            done.set()

        def on_goal_sent(future):
            gh = future.result()
            if not gh.accepted:
                self.get_logger().error('Trajectory rejected')
                done.set()
                return
            gh.get_result_async().add_done_callback(on_result)

        self.exec_ac.wait_for_server()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj
        self.exec_ac.send_goal_async(goal).add_done_callback(on_goal_sent)
        done.wait(timeout=60.0)
        return success[0]

    def _toggle_gripper(self) -> bool:
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            return False
        done   = threading.Event()
        result = [None]

        def on_done(future):
            result[0] = future.result()
            done.set()

        self.gripper_cli.call_async(Trigger.Request()).add_done_callback(on_done)
        done.wait(timeout=5.0)
        self.get_logger().info('Gripper toggled')
        return result[0] is not None

    # ── Response helpers ──────────────────────────────────────────────────────

    def _result(self, response, ok, step):
        response.success = ok
        response.message = step if ok else f'{step} failed'
        if not ok:
            self.get_logger().error(f'{step} failed')
        return response

    def _fail(self, response, msg):
        response.success = False
        response.message = msg
        self.get_logger().error(msg)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(node.ik_planner)  # so IKPlanner futures get processed
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
