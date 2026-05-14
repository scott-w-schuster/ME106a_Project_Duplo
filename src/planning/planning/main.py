import argparse
import queue as _queue
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from planning.controller import UR7eTrajectoryController, PIDJointVelocityController

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

CHECK_X = 0.095
CHECK_Y = 0.418
CHECK_Z = 0.188

LATCH = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

SCAN_POSES = [
    ( 0.095,  0.408, 0.288),
    ( 0.195,  0.408, 0.288),
    ( 0.295,  0.408, 0.288),
    (-0.095,  0.408, 0.288),
    (-0.195,  0.408, 0.288),
    (-0.395,  0.408, 0.288),
]

Kp = 1.5  * np.array([0.4, 2.0, 1.7, 1.5, 2.0, 2.0])
Ki = 0.15 * np.array([1.4, 1.4, 1.4, 1.0, 0.6, 0.6])
Kd = 0.25 * np.array([2.0, 1.0, 2.0, 0.5, 0.8, 0.8])


class UR7e_CubeGrasp(Node):

    PRE_GRASP_OFFSET = 0.185
    GRASP_OFFSET     = 0.10
    PRE_PLACE_OFFSET = 0.185
    PLACE_OFFSET     = 0.10
    MOVE_DURATION    = 5.0
    GRASP_DURATION   = 3.0

    def __init__(self, controller_type: str = 'default'):
        super().__init__('cube_grasp')

        self.controller_type = controller_type
        cb = ReentrantCallbackGroup()

        self.pick_pose  = None
        self.place_pose = None
        self.create_subscription(PoseStamped, '/pick_pose',  self._on_pick_pose,  LATCH, callback_group=cb)
        self.create_subscription(PoseStamped, '/place_pose', self._on_place_pose, LATCH, callback_group=cb)

        self.joint_state   = None
        self._js_lock      = threading.Lock()
        self._js_recv_time = 0.0
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 1, callback_group=cb)

        self._scan_idx = 0
        self.create_service(Trigger, '/move_to_pregrasp',   self._handle_pregrasp,  callback_group=cb)
        self.create_service(Trigger, '/grasp',              self._handle_grasp,     callback_group=cb)
        self.create_service(Trigger, '/move_to_check',      self._handle_check,     callback_group=cb)
        self.create_service(Trigger, '/preplace_and_place', self._handle_place,     callback_group=cb)
        self.create_service(Trigger, '/next_scan_pose',     self._handle_scan_pose, callback_group=cb)

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper', callback_group=cb)

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info('Waiting for /compute_ik service...')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Still waiting for /compute_ik...')

        self.trajectory_controller = UR7eTrajectoryController(self)
        self.pid_controller        = PIDJointVelocityController(self, Kp, Ki, Kd)

        self._velocity_pub = self.create_publisher(
            Float64MultiArray, '/forward_velocity_controller/commands', 10,
            callback_group=cb,
        )

        self._cmd_queue   = _queue.Queue()
        self._motion_lock = threading.Lock()
        self._worker      = threading.Thread(target=self._drain_queue, daemon=True)
        self._worker.start()

        self.get_logger().info(f'UR7e_CubeGrasp ready — controller: {controller_type}')

    def _on_pick_pose(self, msg):
        self.pick_pose = msg

    def _on_place_pose(self, msg):
        self.place_pose = msg

    def _on_joint_state(self, msg):
        with self._js_lock:
            self.joint_state   = msg
            self._js_recv_time = time.time()

    def _get_fresh_joint_state(self, max_age: float = 0.15, timeout: float = 2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._js_lock:
                if self.joint_state is not None and (time.time() - self._js_recv_time) < max_age:
                    return self.joint_state
            time.sleep(0.01)
        self.get_logger().error('Fresh joint state unavailable — camera DDS load?')
        return None

    def compute_ik(self, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0):
        js = self._get_fresh_joint_state()
        if js is None:
            return None

        pose = PoseStamped()
        pose.header.frame_id    = 'base_link'
        pose.pose.position.x    = x
        pose.pose.position.y    = y
        pose.pose.position.z    = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        req = GetPositionIK.Request()
        req.ik_request.group_name              = 'ur_manipulator'
        req.ik_request.robot_state.joint_state = js
        req.ik_request.ik_link_name            = 'wrist_3_link'
        req.ik_request.pose_stamped            = pose
        req.ik_request.timeout                 = Duration(sec=2)
        req.ik_request.avoid_collisions        = True

        done   = threading.Event()
        result = [None]

        def _cb(future):
            result[0] = future.result()
            done.set()

        self.ik_client.call_async(req).add_done_callback(_cb)

        if not done.wait(timeout=5.0):
            self.get_logger().error('IK service timed out')
            return None

        if result[0] is None:
            self.get_logger().error('IK service returned None')
            return None

        if result[0].error_code.val != result[0].error_code.SUCCESS:
            self.get_logger().error(f'IK failed, error code: {result[0].error_code.val}')
            return None

        return result[0].solution.joint_state

    def _extract_positions(self, joint_sol) -> list | None:
        js_dict = dict(zip(joint_sol.name, joint_sol.position))
        try:
            return [js_dict[n] for n in JOINT_NAMES]
        except KeyError as e:
            self.get_logger().error(f'IK solution missing joint: {e}')
            return None

    def _drain_queue(self):
        while True:
            fn, done, result = self._cmd_queue.get()
            if fn is None:
                break
            result[0] = fn()
            done.set()

    def _submit(self, fn) -> bool:
        if not self._motion_lock.acquire(blocking=False):
            print('[MAIN] *** DROPPED — motion already in progress ***', flush=True)
            self.get_logger().warn('Motion already in progress — dropping command')
            return False
        try:
            done   = threading.Event()
            result = [False]
            self._cmd_queue.put((fn, done, result))
            done.wait()
            return result[0]
        finally:
            self._motion_lock.release()

    def _handle_scan_pose(self, _req, response):
        x, y, z = SCAN_POSES[self._scan_idx % len(SCAN_POSES)]
        self._scan_idx += 1
        print(f'[MAIN] /next_scan_pose received → target ({x:.3f}, {y:.3f}, {z:.3f})', flush=True)
        self.get_logger().info(f'[SCAN] → ({x:.3f}, {y:.3f}, {z:.3f})')
        ok = self._submit(lambda: self._move(x, y, z))
        print(f'[MAIN] /next_scan_pose done — ok={ok}', flush=True)
        return self._result(response, ok, 'scan_pose')

    def _handle_pregrasp(self, _req, response):
        print('[MAIN] /move_to_pregrasp received', flush=True)
        if self.pick_pose is None:
            print('[MAIN] /move_to_pregrasp FAILED — no pick pose', flush=True)
            return self._fail(response, 'No pick pose received')
        p = self.pick_pose.pose
        print(f'[MAIN] pre-grasp target: ({p.position.x:.3f}, {p.position.y:.3f}, '
              f'{p.position.z + self.PRE_GRASP_OFFSET:.3f})', flush=True)
        ok = self._submit(lambda: self._move(
            p.position.x, p.position.y, p.position.z + self.PRE_GRASP_OFFSET,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
        ))
        print(f'[MAIN] /move_to_pregrasp done — ok={ok}', flush=True)
        return self._result(response, ok, 'pregrasp')

    def _handle_grasp(self, _req, response):
        print('[MAIN] /grasp received', flush=True)
        if self.pick_pose is None:
            print('[MAIN] /grasp FAILED — no pick pose', flush=True)
            return self._fail(response, 'No pick pose received')
        p  = self.pick_pose.pose
        ox, oy, oz, ow = p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
        print(f'[MAIN] grasp target: ({p.position.x:.3f}, {p.position.y:.3f}, '
              f'{p.position.z + self.GRASP_OFFSET:.3f})', flush=True)
        ok = self._submit(lambda: (
            self._toggle_gripper()
            and self._move(p.position.x, p.position.y, p.position.z + self.GRASP_OFFSET,
                           ox, oy, oz, ow, duration=self.GRASP_DURATION)
            and self._toggle_gripper()
            and self._move(p.position.x, p.position.y, p.position.z + self.PRE_GRASP_OFFSET,
                           ox, oy, oz, ow)
        ))
        print(f'[MAIN] /grasp done — ok={ok}', flush=True)
        return self._result(response, ok, 'grasp')

    def _handle_check(self, _req, response):
        print(f'[MAIN] /move_to_check received → ({CHECK_X}, {CHECK_Y}, {CHECK_Z})', flush=True)
        ok = self._submit(lambda: self._move(CHECK_X, CHECK_Y, CHECK_Z))
        print(f'[MAIN] /move_to_check done — ok={ok}', flush=True)
        return self._result(response, ok, 'move_to_check')

    def _handle_place(self, _req, response):
        print('[MAIN] /preplace_and_place received', flush=True)
        if self.place_pose is None:
            print('[MAIN] /preplace_and_place FAILED — no place pose', flush=True)
            return self._fail(response, 'No place pose received')
        p  = self.place_pose.pose
        ox, oy, oz, ow = p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
        print(f'[MAIN] place target: ({p.position.x:.3f}, {p.position.y:.3f}, '
              f'{p.position.z + self.PLACE_OFFSET:.3f})', flush=True)
        ok = self._submit(lambda: (
            self._move(p.position.x, p.position.y, p.position.z + self.PRE_PLACE_OFFSET,
                       ox, oy, oz, ow)
            and self._move(p.position.x, p.position.y, p.position.z + self.PLACE_OFFSET,
                           ox, oy, oz, ow, duration=self.GRASP_DURATION)
            and self._toggle_gripper()
            and self._move(p.position.x, p.position.y, p.position.z + self.PRE_PLACE_OFFSET,
                           ox, oy, oz, ow)
        ))
        print(f'[MAIN] /preplace_and_place done — ok={ok}', flush=True)
        return self._result(response, ok, 'place')

    def _move(self, x, y, z, qx=0.0, qy=1.0, qz=0.0, qw=0.0,
              duration: float | None = None) -> bool:
        if duration is None:
            duration = self.MOVE_DURATION

        print(f'[MAIN] _move → ({x:.3f}, {y:.3f}, {z:.3f}) dur={duration:.1f}s', flush=True)
        joint_sol = self.compute_ik(x, y, z, qx, qy, qz, qw)
        if joint_sol is None:
            print('[MAIN] _move FAILED — IK returned None', flush=True)
            return False

        target = self._extract_positions(joint_sol)
        if target is None:
            print('[MAIN] _move FAILED — could not extract joint positions', flush=True)
            return False

        if self.controller_type == 'pid':
            print('[MAIN] executing PID move', flush=True)
            return self._move_pid(target, duration)
        print('[MAIN] executing trajectory move', flush=True)
        return self._move_trajectory(target, duration)

    def _move_trajectory(self, target_positions: list, duration: float) -> bool:
        print(f'[MAIN] sending trajectory (dur={duration:.1f}s)...', flush=True)
        jt = JointTrajectory()
        jt.joint_names  = JOINT_NAMES
        jt.header.stamp = self.get_clock().now().to_msg()

        pt = JointTrajectoryPoint()
        pt.positions       = target_positions
        pt.velocities      = [0.0] * 6
        pt.time_from_start = Duration(
            sec    =int(duration),
            nanosec=int((duration % 1) * 1e9),
        )
        jt.points.append(pt)

        result = self.trajectory_controller.execute_joint_trajectory(
            jt, timeout=duration + 10.0)
        print(f'[MAIN] trajectory done — ok={result}', flush=True)
        return result

    def _move_pid(self, target_positions: list, duration: float,
                  control_hz: float = 10.0) -> bool:
        print(f'[MAIN] starting PID move (dur={duration:.1f}s, hz={control_hz})', flush=True)
        js = self._get_fresh_joint_state()
        if js is None:
            print('[MAIN] PID FAILED — no fresh joint state', flush=True)
            return False

        start_dict = dict(zip(js.name, js.position))
        try:
            start = np.array([start_dict[n] for n in JOINT_NAMES])
        except KeyError as e:
            self.get_logger().error(f'Joint state missing joint: {e}')
            return False

        target  = np.array(target_positions)
        ff_vel  = (target - start) / duration
        dt      = 1.0 / control_hz
        n_steps = max(1, int(duration * control_hz))
        vel_msg = Float64MultiArray()

        self.pid_controller.reset()

        for step in range(n_steps):
            step_start = time.time()
            alpha      = min(step * dt / duration, 1.0)
            ref_pos    = start + alpha * (target - start)

            js_now = self._get_fresh_joint_state(max_age=0.5)
            if js_now is None:
                break

            cur_dict = {n: (p, v)
                        for n, p, v in zip(js_now.name, js_now.position, js_now.velocity)}
            try:
                cur_pos = np.array([cur_dict[n][0] for n in JOINT_NAMES])
                cur_vel = np.array([cur_dict[n][1] for n in JOINT_NAMES])
            except KeyError:
                break

            cmd = self.pid_controller.step_control(ref_pos, ff_vel, cur_pos, cur_vel)
            vel_msg.data = cmd.tolist()
            self._velocity_pub.publish(vel_msg)

            remaining = dt - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

        vel_msg.data = [0.0] * 6
        self._velocity_pub.publish(vel_msg)
        print('[MAIN] PID move done', flush=True)
        return True

    def _toggle_gripper(self) -> bool:
        print('[MAIN] toggling gripper...', flush=True)
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            print('[MAIN] gripper FAILED — service unavailable', flush=True)
            self.get_logger().error('Gripper service unavailable')
            return False

        done   = threading.Event()
        result = [None]

        def _cb(future):
            result[0] = future.result()
            done.set()

        self.gripper_cli.call_async(Trigger.Request()).add_done_callback(_cb)
        done.wait(timeout=5.0)
        ok = result[0] is not None
        print(f'[MAIN] gripper toggle done — ok={ok}', flush=True)
        self.get_logger().info('Gripper toggled')
        return ok

    def _result(self, response, ok: bool, step: str):
        response.success = ok
        response.message = step if ok else f'{step} failed'
        if not ok:
            self.get_logger().error(f'{step} failed')
        return response

    def _fail(self, response, msg: str):
        response.success = False
        response.message = msg
        self.get_logger().error(msg)
        return response


def switch_controllers(to: str):
    if to == 'pid':
        deactivate = 'scaled_joint_trajectory_controller'
        activate   = 'forward_velocity_controller'
    else:
        deactivate = 'forward_velocity_controller'
        activate   = 'scaled_joint_trajectory_controller'

    cmd = [
        'ros2', 'control', 'switch_controllers',
        '--deactivate', deactivate,
        '--activate',   activate,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            print(f'[WARN] switch_controllers: {r.stderr.strip()}')
    except Exception as e:
        print(f'[ERROR] switch_controllers failed: {e}')


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--controller', '-c',
        default='pid',
        choices=['default', 'pid'],
    )
    parsed, _ = parser.parse_known_args()

    switch_controllers(parsed.controller)

    rclpy.init(args=args)
    node     = UR7e_CubeGrasp(controller_type=parsed.controller)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node._cmd_queue.put((None, None, None))
        switch_controllers('default')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
