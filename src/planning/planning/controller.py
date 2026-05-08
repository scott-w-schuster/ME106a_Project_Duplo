import threading
import numpy as np
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory

VEL_LIMITS = np.array([2.09, 2.09, 3.14, 3.14, 3.14, 3.14])


class UR7eTrajectoryController:

    def __init__(self, node):
        self._node = node
        self._ac = ActionClient(
            node,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
        )

    def execute_joint_trajectory(self, joint_traj: JointTrajectory, timeout=60.0) -> bool:
        self._ac.wait_for_server()

        done    = threading.Event()
        success = [False]

        def on_result(future):
            try:
                res = future.result().result
                if res.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
                    success[0] = True
                else:
                    self._node.get_logger().error(f'Trajectory error code: {res.error_code}')
            except Exception as e:
                self._node.get_logger().error(f'Trajectory result error: {e}')
            done.set()

        def on_goal_sent(future):
            gh = future.result()
            if not gh.accepted:
                self._node.get_logger().error('Trajectory goal rejected by controller')
                done.set()
                return
            gh.get_result_async().add_done_callback(on_result)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj
        self._ac.send_goal_async(goal).add_done_callback(on_goal_sent)
        done.wait(timeout=timeout)
        return success[0]


class PIDJointVelocityController:

    def __init__(self, node, Kp, Ki, Kd):
        self._node       = node
        self.Kp          = np.asarray(Kp, dtype=float)
        self.Ki          = np.asarray(Ki, dtype=float)
        self.Kd          = np.asarray(Kd, dtype=float)
        self._integral   = np.zeros(6)
        self._prev_error = None
        self._prev_time  = None

    def get_name(self) -> str:
        return 'PIDJointVelocityController'

    def reset(self):
        self._integral   = np.zeros(6)
        self._prev_error = None
        self._prev_time  = None

    def step_control(self, target_position, target_velocity,
                     current_position, current_velocity) -> np.ndarray:
        now   = self._node.get_clock().now().nanoseconds * 1e-9
        error = np.asarray(target_position) - np.asarray(current_position)

        if self._prev_time is None or self._prev_error is None:
            dt      = 0.0
            d_error = np.zeros(6)
        else:
            dt      = now - self._prev_time
            d_error = (error - self._prev_error) / dt if dt > 0 else np.zeros(6)

        if dt > 0:
            self._integral += error * dt

        self._prev_error = error.copy()
        self._prev_time  = now

        cmd = (
            self.Kp * error
            + self.Ki * self._integral
            + self.Kd * d_error
            + np.asarray(target_velocity)
        )
        return np.clip(cmd, -VEL_LIMITS, VEL_LIMITS)
