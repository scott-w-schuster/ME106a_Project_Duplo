import sys
import threading

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import PositionIKRequest, Constraints, JointConstraint
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration


class IKPlanner(Node):
    def __init__(self):
        super().__init__('ik_planner')

        self.declare_parameter('ee_link', 'tool0')
        self._ee_link = self.get_parameter('ee_link').get_parameter_value().string_value

        self.ik_client   = self.create_client(GetPositionIK,  '/compute_ik')
        self.plan_client = self.create_client(GetMotionPlan,   '/plan_kinematic_path')

        for srv, name in [(self.ik_client, 'compute_ik'),
                          (self.plan_client, 'plan_kinematic_path')]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for /{name} service...')

    def _wait_future(self, future, timeout=10.0) -> bool:
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        return done.wait(timeout=timeout)

    def compute_ik(self, current_joint_state, x, y, z,
                   qx=0.0, qy=1.0, qz=0.0, qw=0.0):
        pose = PoseStamped()
        pose.header.frame_id    = 'base_link'
        pose.pose.position.x    = x
        pose.pose.position.y    = y
        pose.pose.position.z    = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name              = 'ur_manipulator'
        ik_req.ik_request.ik_link_name            = self._ee_link
        ik_req.ik_request.avoid_collisions        = True
        ik_req.ik_request.timeout                 = Duration(sec=5)
        ik_req.ik_request.robot_state.joint_state = current_joint_state
        ik_req.ik_request.pose_stamped            = pose

        future = self.ik_client.call_async(ik_req)
        if not self._wait_future(future):
            self.get_logger().error('IK service timed out.')
            return None

        result = future.result()
        if result is None:
            self.get_logger().error('IK service failed.')
            return None
        if result.error_code.val != result.error_code.SUCCESS:
            self.get_logger().error(f'IK failed, code: {result.error_code.val}')
            return None

        self.get_logger().info('IK solution found.')
        return result.solution.joint_state

    def plan_to_joints(self, target_joint_state, velocity_scale=0.1, accel_scale=0.1):
        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name                    = 'ur_manipulator'
        req.motion_plan_request.allowed_planning_time         = 5.0
        req.motion_plan_request.planner_id                    = 'RRTConnectkConfigDefault'
        req.motion_plan_request.max_velocity_scaling_factor     = velocity_scale
        req.motion_plan_request.max_acceleration_scaling_factor = accel_scale

        goal_constraints = Constraints()
        for name, pos in zip(target_joint_state.name, target_joint_state.position):
            goal_constraints.joint_constraints.append(
                JointConstraint(
                    joint_name=name,
                    position=pos,
                    tolerance_above=0.01,
                    tolerance_below=0.01,
                    weight=1.0,
                )
            )

        req.motion_plan_request.goal_constraints.append(goal_constraints)
        future = self.plan_client.call_async(req)
        if not self._wait_future(future):
            self.get_logger().error('Planning service timed out.')
            return None

        result = future.result()
        if result is None:
            self.get_logger().error('Planning service failed.')
            return None
        if result.motion_plan_response.error_code.val != 1:
            self.get_logger().error('Planning failed.')
            return None

        self.get_logger().info('Motion plan computed successfully.')
        return result.motion_plan_response.trajectory


def main(args=None):
    rclpy.init(args=args)
    node = IKPlanner()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    current_state = JointState()
    current_state.name = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
    ]
    current_state.position = [4.722, -1.850, -1.425, -1.405, 1.593, -3.141]

    node.get_logger().info('Testing IK computation...')
    ik_result = node.compute_ik(current_state, 0.125, 0.611, 0.423)

    if ik_result is None:
        node.get_logger().error('IK computation returned None.')
        rclpy.shutdown()
        sys.exit(1)

    if not hasattr(ik_result, 'name') or not hasattr(ik_result, 'position'):
        node.get_logger().error('IK result missing required fields (name, position).')
        rclpy.shutdown()
        sys.exit(1)

    if len(ik_result.name) != len(ik_result.position):
        node.get_logger().error('IK joint names and positions length mismatch.')
        rclpy.shutdown()
        sys.exit(1)

    if len(ik_result.name) < 6:
        node.get_logger().error('IK returned fewer than 6 joints — likely incorrect.')
        rclpy.shutdown()
        sys.exit(1)

    node.get_logger().info('IK check passed.')
    spin_thread.join()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
