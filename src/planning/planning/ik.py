import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK, GetMotionPlan
from moveit_msgs.msg import Constraints, JointConstraint
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration


class IKPlanner(Node):
    def __init__(self):
        super().__init__('ik_planner')

        self.ik_client   = self.create_client(GetPositionIK,  '/compute_ik')
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')

        for srv, name in [(self.ik_client, 'compute_ik'),
                          (self.plan_client, 'plan_kinematic_path')]:
            while not srv.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for /{name} service...')

    def compute_ik(self, current_joint_state, x, y, z,
                   qx=0.0, qy=1.0, qz=0.0, qw=0.0):
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        req = GetPositionIK.Request()
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = Duration(sec=5)
        req.ik_request.group_name = 'ur_manipulator'
        req.ik_request.robot_state.joint_state = current_joint_state
        req.ik_request.pose_stamped = pose

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is None or result.error_code.val != result.error_code.SUCCESS:
            self.get_logger().error(f'IK failed, code: {result.error_code.val if result else "None"}')
            return None

        self.get_logger().info('IK solution found.')
        return result.solution.joint_state

    def plan_to_joints(self, target_joint_state, current_joint_state=None, vel=0.1, accel=0.1):
        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = 'ur_manipulator'
        req.motion_plan_request.allowed_planning_time = 5.0
        req.motion_plan_request.planner_id = 'RRTConnectkConfigDefault'
        req.motion_plan_request.max_velocity_scaling_factor     = vel
        req.motion_plan_request.max_acceleration_scaling_factor = accel

        if current_joint_state is not None:
            req.motion_plan_request.start_state.joint_state = current_joint_state

        constraints = Constraints()
        for name, pos in zip(target_joint_state.name, target_joint_state.position):
            constraints.joint_constraints.append(
                JointConstraint(
                    joint_name=name,
                    position=pos,
                    tolerance_above=0.01,
                    tolerance_below=0.01,
                    weight=1.0,
                )
            )
        req.motion_plan_request.goal_constraints.append(constraints)

        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is None or result.motion_plan_response.error_code.val != 1:
            self.get_logger().error('Planning failed.')
            return None

        self.get_logger().info('Motion plan computed successfully.')
        return result.motion_plan_response.trajectory
