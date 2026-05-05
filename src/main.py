# ROS Libraries
from std_srvs.srv import Trigger
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PointStamped 
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np

from planning.ik import IKPlanner

class UR7e_CubeGrasp(Node):
    def __init__(self):
        super().__init__('cube_grasp')

        self.cube_pub = self.create_subscription(PointStamped, '/cube_pose', self.cube_callback, 1)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 1)

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.gripper_cli = self.create_client(Trigger, '/toggle_gripper')

        self.cube_pose = None
        self.current_plan = None
        self.joint_state = None

        self.ik_planner = IKPlanner()

        self.job_queue = [] # Entries should be of type either (x, y, z, vel_scale, accel_scale) or String('toggle_grip')

    def joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def cube_callback(self, cube_pose):
        if self.cube_pose is not None:
            return

        if self.joint_state is None:
            self.get_logger().info("No joint state yet, cannot proceed")
            return

        self.cube_pose = cube_pose
        self.pregraspZOffset = 0.185
        self.graspZOffest = 0.05
        self.prePlaceZOffeset = 0.185

        self.blockSearch
        self.preGrasp(self, cube_pose)
        self.graspBlock(self, cube_pose)

        self.execute_jobs()

    def blockSearch(self):
        return

    def preGrasp(self, cube_pose):
        self.job_queue.append(cube_pose.point.x, cube_pose.point.y, cube_pose.point.z + self.pregraspZOffset, 0.1, 0.1)
        return
    
    def graspBlock(self, cube_pose):
        self.job_queue.append(cube_pose.point.x, cube_pose.point.y, cube_pose.point.z + self.graspZOffest, 0.1, 0.1)
        self.job_queue.append('toggle_grip')
        self.job_queue.append(cube_pose.point.x, cube_pose.point.y, cube_pose.point.z + self.pregraspZOffset, 0.1, 0.1)
        return
    
    def checkPickup(self, cube_pose):
        return
    
    def prePlace(self, place_pose):
        self.job_queue.append(place_pose.point.x, place_pose.point.y, place_pose.point.z + self.prePlaceZOffeset, 0.1, 0.1)
        return
    
    def placeBlock(self, place_pose):
        return

    

    def execute_jobs(self):
        if not self.job_queue:
            self.get_logger().info("All jobs completed.")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Executing job queue, {len(self.job_queue)} jobs remaining.")
        next_job = self.job_queue.pop(0)

        if isinstance(next_job, tuple):
            x, y, z, vel_scale, accel_scale = next_job
            joint_solution = self.ik_planner.compute_ik(self.joint_state, x, y, z)
            if joint_solution is None:
                self.get_logger().error("IK failed for position")
                return

            traj = self.ik_planner.plan_to_joints(joint_solution, vel_scale, accel_scale)
            if traj is None:
                self.get_logger().error("Failed to plan to position")
                return

            self.get_logger().info("Planned to position")

            self._execute_joint_trajectory(traj.joint_trajectory)
        elif next_job == 'toggle_grip':
            self.get_logger().info("Toggling gripper")
            self._toggle_gripper()
        else:
            self.get_logger().error("Unknown job type.")
            self.execute_jobs()  # Proceed to next job

    def _toggle_gripper(self):
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Gripper service not available')
            rclpy.shutdown()
            return

        req = Trigger.Request()
        future = self.gripper_cli.call_async(req)
        # wait for 2 seconds
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info('Gripper toggled.')
        self.execute_jobs()  # Proceed to next job

            
    def _execute_joint_trajectory(self, joint_traj):
        self.get_logger().info('Waiting for controller action server...')
        self.exec_ac.wait_for_server()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info('Sending trajectory to controller...')
        send_future = self.exec_ac.send_goal_async(goal)
        print(send_future)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('bonk')
            rclpy.shutdown()
            return

        self.get_logger().info('Executing...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_exec_done)

    def _on_exec_done(self, future):
        try:
            result = future.result().result
            self.get_logger().info('Execution complete.')
            self.execute_jobs()  # Proceed to next job
        except Exception as e:
            self.get_logger().error(f'Execution failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_CubeGrasp()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()
