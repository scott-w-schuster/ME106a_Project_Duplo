import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
import numpy as np


class ConstantTransformPublisher(Node):
    def __init__(self):
        super().__init__('constant_tf_publisher')
        self.br = StaticTransformBroadcaster(self)

        G = np.array([[1.0, 0.0, 0.0, -0.025],
                      [0.0, 1.0, 0.0,  0.13],
                      [0.0, 0.0, 1.0,  0.0],
                      [0.0, 0.0, 0.0,  1.0]])

        q = R.from_matrix(G[:3, :3]).as_quat()

        self.transform = TransformStamped()
        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.transform.header.frame_id = 'wrist_3_link'
        self.transform.child_frame_id = 'camera_depth_optical_frame'
        self.transform.transform.translation.x = G[0, 3]
        self.transform.transform.translation.y = G[1, 3]
        self.transform.transform.translation.z = G[2, 3]
        self.transform.transform.rotation.x = q[0]
        self.transform.transform.rotation.y = q[1]
        self.transform.transform.rotation.z = q[2]
        self.transform.transform.rotation.w = q[3]

        self.get_logger().info(f'Broadcasting wrist_3_link -> camera_depth_optical_frame: t={G[:3,3]} q={q}')
        self.timer = self.create_timer(0.05, self.broadcast_tf)

    def broadcast_tf(self):
        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.br.sendTransform(self.transform)


def main():
    rclpy.init()
    node = ConstantTransformPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
