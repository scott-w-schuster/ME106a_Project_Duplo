"""

Workflow (for Readability):
  1. Subscribe to RealSense RGB + depth topics
  2. Color segmentation (HSV) -> per-color blobs
  3. Depth discontinuity -> split touching same-color blobs
  4. Stud detection (HoughCircles) -> brick shape + orientation
  5. Back-project to 3D using camera intrinsics + depth
  6. Publish detected brick poses (in camera frame)
     -> TF chain handles camera_frame -> base_link automatically

ROS2 Topics:
  Subscribed:
    /camera/camera/color/image_raw        (sensor_msgs/Image)
    /camera/camera/depth/image_rect_raw   (sensor_msgs/Image)
    /camera/camera/color/camera_info      (sensor_msgs/CameraInfo)
  Published:
    /detected_bricks                      (geometry_msgs/PoseArray)
    /brick_debug_image                    (sensor_msgs/Image)  [visualization]
"""

import json

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header, String

import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# HSV ranges for Duplo brick colors
# Format: (lower_hsv, upper_hsv)
# ---------------------------------------------------------------------------
COLOR_RANGES = {
    # Each value is a list of (lower_hsv, upper_hsv) ranges — OR'd together.
    # Red wraps around 0°/180° in OpenCV HSV (0-179), so it needs two ranges.
    "red":    [((0,   120, 70),  (10,  255, 255)),
               ((170, 120, 70),  (179, 255, 255))],
    "blue":   [((100, 120, 70),  (130, 255, 255))],
    "yellow": [((20,  120, 70),  (35,  255, 255))],
    "green":  [((40,  70,  70),  (80,  255, 255))],
    # more colors to be added
}

# ---------------------------------------------------------------------------
# Stud geometry constants — tune empirically once camera height is fixed
# Duplo stud pitch is 16mm, stud diameter is ~9mm
# ---------------------------------------------------------------------------
STUD_MIN_RADIUS_PX = 5    # minimum stud radius in pixels
STUD_MAX_RADIUS_PX = 20   # maximum stud radius in pixels
STUD_MIN_DIST_PX   = 15   # minimum distance between stud centers


class BrickDetectorNode(Node):

    def __init__(self):
        super().__init__('brick_detector')
        self.bridge = CvBridge()

        # Camera intrinsics filled in by camera_info_callback
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.latest_rgb   = None
        self.latest_depth = None

        # Subscribers
        self.create_subscription(Image,'/camera/camera/color/image_raw',self.rgb_callback,10)
        self.create_subscription(Image,'/camera/camera/depth/image_rect_raw',self.depth_callback,10)
        self.create_subscription(CameraInfo,'/camera/camera/color/camera_info',self.camera_info_callback,10)

        # Publishers
        self.pose_pub  = self.create_publisher(PoseArray, '/detected_bricks',      10)
        self.debug_pub = self.create_publisher(Image,     '/brick_debug_image',     10)
        # JSON array of {color, shape} — one entry per pose, same indices as /detected_bricks
        self.meta_pub  = self.create_publisher(String,    '/detected_bricks_meta',  10)

        # TF
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')

        self.create_timer(0.2, self.process)
        self.get_logger().info('BrickDetectorNode initialized.')



    # Callbacks

    def camera_info_callback(self, msg: CameraInfo):
        # K is the 3x3 row-major intrinsic matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def rgb_callback(self, msg: Image):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg: Image):
        # Depth is in millimeters as uint16 (convert to float meters)
        depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_depth = depth_mm.astype(np.float32) / 1000.0



    # Main processing loop

    def process(self):

        if self.latest_rgb is None or self.latest_depth is None:
            return
        if self.fx is None:
            self.get_logger().warn('Waiting for camera intrinsics...', once=True)
            return

        rgb   = self.latest_rgb.copy()
        depth = self.latest_depth.copy()
        debug = rgb.copy()

        all_bricks = []  # list of dicts: {color, shape, pose}

        # segment each color, then split touching same-color bricks by depth
        for color_name, ranges in COLOR_RANGES.items():
            mask   = self.color_segment(rgb, ranges)
            blocks = self.split_by_depth(mask, depth)

            # detect studs and infer shape/orientation per blob
            for block_mask in blocks:
                result = self.detect_studs(rgb, block_mask)
                if result is None:
                    continue
                stud_centers, brick_shape, angle_deg = result

                # back-project stud center to 3D using depth + intrinsics
                pose_cam = self.compute_3d_pose(stud_centers, angle_deg, depth)
                if pose_cam is None:
                    continue

                all_bricks.append({
                    'color': color_name,
                    'shape': brick_shape,   # e.g. (2, 4) for a 2x4
                    'pose':  pose_cam,
                })

                self.draw_detection(debug, stud_centers, color_name, brick_shape)

        # transform poses to base_link and publish 
        self.publish_poses(all_bricks)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

    # Main Helper Functions


    def color_segment(self, rgb: np.ndarray, ranges: list) -> np.ndarray:
        """
        Convert BGR -> HSV, OR together all (lower, upper) ranges, return binary mask.
        Accepts a list of ranges so wrap-around colors (red) work without special-casing.
        """
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))

        # Opening removes small noise specks; closing fills small holes inside bricks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def split_by_depth(self, color_mask: np.ndarray, depth: np.ndarray) -> list:
        """
        Split a color mask into individual brick blobs using depth discontinuities.

        Approach:
          1. Find connected components in the color mask
          2. Within each component, compute the depth gradient
          3. High-gradient pixels (brick edges) are removed as barriers
          4. Re-run connected components on the barrier-split mask

        Returns: list of binary masks, one per detected blob
        """
        # A depth step of ~20 mm between adjacent pixels signals a brick boundary
        DEPTH_EDGE_THRESH = 0.02

        blobs = []
        num_labels, labels = cv2.connectedComponents(color_mask)

        for label in range(1, num_labels):  # 0 is background
            blob = (labels == label).astype(np.uint8) * 255

            # Zero out depth outside this blob so gradients don't bleed
            depth_in_blob = np.where(blob > 0, depth, 0.0).astype(np.float32)

            grad_x = cv2.Sobel(depth_in_blob, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_in_blob, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            # Pixels where depth jumps sharply are interior brick boundaries
            depth_edges = ((grad_mag > DEPTH_EDGE_THRESH) & (blob > 0)).astype(np.uint8) * 255

            # Remove those boundary pixels from the blob
            split_mask = cv2.bitwise_and(blob, cv2.bitwise_not(depth_edges))

            # Closing reconnects any small gaps created by the edge removal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            split_mask = cv2.morphologyEx(split_mask, cv2.MORPH_CLOSE, kernel)

            num_sub, sub_labels = cv2.connectedComponents(split_mask)

            if num_sub <= 2:  # only background + original blob — no split
                blobs.append(blob)
            else:
                for sub_label in range(1, num_sub):
                    sub_blob = (sub_labels == sub_label).astype(np.uint8) * 255
                    blobs.append(sub_blob)

        return blobs

    def detect_studs(self, rgb: np.ndarray, blob_mask: np.ndarray):
        """
        Detect studs within a single brick blob using HoughCircles.

        Returns (stud_centers, brick_shape, angle_deg) or None on failure.
            stud_centers: list of (x, y) pixel coords
            brick_shape:  tuple (rows, cols) e.g. (2, 4) for a 2x4 brick
            angle_deg:    orientation of the brick's long axis in degrees
        """
        # Isolate the blob region for HoughCircles
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=blob_mask)

        # Blur to reduce noise before circle detection
        blurred = cv2.GaussianBlur(gray_masked, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=STUD_MIN_DIST_PX,
            param1=50,   # Canny high threshold — tune empirically
            param2=20,   # accumulator threshold (lower = more circles) — tune empirically
            minRadius=STUD_MIN_RADIUS_PX,
            maxRadius=STUD_MAX_RADIUS_PX
        )

        if circles is None:
            return None

        circles = np.round(circles[0, :]).astype(int)
        stud_centers = [(c[0], c[1]) for c in circles]

        brick_shape = self.infer_brick_shape(stud_centers)
        angle_deg   = self.infer_brick_angle(stud_centers)

        return stud_centers, brick_shape, angle_deg

    def infer_brick_shape(self, stud_centers: list) -> tuple:
        """
        Given stud center pixel coordinates, determine brick shape (rows, cols).
        Uses PCA to find the two principal axes, then counts stud grid positions
        along each axis by measuring the span vs. the minimum stud pitch.
        """
        n = len(stud_centers)
        if n == 0:
            return (0, 0)
        if n == 1:
            return (1, 1)

        pts = np.array(stud_centers, dtype=np.float32)
        mean, eigenvectors, _ = cv2.PCACompute2(pts, mean=None)

        # Project all studs onto the two principal axes
        proj = cv2.PCAProject(pts, mean, eigenvectors)  # shape (n, 2)

        def count_grid_positions(coords: np.ndarray) -> int:
            if len(coords) < 2:
                return 1
            sorted_c = np.sort(coords)
            diffs = np.diff(sorted_c)
            # Only care about gaps larger than a few pixels (filters duplicate/noise)
            significant = diffs[diffs > 3.0]
            if len(significant) == 0:
                return 1
            pitch = float(np.min(significant))   # smallest gap ≈ one stud pitch
            span  = float(sorted_c[-1] - sorted_c[0])
            return max(1, round(span / pitch) + 1)

        # PCA axis 0 is the long axis (most variance) cols
        # PCA axis 1 is the short axis                rows
        ncols = count_grid_positions(proj[:, 0])
        nrows = count_grid_positions(proj[:, 1])

        return (nrows, ncols)

    def infer_brick_angle(self, stud_centers: list) -> float:
        """
        Return the orientation angle (degrees) of the brick's long axis relative
        to the image x-axis. Uses PCA — first principal component = long axis.
        """
        if len(stud_centers) < 2:
            return 0.0

        pts = np.array(stud_centers, dtype=np.float32)
        mean, eigenvectors, _ = cv2.PCACompute2(pts, mean=None)
        angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
        return angle

    def compute_3d_pose(self, stud_centers: list, angle_deg: float,
                        depth: np.ndarray) -> Pose:
        """
        Back-project the brick center from image coords to 3D camera frame.
        Uses camera intrinsics (same equations as Lab 8).

        Returns a geometry_msgs/Pose in camera frame, or None on failure.
        """
        if not stud_centers:
            return None

        # Pixel center = mean of stud centers
        u = np.mean([c[0] for c in stud_centers])
        v = np.mean([c[1] for c in stud_centers])
        u_int, v_int = int(u), int(v)

        # Average depth over a 5x5 window more robust than a single pixel
        h, w = depth.shape
        half = 2
        y0, y1 = max(0, v_int - half), min(h, v_int + half + 1)
        x0, x1 = max(0, u_int - half), min(w, u_int + half + 1)
        window = depth[y0:y1, x0:x1]
        valid  = window[(window > 0) & ~np.isnan(window)]
        if len(valid) == 0:
            self.get_logger().warn(f'Invalid depth near ({u_int}, {v_int})')
            return None
        Z = float(np.median(valid))

        # Back-project using intrinsics
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        # Build Pose message
        pose = Pose()
        pose.position.x = float(X)
        pose.position.y = float(Y)
        pose.position.z = float(Z)

        # Convert yaw angle to quaternion
        # Yaw is rotation about Z axis (optical axis) in camera frame
        r = Rotation.from_euler('z', angle_deg, degrees=True)
        q = r.as_quat()  # [x, y, z, w]
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        return pose

    def publish_poses(self, bricks: list):
        """
        Transform each detected brick pose from camera frame -> base_link,
        publish as a PoseArray, and publish parallel JSON metadata on
        /detected_bricks_meta so callers know color/shape for each pose index.
        """
        camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp    = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'base_link'

        meta_list = []

        for brick in bricks:
            pose_cam = brick['pose']

            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    camera_frame,
                    rclpy.time.Time()
                )
                pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam, transform)
                pose_array.poses.append(pose_base)
                meta_list.append({'color': brick['color'], 'shape': list(brick['shape'])})

            except tf2_ros.LookupException as e:
                self.get_logger().warn(f'TF lookup failed: {e}')
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(f'TF extrapolation error: {e}')

        self.pose_pub.publish(pose_array)

        meta_msg = String()
        meta_msg.data = json.dumps(meta_list)
        self.meta_pub.publish(meta_msg)

        self.get_logger().info(f'Published {len(pose_array.poses)} brick poses.')

    def draw_detection(self, img: np.ndarray, stud_centers: list,
                       color_name: str, brick_shape: tuple):
        """Draw detected studs and brick info onto debug image."""
        for (x, y) in stud_centers:
            cv2.circle(img, (x, y), STUD_MAX_RADIUS_PX, (0, 255, 0), 2)
        if stud_centers:
            cx = int(np.mean([c[0] for c in stud_centers]))
            cy = int(np.mean([c[1] for c in stud_centers]))
            label = f'{color_name} {brick_shape[0]}x{brick_shape[1]}'
            cv2.putText(img, label, (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# Entry point


def main(args=None):
    rclpy.init(args=args)
    node = BrickDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
