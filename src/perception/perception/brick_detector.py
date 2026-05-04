"""
brick_detector.py

Workflow:
  1. Subscribe to RealSense RGB + depth topics
  2. Color segmentation (HSV) -> per-color blobs
  3. Depth discontinuity -> split touching same-color bricks
  4. Stud detection (HoughCircles) -> brick shape + orientation
  5. Back-project centroid to 3D using camera intrinsics + depth
  6. Estimate brick height type (half / normal / tall) from depth vs baseplate
  7. Publish detected brick poses (centroid, in base_link) + metadata

ROS2 Topics:
  Subscribed:
    /camera/camera/color/image_raw        (sensor_msgs/Image)
    /camera/camera/depth/image_rect_raw   (sensor_msgs/Image)
    /camera/camera/color/camera_info      (sensor_msgs/CameraInfo)
  Published:
    /detected_bricks                      (geometry_msgs/PoseArray)
      Poses are the CENTROID of each brick's top surface in base_link.
    /detected_bricks_meta                 (std_msgs/String)
      JSON array — one entry per pose: {color, shape, height_type}
    /brick_debug_image                    (sensor_msgs/Image)   [visualization]
  TF Broadcast:
    baseplate_frame  (child of base_link) — published as STATIC once detected
    All TF lookups use rclpy.time.Time() with a timeout so static transforms
    (e.g. camera_frame -> base_link from a static_transform_publisher) are
    correctly resolved even immediately after node startup.
"""

import json

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
from std_msgs.msg import Header, String

import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# HSV color ranges for Duplo brick detection
# Format: list of (lower_hsv, upper_hsv) — ranges are OR'd together.
# OpenCV convention: H 0-179, S 0-255, V 0-255.
# These are starting values calibrated from the physical brick set under
# indoor LED lighting — tune param1/param2 after testing with your camera.
# ---------------------------------------------------------------------------
COLOR_RANGES = {
    # Red wraps around H=0/179 in OpenCV HSV — needs two ranges.
    "red":         [((0,   160,  80), (8,   255, 210)),
                    ((172, 160,  80), (179, 255, 210))],

    "orange":      [((5,   200, 150), (15,  255, 235))],

    # Yellow and orange are close in hue; saturation floor separates them.
    "yellow":      [((20,  150, 150), (35,  255, 240))],

    # Light green (lime) bricks — distinct from the dark green baseplate.
    "light_green": [((35,  120,  80), (55,  255, 210))],

    # Sky blue — medium-high saturation, clearly blue hue.
    "sky_blue":    [((95,  150, 140), (115, 255, 235))],

    # Mint — lighter, less saturated than sky_blue, shifted slightly green.
    "mint":        [((80,   50, 170), (95,  140, 235))],

    # White — saturation-based only; hue is irrelevant for near-white bricks.
    "white":       [((0,     0, 200), (179,  50, 255))],

    "purple":      [((130,  60,  90), (155, 180, 185))],
}

# ---------------------------------------------------------------------------
# Baseplate HSV range — tuned to the dark green Duplo baseplate.
# Darker and more saturated than the light green bricks.
# ---------------------------------------------------------------------------
BASEPLATE_HSV_LOWER   = (55,  80,  30)
BASEPLATE_HSV_UPPER   = (80, 200, 100)
BASEPLATE_MIN_AREA_PX = 5000   # ignore blobs smaller than this

# ---------------------------------------------------------------------------
# Brick height classification (meters above baseplate surface)
# half   =  9.6 mm, normal = 19.2 mm, tall = 38.4 mm (2× normal)
# Thresholds are set at the midpoints between adjacent heights.
# ---------------------------------------------------------------------------
HEIGHT_THRESHOLDS = {
    'half':   (0.000, 0.013),   # 0 – 13 mm
    'normal': (0.013, 0.030),   # 13 – 30 mm
    'tall':   (0.030, 9.999),   # > 30 mm
}

# ---------------------------------------------------------------------------
# Stud geometry constants — tune empirically once camera height is fixed.
# Duplo stud pitch is 16 mm, stud diameter ~9 mm.
# ---------------------------------------------------------------------------
STUD_MIN_RADIUS_PX = 5
STUD_MAX_RADIUS_PX = 20
STUD_MIN_DIST_PX   = 15

# How long to wait for a TF transform before giving up (seconds).
TF_TIMEOUT_SEC = 1.0


class BrickDetectorNode(Node):

    def __init__(self):
        super().__init__('brick_detector')
        self.bridge = CvBridge()

        # Camera intrinsics (filled by camera_info_callback)
        self.fx = self.fy = self.cx = self.cy = None

        self.latest_rgb   = None
        self.latest_depth = None

        # Baseplate surface depth in camera frame (meters).
        # Set each cycle when the baseplate is detected; used for height typing.
        self.baseplate_z_cam = None

        # --- Subscribers ---
        self.create_subscription(
            Image,      '/camera/camera/color/image_raw',
            self.rgb_callback, 10)
        self.create_subscription(
            Image,      '/camera/camera/depth/image_rect_raw',
            self.depth_callback, 10)
        self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info',
            self.camera_info_callback, 10)

        # --- Publishers ---
        self.pose_pub  = self.create_publisher(PoseArray, '/detected_bricks',     10)
        self.meta_pub  = self.create_publisher(String,    '/detected_bricks_meta', 10)
        self.debug_pub = self.create_publisher(Image,     '/brick_debug_image',    10)

        # --- TF ---
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Dynamic broadcaster for per-cycle updates (bricks are on the table).
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Static broadcaster for the baseplate frame.
        # StaticTransformBroadcaster latches the message — the transform
        # persists in the TF tree even if the baseplate is temporarily occluded,
        # and it is immediately visible to nodes using rclpy.time.Time().
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')

        self.create_timer(0.2, self.process)
        self.get_logger().info('BrickDetectorNode initialized.')

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def camera_info_callback(self, msg: CameraInfo):
        # K is the 3x3 row-major intrinsic matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def rgb_callback(self, msg: Image):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg: Image):
        depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_depth = depth_mm.astype(np.float32) / 1000.0

    # -----------------------------------------------------------------------
    # Main processing loop
    # -----------------------------------------------------------------------

    def process(self):
        if self.latest_rgb is None or self.latest_depth is None:
            return
        if self.fx is None:
            self.get_logger().warn('Waiting for camera intrinsics...', once=True)
            return

        rgb   = self.latest_rgb.copy()
        depth = self.latest_depth.copy()
        debug = rgb.copy()

        # Detect baseplate and broadcast its static TF frame
        bp = self.detect_baseplate(rgb, depth)
        if bp is not None:
            X, Y, Z, angle_deg, box_pts = bp
            self.baseplate_z_cam = Z   # store for height classification
            self.publish_baseplate_tf(X, Y, Z, angle_deg)
            cv2.drawContours(debug, [box_pts], 0, (0, 200, 0), 2)
            cv2.putText(debug, 'baseplate', tuple(box_pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        all_bricks = []   # list of {color, shape, height_type, pose}

        for color_name, ranges in COLOR_RANGES.items():
            mask   = self.color_segment(rgb, ranges)
            blocks = self.split_by_depth(mask, depth)

            for block_mask in blocks:
                result = self.detect_studs(rgb, block_mask)
                if result is None:
                    continue
                stud_centers, brick_shape, angle_deg = result

                pose_cam = self.compute_3d_pose(stud_centers, angle_deg, depth)
                if pose_cam is None:
                    continue

                height_type = self._estimate_height_type(pose_cam.position.z)

                all_bricks.append({
                    'color':       color_name,
                    'shape':       brick_shape,
                    'height_type': height_type,
                    'pose':        pose_cam,
                })

                self.draw_detection(debug, stud_centers, color_name,
                                    brick_shape, height_type)

        self.publish_poses(all_bricks)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

    # -----------------------------------------------------------------------
    # Detection helpers
    # -----------------------------------------------------------------------

    def color_segment(self, rgb: np.ndarray, ranges: list) -> np.ndarray:
        """
        Convert BGR → HSV, OR together all (lower, upper) ranges, return binary mask.
        Accepts a list of ranges so wrap-around colors (red) work without special-casing.
        """
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask,
                                  cv2.inRange(hsv, np.array(lower), np.array(upper)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def split_by_depth(self, color_mask: np.ndarray, depth: np.ndarray) -> list:
        """
        Split a color mask into individual brick blobs using depth discontinuities.

        A depth step of ~20 mm between adjacent pixels signals a brick boundary.
        Returns a list of binary masks, one per detected blob.
        """
        DEPTH_EDGE_THRESH = 0.02

        blobs = []
        num_labels, labels = cv2.connectedComponents(color_mask)

        for label in range(1, num_labels):
            blob = (labels == label).astype(np.uint8) * 255

            depth_in_blob = np.where(blob > 0, depth, 0.0).astype(np.float32)
            grad_x   = cv2.Sobel(depth_in_blob, cv2.CV_32F, 1, 0, ksize=3)
            grad_y   = cv2.Sobel(depth_in_blob, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            depth_edges = ((grad_mag > DEPTH_EDGE_THRESH) & (blob > 0)).astype(np.uint8) * 255
            split_mask  = cv2.bitwise_and(blob, cv2.bitwise_not(depth_edges))

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            split_mask = cv2.morphologyEx(split_mask, cv2.MORPH_CLOSE, kernel)

            num_sub, sub_labels = cv2.connectedComponents(split_mask)
            if num_sub <= 2:
                blobs.append(blob)
            else:
                for sub_label in range(1, num_sub):
                    blobs.append((sub_labels == sub_label).astype(np.uint8) * 255)

        return blobs

    def detect_studs(self, rgb: np.ndarray, blob_mask: np.ndarray):
        """
        Detect studs within a single brick blob using HoughCircles.

        Returns (stud_centers, brick_shape, angle_deg) or None on failure.
            stud_centers: list of (x, y) pixel coords
            brick_shape:  tuple (rows, cols) e.g. (2, 4)
            angle_deg:    orientation of brick's long axis in image degrees
        """
        gray        = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=blob_mask)
        blurred     = cv2.GaussianBlur(gray_masked, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=STUD_MIN_DIST_PX,
            param1=50,
            param2=20,
            minRadius=STUD_MIN_RADIUS_PX,
            maxRadius=STUD_MAX_RADIUS_PX,
        )
        if circles is None:
            return None

        circles      = np.round(circles[0, :]).astype(int)
        stud_centers = [(c[0], c[1]) for c in circles]
        brick_shape  = self.infer_brick_shape(stud_centers)
        angle_deg    = self.infer_brick_angle(stud_centers)

        return stud_centers, brick_shape, angle_deg

    def infer_brick_shape(self, stud_centers: list) -> tuple:
        """
        Given stud center pixel coordinates, determine brick shape (rows, cols).
        Uses PCA to find the two principal axes, counts stud positions along each.
        """
        n = len(stud_centers)
        if n == 0:
            return (0, 0)
        if n == 1:
            return (1, 1)

        pts = np.array(stud_centers, dtype=np.float32)
        mean, eigenvectors, _ = cv2.PCACompute2(pts, mean=None)
        proj = cv2.PCAProject(pts, mean, eigenvectors)

        def count_grid_positions(coords: np.ndarray) -> int:
            if len(coords) < 2:
                return 1
            sorted_c    = np.sort(coords)
            diffs       = np.diff(sorted_c)
            significant = diffs[diffs > 3.0]
            if len(significant) == 0:
                return 1
            pitch = float(np.min(significant))
            span  = float(sorted_c[-1] - sorted_c[0])
            return max(1, round(span / pitch) + 1)

        # PCA axis 0 = long axis (most variance) → cols
        # PCA axis 1 = short axis                → rows
        ncols = count_grid_positions(proj[:, 0])
        nrows = count_grid_positions(proj[:, 1])
        return (nrows, ncols)

    def infer_brick_angle(self, stud_centers: list) -> float:
        """Return orientation angle (degrees) of brick long axis vs image x-axis."""
        if len(stud_centers) < 2:
            return 0.0
        pts = np.array(stud_centers, dtype=np.float32)
        mean, eigenvectors, _ = cv2.PCACompute2(pts, mean=None)
        return np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))

    def compute_3d_pose(self, stud_centers: list, angle_deg: float,
                        depth: np.ndarray) -> Pose:
        """
        Back-project the brick CENTROID from image coords to 3D camera frame.

        Position = mean of all stud centers → centroid of the brick's top surface.
        The build_planner expects centroid poses so its grasp offset math works
        correctly for all brick sizes.

        Returns a geometry_msgs/Pose in camera frame, or None on failure.
        """
        if not stud_centers:
            return None

        u     = np.mean([c[0] for c in stud_centers])
        v     = np.mean([c[1] for c in stud_centers])
        u_int = int(u)
        v_int = int(v)

        # Average depth over a 5×5 window — more robust than a single pixel
        h, w = depth.shape
        half  = 2
        y0, y1 = max(0, v_int - half), min(h, v_int + half + 1)
        x0, x1 = max(0, u_int - half), min(w, u_int + half + 1)
        window  = depth[y0:y1, x0:x1]
        valid   = window[(window > 0) & ~np.isnan(window)]
        if len(valid) == 0:
            self.get_logger().warn(f'Invalid depth near ({u_int}, {v_int})')
            return None
        Z = float(np.median(valid))

        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        pose = Pose()
        pose.position.x = float(X)
        pose.position.y = float(Y)
        pose.position.z = float(Z)

        r = Rotation.from_euler('z', angle_deg, degrees=True)
        q = r.as_quat()
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]

        return pose

    def _estimate_height_type(self, brick_z_cam: float) -> str:
        """
        Classify brick height by comparing its top-surface camera-frame Z against
        the stored baseplate surface Z.

        In the camera optical frame, Z increases with distance from the camera.
        Bricks sit on top of the baseplate, so brick_z < baseplate_z.
        height_above_baseplate = baseplate_z_cam - brick_z_cam.

        Returns 'half', 'normal', or 'tall'. Falls back to 'normal' if the
        baseplate hasn't been detected yet.
        """
        if self.baseplate_z_cam is None:
            return 'normal'

        height_m = self.baseplate_z_cam - brick_z_cam
        for label, (lo, hi) in HEIGHT_THRESHOLDS.items():
            if lo <= height_m < hi:
                return label

        return 'normal'   # shouldn't reach here, but safe default

    # -----------------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------------

    def publish_poses(self, bricks: list):
        """
        Transform each detected brick centroid pose from camera frame → base_link,
        publish as a PoseArray.  Parallel metadata (color, shape, height_type) is
        published as a JSON array on /detected_bricks_meta at the same indices.

        Uses rclpy.time.Time() with a TF_TIMEOUT_SEC timeout so static camera
        transforms (published by static_transform_publisher) are correctly resolved
        on the first lookup without requiring a warm-up wait.
        """
        camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp    = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'base_link'

        meta_list = []

        for brick in bricks:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link',
                    camera_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=TF_TIMEOUT_SEC),
                )
                pose_base = tf2_geometry_msgs.do_transform_pose(brick['pose'], transform)
                pose_array.poses.append(pose_base)
                meta_list.append({
                    'color':       brick['color'],
                    'shape':       list(brick['shape']),
                    'height_type': brick['height_type'],
                })
            except tf2_ros.LookupException as e:
                self.get_logger().warn(f'TF lookup failed: {e}')
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(f'TF extrapolation error: {e}')

        self.pose_pub.publish(pose_array)

        meta_msg      = String()
        meta_msg.data = json.dumps(meta_list)
        self.meta_pub.publish(meta_msg)

        self.get_logger().info(f'Published {len(pose_array.poses)} brick poses.')

    def detect_baseplate(self, rgb: np.ndarray, depth: np.ndarray):
        """
        Detect the green Duplo baseplate in the scene.

        Returns (X, Y, Z, angle_deg, box_pts) in camera frame, or None if not found.
            X, Y, Z:   3D centroid of the baseplate surface (meters, camera frame)
            angle_deg: yaw of the baseplate long axis (degrees)
            box_pts:   4 corner pixels of the fitted rectangle (for debug drawing)
        """
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array(BASEPLATE_HSV_LOWER),
                           np.array(BASEPLATE_HSV_UPPER))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < BASEPLATE_MIN_AREA_PX:
            self.get_logger().warn('Largest green blob too small — baseplate not found', once=True)
            return None

        rect = cv2.minAreaRect(largest)
        (cx_px, cy_px), (w_px, h_px), angle_deg = rect
        if w_px < h_px:
            angle_deg += 90.0
        box_pts = np.int0(cv2.boxPoints(rect))

        cx_int, cy_int = int(cx_px), int(cy_px)
        img_h, img_w   = depth.shape
        half           = 5
        y0, y1 = max(0, cy_int - half), min(img_h, cy_int + half + 1)
        x0, x1 = max(0, cx_int - half), min(img_w, cx_int + half + 1)
        window  = depth[y0:y1, x0:x1]
        valid   = window[(window > 0) & ~np.isnan(window)]
        if len(valid) == 0:
            self.get_logger().warn(f'Invalid depth at baseplate center ({cx_int}, {cy_int})')
            return None
        Z = float(np.median(valid))

        X = (cx_px - self.cx) * Z / self.fx
        Y = (cy_px - self.cy) * Z / self.fy

        return X, Y, Z, angle_deg, box_pts

    def publish_baseplate_tf(self, X: float, Y: float, Z: float, angle_deg: float):
        """
        Transform the detected baseplate pose from camera frame → base_link,
        then broadcast it as the 'baseplate_frame' static TF.

        Using StaticTransformBroadcaster means:
          - The transform is immediately available to TF listeners (no warm-up wait).
          - It persists in the TF tree if the baseplate is temporarily occluded.
          - Re-broadcasting with updated values (each detection cycle) is valid and
            simply refreshes the latched message.

        TF lookup uses rclpy.time.Time() with a timeout so a static
        camera → base_link transform is resolved on the very first call.
        """
        camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        pose_cam = Pose()
        pose_cam.position.x = X
        pose_cam.position.y = Y
        pose_cam.position.z = Z
        rot = Rotation.from_euler('z', angle_deg, degrees=True)
        q   = rot.as_quat()
        pose_cam.orientation.x = q[0]
        pose_cam.orientation.y = q[1]
        pose_cam.orientation.z = q[2]
        pose_cam.orientation.w = q[3]

        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                camera_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=TF_TIMEOUT_SEC),
            )
            pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam, transform)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'Baseplate TF lookup failed: {e}')
            return

        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id  = 'baseplate_frame'
        t.transform.translation.x = pose_base.position.x
        t.transform.translation.y = pose_base.position.y
        t.transform.translation.z = pose_base.position.z
        t.transform.rotation      = pose_base.orientation

        self.static_tf_broadcaster.sendTransform(t)

    def draw_detection(self, img: np.ndarray, stud_centers: list,
                       color_name: str, brick_shape: tuple, height_type: str):
        """Draw detected studs and brick info onto the debug image."""
        for (x, y) in stud_centers:
            cv2.circle(img, (x, y), STUD_MAX_RADIUS_PX, (0, 255, 0), 2)
        if stud_centers:
            cx    = int(np.mean([c[0] for c in stud_centers]))
            cy    = int(np.mean([c[1] for c in stud_centers]))
            label = f'{color_name} {brick_shape[0]}x{brick_shape[1]} [{height_type}]'
            cv2.putText(img, label, (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
