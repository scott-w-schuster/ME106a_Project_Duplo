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
  8. Publish RViz MarkerArray for live 3D visualization

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
    /brick_debug_image                    (sensor_msgs/Image)   [2D visualization]
    /duplo_markers                        (visualization_msgs/MarkerArray)
      - Baseplate flat box     (ns: 'baseplate', id 0)
      - Stud grid lines        (ns: 'baseplate', id 1)
      - One box per brick      (ns: 'bricks',    id 0..N)
      - Text label per brick   (ns: 'brick_labels', id 0..N)
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
from geometry_msgs.msg import Pose, PoseArray, Point, TransformStamped
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# HSV color ranges for Duplo brick detection
# Format: list of (lower_hsv, upper_hsv) — ranges are OR'd together.
# OpenCV convention: H 0-179, S 0-255, V 0-255.
# Starting values calibrated from the physical brick set under indoor LED
# lighting — tune after testing with your camera.
# ---------------------------------------------------------------------------
COLOR_RANGES = {
    # Red wraps around H=0/179 in OpenCV HSV — needs two ranges.
    'red':         [((0,   160,  80), (8,   255, 210)),
                    ((172, 160,  80), (179, 255, 210))],
    'orange':      [((5,   200, 150), (15,  255, 235))],
    # Yellow and orange are close in hue; saturation floor separates them.
    'yellow':      [((20,  150, 150), (35,  255, 240))],
    # Light green (lime) bricks — distinct from the dark green baseplate.
    'light_green': [((35,  120,  80), (55,  255, 210))],
    # Sky blue — medium-high saturation, clearly blue hue.
    'sky_blue':    [((95,  150, 140), (115, 255, 235))],
    # Mint — lighter, less saturated than sky_blue, shifted slightly green.
    'mint':        [((80,   50, 170), (95,  140, 235))],
    # White — saturation-based only; hue is irrelevant for near-white bricks.
    'white':       [((0,     0, 200), (179,  50, 255))],
    'purple':      [((130,  60,  90), (155, 180, 185))],
}

# ---------------------------------------------------------------------------
# RViz marker colors — (R, G, B, A) floats 0-1 per color name
# ---------------------------------------------------------------------------
BRICK_COLORS_RGBA = {
    'red':         (1.00, 0.15, 0.15, 1.0),
    'orange':      (1.00, 0.50, 0.00, 1.0),
    'yellow':      (1.00, 0.90, 0.00, 1.0),
    'light_green': (0.50, 1.00, 0.20, 1.0),
    'sky_blue':    (0.30, 0.70, 1.00, 1.0),
    'mint':        (0.60, 1.00, 0.82, 1.0),
    'white':       (0.95, 0.95, 0.95, 1.0),
    'purple':      (0.60, 0.20, 0.85, 1.0),
}

# ---------------------------------------------------------------------------
# Baseplate HSV range — tuned to the dark green Duplo baseplate.
# Darker and more saturated than the light green bricks.
# ---------------------------------------------------------------------------
BASEPLATE_HSV_LOWER   = (55,  80,  30)
BASEPLATE_HSV_UPPER   = (80, 200, 100)
BASEPLATE_MIN_AREA_PX = 5000

# ---------------------------------------------------------------------------
# Duplo physical constants (meters)
# ---------------------------------------------------------------------------
STUD_PITCH_M   = 0.016    # 16 mm between stud centres
BASEPLATE_ROWS = 16
BASEPLATE_COLS = 16

BRICK_HEIGHTS = {
    'half':   0.0096,    #  9.6 mm — Duplo flat / plate
    'normal': 0.0192,    # 19.2 mm — standard Duplo brick
    'tall':   0.0384,    # 38.4 mm — 2× standard (confirmed)
}

# ---------------------------------------------------------------------------
# Brick height classification (meters above baseplate surface)
# Thresholds set at midpoints between adjacent heights.
# ---------------------------------------------------------------------------
HEIGHT_THRESHOLDS = {
    'half':   (0.000, 0.013),   #  0 – 13 mm
    'normal': (0.013, 0.030),   # 13 – 30 mm
    'tall':   (0.030, 9.999),   # > 30 mm
}

# ---------------------------------------------------------------------------
# Stud detection constants — tune empirically once camera height is fixed.
# Duplo stud pitch 16 mm, stud diameter ~9 mm.
# ---------------------------------------------------------------------------
STUD_MIN_RADIUS_PX = 5
STUD_MAX_RADIUS_PX = 20
STUD_MIN_DIST_PX   = 15

# How long to wait for a TF transform before giving up.
TF_TIMEOUT_SEC = 1.0


class BrickDetectorNode(Node):

    def __init__(self):
        super().__init__('brick_detector')
        self.bridge = CvBridge()

        # Camera intrinsics (filled by camera_info_callback)
        self.fx = self.fy = self.cx = self.cy = None

        self.latest_rgb   = None
        self.latest_depth = None

        # Baseplate surface depth in camera frame — used for height typing.
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
        self.pose_pub   = self.create_publisher(PoseArray,   '/detected_bricks',      10)
        self.meta_pub   = self.create_publisher(String,      '/detected_bricks_meta', 10)
        self.debug_pub  = self.create_publisher(Image,       '/brick_debug_image',    10)
        self.marker_pub = self.create_publisher(MarkerArray, '/duplo_markers',        10)

        # --- TF ---
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Dynamic broadcaster (kept for future per-cycle dynamic frames).
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Static broadcaster for baseplate_frame — latches so the frame
        # persists if the baseplate is temporarily occluded.
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')

        self.create_timer(0.2, self.process)
        self.get_logger().info('BrickDetectorNode initialized.')

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def camera_info_callback(self, msg: CameraInfo):
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
            self.baseplate_z_cam = Z
            self.publish_baseplate_tf(X, Y, Z, angle_deg)
            cv2.drawContours(debug, [box_pts], 0, (0, 200, 0), 2)
            cv2.putText(debug, 'baseplate', tuple(box_pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        all_bricks = []   # {color, shape, height_type, pose (camera frame)}

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

        # Transform to base_link once; use result for both PoseArray and markers
        bricks_base = self.publish_poses(all_bricks)
        self.publish_markers(bricks_base)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

    # -----------------------------------------------------------------------
    # Detection helpers
    # -----------------------------------------------------------------------

    def color_segment(self, rgb: np.ndarray, ranges: list) -> np.ndarray:
        """OR together all HSV ranges and return a cleaned binary mask."""
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask,
                                  cv2.inRange(hsv, np.array(lower), np.array(upper)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def split_by_depth(self, color_mask: np.ndarray, depth: np.ndarray) -> list:
        """
        Split a color mask into individual brick blobs using depth discontinuities.
        A depth step of ~20 mm between adjacent pixels signals a brick boundary.
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

            kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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
        """
        gray        = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=blob_mask)
        blurred     = cv2.GaussianBlur(gray_masked, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1, minDist=STUD_MIN_DIST_PX,
            param1=50, param2=20,
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
        """Determine brick shape (rows, cols) from stud positions via PCA."""
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

        ncols = count_grid_positions(proj[:, 0])   # long axis
        nrows = count_grid_positions(proj[:, 1])   # short axis
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
        Back-project the brick CENTROID (mean of stud centers) to 3D camera frame.
        Returns a Pose in camera frame, or None on failure.
        """
        if not stud_centers:
            return None

        u, v      = np.mean([c[0] for c in stud_centers]), np.mean([c[1] for c in stud_centers])
        u_int, v_int = int(u), int(v)

        h, w    = depth.shape
        half    = 2
        y0, y1  = max(0, v_int - half), min(h, v_int + half + 1)
        x0, x1  = max(0, u_int - half), min(w, u_int + half + 1)
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
        Classify brick height from depth relative to the baseplate surface.
        Falls back to 'normal' if the baseplate hasn't been detected yet.
        """
        if self.baseplate_z_cam is None:
            return 'normal'
        height_m = self.baseplate_z_cam - brick_z_cam
        for label, (lo, hi) in HEIGHT_THRESHOLDS.items():
            if lo <= height_m < hi:
                return label
        return 'normal'

    # -----------------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------------

    def publish_poses(self, bricks: list) -> list:
        """
        Transform each detected brick from camera frame → base_link.
        Publishes /detected_bricks (PoseArray) and /detected_bricks_meta (JSON).
        Returns a list of dicts enriched with 'pose_base' (Pose in base_link)
        for use by publish_markers — avoids a second TF lookup.
        """
        camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp    = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'base_link'

        meta_list   = []
        bricks_base = []   # bricks successfully transformed to base_link

        for brick in bricks:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link', camera_frame,
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
                bricks_base.append({**brick, 'pose_base': pose_base})

            except tf2_ros.LookupException as e:
                self.get_logger().warn(f'TF lookup failed: {e}')
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(f'TF extrapolation error: {e}')

        self.pose_pub.publish(pose_array)

        meta_msg      = String()
        meta_msg.data = json.dumps(meta_list)
        self.meta_pub.publish(meta_msg)

        self.get_logger().info(f'Published {len(pose_array.poses)} brick poses.')
        return bricks_base

    def publish_markers(self, bricks_base: list):
        """
        Publish a MarkerArray to /duplo_markers for RViz visualization.

        Contents:
          ns='baseplate' id=0  — flat green box showing the physical baseplate
                                 footprint, published in baseplate_frame so it
                                 automatically tracks the detected plate position.
          ns='baseplate' id=1  — stud grid lines (LINE_LIST) in baseplate_frame.
          ns='bricks'    id=N  — one CUBE per detected brick in base_link,
                                 sized to actual shape × height, coloured to match.
          ns='brick_labels' id=N — TEXT_VIEW_FACING label above each brick.

        A DELETEALL marker is sent first each cycle to clear stale markers from
        bricks that have been removed or moved.
        """
        now     = self.get_clock().now().to_msg()
        markers = MarkerArray()

        # --- Clear all previous markers ---
        clear              = Marker()
        clear.action       = Marker.DELETEALL
        clear.header.frame_id = 'base_link'
        clear.header.stamp = now
        markers.markers.append(clear)

        # --- Baseplate flat box (published in baseplate_frame) ---
        bp_w = BASEPLATE_COLS * STUD_PITCH_M   # 0.256 m
        bp_d = BASEPLATE_ROWS * STUD_PITCH_M   # 0.256 m

        bp_box                 = Marker()
        bp_box.header.frame_id = 'baseplate_frame'
        bp_box.header.stamp    = now
        bp_box.ns              = 'baseplate'
        bp_box.id              = 0
        bp_box.type            = Marker.CUBE
        bp_box.action          = Marker.ADD
        # Centre of plate in baseplate_frame (origin = corner stud)
        bp_box.pose.position.x = bp_w / 2.0
        bp_box.pose.position.y = bp_d / 2.0
        bp_box.pose.position.z = -0.005   # sit slightly below stud surface
        bp_box.pose.orientation.w = 1.0
        bp_box.scale.x         = bp_w
        bp_box.scale.y         = bp_d
        bp_box.scale.z         = 0.008   # 8 mm thin slab
        bp_box.color.r         = 0.10
        bp_box.color.g         = 0.45
        bp_box.color.b         = 0.15
        bp_box.color.a         = 0.85
        markers.markers.append(bp_box)

        # --- Stud grid lines (LINE_LIST in baseplate_frame) ---
        grid                 = Marker()
        grid.header.frame_id = 'baseplate_frame'
        grid.header.stamp    = now
        grid.ns              = 'baseplate'
        grid.id              = 1
        grid.type            = Marker.LINE_LIST
        grid.action          = Marker.ADD
        grid.pose.orientation.w = 1.0
        grid.scale.x         = 0.0008   # line width (0.8 mm)
        grid.color.r         = 0.05
        grid.color.g         = 0.25
        grid.color.b         = 0.05
        grid.color.a         = 0.70

        Z_GRID = 0.001   # 1 mm above the baseplate surface
        # Horizontal lines — one per row boundary
        for row in range(BASEPLATE_ROWS + 1):
            y  = row * STUD_PITCH_M
            p0 = Point(); p0.x = 0.0;  p0.y = y; p0.z = Z_GRID
            p1 = Point(); p1.x = bp_w; p1.y = y; p1.z = Z_GRID
            grid.points += [p0, p1]
        # Vertical lines — one per column boundary
        for col in range(BASEPLATE_COLS + 1):
            x  = col * STUD_PITCH_M
            p0 = Point(); p0.x = x; p0.y = 0.0;  p0.z = Z_GRID
            p1 = Point(); p1.x = x; p1.y = bp_d; p1.z = Z_GRID
            grid.points += [p0, p1]
        markers.markers.append(grid)

        # --- Per-brick cube + label (published in base_link) ---
        for i, brick in enumerate(bricks_base):
            color_name  = brick['color']
            br, bc      = brick['shape']
            height_type = brick['height_type']
            pose_base   = brick['pose_base']
            brick_h     = BRICK_HEIGHTS.get(height_type, BRICK_HEIGHTS['normal'])
            rgba        = BRICK_COLORS_RGBA.get(color_name, (0.8, 0.8, 0.8, 1.0))

            # Cube sized to the brick's physical footprint and height
            box                 = Marker()
            box.header.frame_id = 'base_link'
            box.header.stamp    = now
            box.ns              = 'bricks'
            box.id              = i
            box.type            = Marker.CUBE
            box.action          = Marker.ADD
            box.pose            = pose_base   # centroid + orientation from detector
            # Long axis (cols) → scale.x; short axis (rows) → scale.y
            box.scale.x         = bc * STUD_PITCH_M
            box.scale.y         = br * STUD_PITCH_M
            box.scale.z         = brick_h
            box.color.r         = rgba[0]
            box.color.g         = rgba[1]
            box.color.b         = rgba[2]
            box.color.a         = rgba[3]
            markers.markers.append(box)

            # Text label floating above the brick
            txt                 = Marker()
            txt.header.frame_id = 'base_link'
            txt.header.stamp    = now
            txt.ns              = 'brick_labels'
            txt.id              = i
            txt.type            = Marker.TEXT_VIEW_FACING
            txt.action          = Marker.ADD
            # Explicit position copy — avoid mutating pose_base
            txt.pose.position.x    = pose_base.position.x
            txt.pose.position.y    = pose_base.position.y
            txt.pose.position.z    = pose_base.position.z + brick_h + 0.025
            txt.pose.orientation.w = 1.0
            txt.scale.z            = 0.020   # text height in metres
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text               = f'{color_name}\n{br}x{bc} [{height_type}]'
            markers.markers.append(txt)

        self.marker_pub.publish(markers)

    def detect_baseplate(self, rgb: np.ndarray, depth: np.ndarray):
        """
        Detect the green Duplo baseplate.
        Returns (X, Y, Z, angle_deg, box_pts) in camera frame, or None.
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
        then broadcast as the 'baseplate_frame' STATIC TF.
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
                'base_link', camera_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=TF_TIMEOUT_SEC),
            )
            pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam, transform)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'Baseplate TF lookup failed: {e}')
            return

        t                         = TransformStamped()
        t.header.stamp            = self.get_clock().now().to_msg()
        t.header.frame_id         = 'base_link'
        t.child_frame_id          = 'baseplate_frame'
        t.transform.translation.x = pose_base.position.x
        t.transform.translation.y = pose_base.position.y
        t.transform.translation.z = pose_base.position.z
        t.transform.rotation      = pose_base.orientation

        self.static_tf_broadcaster.sendTransform(t)

    def draw_detection(self, img: np.ndarray, stud_centers: list,
                       color_name: str, brick_shape: tuple, height_type: str):
        """Draw detected studs and brick info onto the 2D debug image."""
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
