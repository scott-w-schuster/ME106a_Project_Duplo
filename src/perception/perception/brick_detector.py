"""
brick_detector.py

Workflow:
  1. Subscribe to RealSense RGB, depth, camera_info, and XYZRGB point cloud
  2. Color segmentation (HSV on RGB image) -> per-color blobs
  3. Point cloud: RANSAC plane fit -> ground plane (baseplate surface)
  4. Depth discontinuity -> split touching same-color blobs
  5. Stud detection (HoughCircles) -> brick shape + orientation
  6. Back-project centroid to 3D using camera intrinsics + depth
  7. Height = 90th-percentile Z of blob's point cluster above ground plane
     (falls back to single-pixel depth if point cloud is unavailable)
  8. Publish poses, metadata, RViz markers, 2D debug image

Why point cloud for height?
  RealSense depth noise at 0.5-1m is ±3-8mm per pixel. Our thinnest brick
  (half-height) is only 9.6mm, so a single-pixel sample is unreliable.
  Averaging hundreds of points per cluster and fitting the ground plane with
  RANSAC reduces effective height error to <2mm.

ROS2 Topics:
  Subscribed:
    /camera/camera/color/image_raw        (sensor_msgs/Image)
    /camera/camera/depth/image_rect_raw   (sensor_msgs/Image)
    /camera/camera/color/camera_info      (sensor_msgs/CameraInfo)
    /camera/camera/depth/color/points     (sensor_msgs/PointCloud2)
  Published:
    /detected_bricks        (geometry_msgs/PoseArray)   — centroid in base_link
    /detected_bricks_meta   (std_msgs/String)            — JSON: color, shape, height_type, height_m
    /brick_debug_image      (sensor_msgs/Image)          — 2D annotated view
    /duplo_markers          (visualization_msgs/MarkerArray) — RViz 3D view
  TF Broadcast:
    baseplate_frame (static, child of base_link)
"""

import json

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Pose, PoseArray, Point, TransformStamped
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# HSV color ranges — calibrated from physical brick set under indoor LED
# lighting.  These WILL need tuning with your actual camera.  A good workflow:
#   1. Run the node and open /brick_debug_image in rqt_image_view
#   2. Adjust lower/upper bounds until each colour mask isolates cleanly
#   3. OpenCV convention: H 0-179, S 0-255, V 0-255
# ---------------------------------------------------------------------------
COLOR_RANGES = {
    'red':          [((  0, 138, 102), (  7, 255, 167)),
                       ((125, 138, 102), (179, 255, 167))],
    'orange':       [((  0, 131, 146), ( 17, 255, 230))],
    'yellow':       [(( 13,  47, 120), ( 50, 255, 236))],
    'light_green':  [(( 28,  82,  31), ( 79, 255, 167))],
    'sky_blue':     [(( 70, 132, 124), (135, 255, 236))],
    'mint':         [(( 49,   7, 148), ( 92, 115, 205))],
    'white':        [(( 88,  12, 158), (164,  82, 255))],
    'purple':       [((108,  44,  84), (144, 186, 202))],
    'brown':       [((0, 0, 39), (179, 107, 133))],
    'pink':       [((170, 139, 150), (179, 255, 208))],
    'light_blue': [((91, 49, 132), (108, 142, 255))],
}

# ---------------------------------------------------------------------------
# RViz marker colours (R, G, B, A) per colour name
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
    'brown':       (0.55, 0.27, 0.07, 1.0),
    'hot_pink':    (1.00, 0.08, 0.58, 1.0),
    'light_blue':  (0.68, 0.85, 0.90, 1.0),
}

# ---------------------------------------------------------------------------
# Baseplate HSV — used for 2D debug overlay only.
# Ground plane is now found via RANSAC on the point cloud.
# ---------------------------------------------------------------------------
BASEPLATE_HSV_LOWER   = (55,  80,  30)
BASEPLATE_HSV_UPPER   = (80, 200, 100)
BASEPLATE_MIN_AREA_PX = 5000

# ---------------------------------------------------------------------------
# Duplo physical constants (meters)
# ---------------------------------------------------------------------------
STUD_PITCH_M   = 0.016
BASEPLATE_ROWS = 16
BASEPLATE_COLS = 16

BRICK_HEIGHTS = {
    'half':   0.0096,
    'normal': 0.0192,
    'tall':   0.0384,
}

# Height classification thresholds (meters above ground plane).
# Midpoints between adjacent brick heights.
HEIGHT_THRESHOLDS = {
    'half':   (0.000, 0.013),
    'normal': (0.013, 0.030),
    'tall':   (0.030, 9.999),
}

# ---------------------------------------------------------------------------
# Blob size filter — contours smaller than this are ignored as noise.
# Increase if spurious detections appear; decrease if small bricks are missed.
# ---------------------------------------------------------------------------
BLOB_MIN_AREA_PX = 150

# ---------------------------------------------------------------------------
# Point cloud / RANSAC parameters
# ---------------------------------------------------------------------------
# Keep every Nth point for RANSAC — 640×480 / 8 ≈ 38 k points, fast enough.
PC_SUBSAMPLE          = 8
RANSAC_ITERATIONS     = 100
# Points within this distance of the fitted plane count as inliers.
RANSAC_INLIER_THRESH  = 0.008   # 8 mm
# Ground plane normal must be mostly vertical in camera frame (|n_z| > this).
PLANE_NORMAL_MIN_Z    = 0.85
# Use this percentile of cluster depths as the brick top surface.
# 90th percentile rejects a few noisy high points without discarding valid ones.
HEIGHT_PERCENTILE     = 90
# Minimum number of valid cluster points required to trust height estimate.
MIN_CLUSTER_PTS       = 15

TF_TIMEOUT_SEC = 1.0


class BrickDetectorNode(Node):

    def __init__(self):
        super().__init__('brick_detector')
        self.bridge = CvBridge()

        self.fx = self.fy = self.cx = self.cy = None
        self.latest_rgb   = None
        self.latest_depth = None

        # Point cloud as H×W×3 float32 numpy array (NaN where invalid).
        self.latest_xyz   = None

        # Fitted ground plane: (normal_vec [3], d) satisfying normal·p + d = 0.
        # Normal points away from camera (positive Z in camera frame).
        # Height above plane = -(normal·p + d).
        self.ground_plane = None

        # Fallback: baseplate surface Z in camera frame (single-pixel method).
        self.baseplate_z_cam = None

        # --- Subscribers ---
        self.create_subscription(
            Image,        '/camera/camera/color/image_raw',
            self.rgb_callback, 10)
        self.create_subscription(
            Image,        '/camera/camera/depth/image_rect_raw',
            self.depth_callback, 10)
        self.create_subscription(
            CameraInfo,   '/camera/camera/color/camera_info',
            self.camera_info_callback, 10)
        self.create_subscription(
            PointCloud2,  '/camera/camera/depth/color/points',
            self.pointcloud_callback, 10)

        # --- Publishers ---
        self.pose_pub   = self.create_publisher(PoseArray,   '/detected_bricks',      10)
        self.meta_pub   = self.create_publisher(String,      '/detected_bricks_meta', 10)
        self.debug_pub  = self.create_publisher(Image,       '/brick_debug_image',    10)
        self.marker_pub = self.create_publisher(MarkerArray, '/duplo_markers',        10)

        # --- TF ---
        self.tf_buffer             = tf2_ros.Buffer()
        self.tf_listener           = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster        = tf2_ros.TransformBroadcaster(self)
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')

        self.create_timer(0.2, self.process)
        self.get_logger().info('BrickDetectorNode initialized.')

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0];  self.fy = msg.k[4]
        self.cx = msg.k[2];  self.cy = msg.k[5]

    def rgb_callback(self, msg: Image):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg: Image):
        depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_depth = depth_mm.astype(np.float32) / 1000.0

    def pointcloud_callback(self, msg: PointCloud2):
        """Unpack the RealSense organised XYZRGB point cloud into H×W×3 float32."""
        self.latest_xyz = self._unpack_pointcloud(msg)

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

        # ── Ground plane via RANSAC on point cloud ──────────────────────────
        if self.latest_xyz is not None:
            gp = self._fit_ground_plane(self.latest_xyz)
            if gp is not None:
                self.ground_plane = gp

        # ── Table / baseplate depth mask ─────────────────────────────────────
        # Any pixel at (or below) the ground plane is table — not a brick.
        # We erode the mask slightly to handle depth-sensor edge noise.
        above_table_mask = self._make_above_table_mask(depth, rgb.shape)

        # ── Baseplate 2D detection (debug overlay + TF only) ────────────────
        bp = self.detect_baseplate(rgb, depth)
        if bp is not None:
            X, Y, Z, angle_deg, box_pts = bp
            self.baseplate_z_cam = Z   # fallback if point cloud unavailable
            self.publish_baseplate_tf(X, Y, Z, angle_deg)
            cv2.drawContours(debug, [box_pts], 0, (0, 200, 0), 2)
            cv2.putText(debug, 'baseplate', tuple(box_pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        all_bricks = []

        for color_name, ranges in COLOR_RANGES.items():
            mask   = self.color_segment(rgb, ranges)
            # Remove pixels at table/baseplate depth — only keep elevated blobs
            if above_table_mask is not None:
                mask = cv2.bitwise_and(mask, above_table_mask)
            blocks = self.split_by_depth(mask, depth)

            for block_mask in blocks:
                result = self.detect_brick(block_mask, depth, rgb.shape)
                if result is None:
                    continue
                centroid_px, brick_shape, angle_deg = result

                pose_cam = self.compute_3d_pose(centroid_px, angle_deg, depth)
                if pose_cam is None:
                    continue

                # Height from point cloud cluster (falls back to single-pixel)
                height_m, height_type = self._brick_height(block_mask, rgb.shape,
                                                           pose_cam.position.z)

                all_bricks.append({
                    'color':       color_name,
                    'shape':       brick_shape,
                    'height_type': height_type,
                    'height_m':    height_m,
                    'pose':        pose_cam,
                })

                self.draw_detection(debug, block_mask, color_name,
                                    brick_shape, height_type, height_m)

        bricks_base = self.publish_poses(all_bricks)
        self.publish_markers(bricks_base)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

    # -----------------------------------------------------------------------
    # Point cloud helpers
    # -----------------------------------------------------------------------

    def _unpack_pointcloud(self, msg: PointCloud2) -> np.ndarray:
        """
        Unpack an organised RealSense PointCloud2 into an H×W×3 float32 array.

        The RealSense XYZRGB cloud has point_step=32 with X,Y,Z as float32 at
        byte offsets 0, 4, 8.  We read the field offsets from the message so
        this works even if the layout varies between driver versions.
        """
        H, W     = msg.height, msg.width
        step     = msg.point_step
        n_floats = step // 4   # floats per point (step is always a multiple of 4)

        fields = {f.name: f.offset // 4 for f in msg.fields}   # offset in float32 units

        data = np.frombuffer(msg.data, dtype=np.float32).reshape(H * W, n_floats)

        x = data[:, fields['x']].reshape(H, W)
        y = data[:, fields['y']].reshape(H, W)
        z = data[:, fields['z']].reshape(H, W)

        xyz = np.stack([x, y, z], axis=2)
        xyz[~np.isfinite(xyz)] = np.nan
        return xyz

    def _fit_ground_plane(self, xyz: np.ndarray):
        """
        Fit a ground plane to the point cloud using RANSAC.

        Only considers planes whose normal is mostly along the camera Z axis
        (|n_z| > PLANE_NORMAL_MIN_Z), ensuring we find the flat baseplate/table
        rather than a vertical surface.

        Returns (normal, d) where normal·p + d = 0, normal points away from
        the camera (positive Z component).  Height above plane = -(normal·p + d).
        Returns None if fewer than 100 valid points or no plane is found.
        """
        # Subsample for speed
        pts_full = xyz.reshape(-1, 3)
        pts      = pts_full[::PC_SUBSAMPLE]
        valid    = pts[np.all(np.isfinite(pts), axis=1)]

        if len(valid) < 100:
            return None

        best_normal   = None
        best_d        = None
        best_inliers  = 0

        rng = np.random.default_rng()   # seeded per call — reproducible within session

        for _ in range(RANSAC_ITERATIONS):
            idx = rng.choice(len(valid), 3, replace=False)
            p0, p1, p2 = valid[idx]

            normal = np.cross(p1 - p0, p2 - p0)
            norm   = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal /= norm

            # Ground plane normal must be mostly along camera Z axis
            if abs(normal[2]) < PLANE_NORMAL_MIN_Z:
                continue

            # Ensure normal points away from camera (positive Z)
            if normal[2] < 0:
                normal = -normal

            d = -np.dot(normal, p0)

            dists     = np.abs(valid @ normal + d)
            n_inliers = int(np.sum(dists < RANSAC_INLIER_THRESH))

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_normal  = normal
                best_d       = d

        if best_normal is None or best_inliers < 50:
            return None

        # Refine: refit to all inliers
        dists   = np.abs(valid @ best_normal + best_d)
        inliers = valid[dists < RANSAC_INLIER_THRESH]
        if len(inliers) >= 3:
            # Least-squares plane through inliers
            centroid = inliers.mean(axis=0)
            _, _, Vt = np.linalg.svd(inliers - centroid)
            normal   = Vt[-1]
            if normal[2] < 0:
                normal = -normal
            d = -np.dot(normal, centroid)
            best_normal, best_d = normal, d

        self.get_logger().debug(
            f'Ground plane: normal={best_normal.round(3)}, d={best_d:.4f}, '
            f'inliers={best_inliers}'
        )
        return best_normal, best_d

    def _brick_height(self, blob_mask: np.ndarray, rgb_shape: tuple,
                      fallback_z_cam: float) -> tuple:
        """
        Estimate brick height above the ground plane.

        Primary method — point cloud cluster:
          1. Resize blob_mask to match point cloud dimensions if needed.
          2. Extract XYZ points within the mask.
          3. Compute height above fitted ground plane for each point.
          4. Use HEIGHT_PERCENTILE to get the brick top surface (rejects a few
             noisy outliers without discarding valid points).

        Fallback — single-pixel depth (used when point cloud or plane is absent):
          Uses the legacy baseplate_z_cam reference depth.

        Returns (height_m, height_type).
        """
        if self.latest_xyz is not None and self.ground_plane is not None:
            H_pc, W_pc = self.latest_xyz.shape[:2]
            H_rgb, W_rgb = rgb_shape[:2]

            if (H_pc, W_pc) != (H_rgb, W_rgb):
                mask_pc = cv2.resize(blob_mask, (W_pc, H_pc),
                                     interpolation=cv2.INTER_NEAREST)
            else:
                mask_pc = blob_mask

            cluster = self.latest_xyz[mask_pc > 0]
            valid   = cluster[np.all(np.isfinite(cluster), axis=1)]

            if len(valid) >= MIN_CLUSTER_PTS:
                normal, d = self.ground_plane
                # Height above plane: positive = brick sits above the plane
                heights = -(valid @ normal + d)
                heights = heights[heights > 0]

                if len(heights) >= MIN_CLUSTER_PTS // 2:
                    height_m = float(np.percentile(heights, HEIGHT_PERCENTILE))
                    return height_m, self._classify_height(height_m)

        # Fallback: compare single centroid depth to baseplate reference
        if self.baseplate_z_cam is not None:
            height_m = max(0.0, self.baseplate_z_cam - fallback_z_cam)
        else:
            height_m = BRICK_HEIGHTS['normal']   # last-resort default

        return height_m, self._classify_height(height_m)

    def _classify_height(self, height_m: float) -> str:
        """Map a measured height in meters to 'half', 'normal', or 'tall'."""
        for label, (lo, hi) in HEIGHT_THRESHOLDS.items():
            if lo <= height_m < hi:
                return label
        return 'normal'

    def _make_above_table_mask(self, depth: np.ndarray,
                                rgb_shape: tuple) -> np.ndarray:
        """
        Build a uint8 mask (255 = above table, 0 = table/background) by
        comparing each depth pixel to the fitted ground plane.

        Any pixel whose 3D point is more than TABLE_MASK_MARGIN_M above the
        ground plane is kept; everything at table level is zeroed out.

        Returns None if the ground plane hasn't been fitted yet (first few
        frames), in which case callers should skip masking rather than
        discarding all detections.
        """
        TABLE_MASK_MARGIN_M = 0.008   # 8 mm — ignore anything ≤ this above table

        if self.ground_plane is None or self.fx is None:
            return None

        normal, d = self.ground_plane
        H, W      = depth.shape

        # Back-project every depth pixel to 3D in camera frame
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        Z = depth.astype(np.float32)
        X = (uu - self.cx) * Z / self.fx
        Y = (vv - self.cy) * Z / self.fy

        # Height above plane: -(normal · [X,Y,Z] + d)
        height_map = -(X * normal[0] + Y * normal[1] + Z * normal[2] + d)

        # Pixels with invalid depth or at/below table surface → 0
        above = ((height_map > TABLE_MASK_MARGIN_M) &
                 (Z > 0) & np.isfinite(Z)).astype(np.uint8) * 255

        # Erode slightly to trim noisy brick edges that straddle the table
        k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        above = cv2.erode(above, k, iterations=1)

        # Resize to RGB image dimensions if depth resolution differs
        if (H, W) != rgb_shape[:2]:
            above = cv2.resize(above, (rgb_shape[1], rgb_shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        return above

    # -----------------------------------------------------------------------
    # Color / blob helpers
    # -----------------------------------------------------------------------

    def color_segment(self, rgb: np.ndarray, ranges: list) -> np.ndarray:
        """OR together all HSV ranges and return a cleaned binary mask."""
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask,
                                  cv2.inRange(hsv, np.array(lower), np.array(upper)))
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def split_by_depth(self, color_mask: np.ndarray, depth: np.ndarray) -> list:
        """Split a colour mask into individual brick blobs via depth discontinuities."""
        DEPTH_EDGE_THRESH = 0.02
        blobs = []
        num_labels, labels = cv2.connectedComponents(color_mask)

        for label in range(1, num_labels):
            blob = (labels == label).astype(np.uint8) * 255

            depth_in_blob = np.where(blob > 0, depth, 0.0).astype(np.float32)
            gx      = cv2.Sobel(depth_in_blob, cv2.CV_32F, 1, 0, ksize=3)
            gy      = cv2.Sobel(depth_in_blob, cv2.CV_32F, 0, 1, ksize=3)
            edges   = ((np.sqrt(gx**2 + gy**2) > DEPTH_EDGE_THRESH) & (blob > 0)).astype(np.uint8) * 255
            split   = cv2.bitwise_and(blob, cv2.bitwise_not(edges))

            k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            split = cv2.morphologyEx(split, cv2.MORPH_CLOSE, k)

            n_sub, sub_labels = cv2.connectedComponents(split)
            if n_sub <= 2:
                blobs.append(blob)
            else:
                for s in range(1, n_sub):
                    blobs.append((sub_labels == s).astype(np.uint8) * 255)

        return blobs

    # -----------------------------------------------------------------------
    # Brick detection — footprint-based (replaces stud counting)
    #
    # Root cause of the 1×1 problem: HoughCircles needs every stud to be
    # clearly visible and perfectly circular.  In practice, lighting, focus,
    # and viewing angle cause it to miss most studs, collapsing every brick
    # to a single detected circle → (1,1) shape.
    #
    # Fix: measure the *physical footprint* of the colour blob instead.
    #   • Orientation — minAreaRect on the blob contour (robust, one-shot)
    #   • Shape       — 3D cluster extent projected onto the ground plane
    #                   (primary), or 2D blob size ÷ depth (fallback)
    #   • Centroid    — centre of minAreaRect (one clean point → pose)
    # -----------------------------------------------------------------------

    def detect_brick(self, blob_mask: np.ndarray, depth: np.ndarray,
                     rgb_shape: tuple):
        """
        Detect a brick from its colour blob.

        1. Find the largest contour in blob_mask.
        2. Fit a minAreaRect → centroid pixel, long-axis angle.
        3. Infer shape from physical footprint (3D primary, 2D fallback).

        Returns ([(cx_px, cy_px)], (rows, cols), angle_deg) or None.
        The centroid is returned as a single-element list so compute_3d_pose
        (which averages a list of pixel coords) works without modification.
        """
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < BLOB_MIN_AREA_PX:
            return None

        rect = cv2.minAreaRect(largest)
        (cx_px, cy_px), (w_px, h_px), angle = rect

        # minAreaRect convention: width is along `angle`, height is perpendicular.
        # Ensure (w_px, angle) always describe the LONG axis so cols ≥ rows.
        if w_px < h_px:
            w_px, h_px = h_px, w_px
            angle += 90.0

        # Infer shape — try point cloud first, fall back to 2D
        shape = self._shape_from_pointcloud(blob_mask, rgb_shape)
        if shape is None:
            shape = self._shape_from_blob_2d(w_px, h_px, cx_px, cy_px, depth)

        centroid = [(int(cx_px), int(cy_px))]
        return centroid, shape, angle

    def _shape_from_pointcloud(self, blob_mask: np.ndarray,
                                rgb_shape: tuple):
        """
        Infer brick shape by projecting its 3D point cluster onto the fitted
        ground plane, then measuring the cluster's XY extent along its two
        principal axes and snapping to the nearest Duplo stud count.

        Returns (rows, cols) or None if the point cloud / ground plane is
        unavailable or the cluster is too small.
        """
        if self.latest_xyz is None or self.ground_plane is None:
            return None

        H_pc, W_pc = self.latest_xyz.shape[:2]
        H_rgb, W_rgb = rgb_shape[:2]

        if (H_pc, W_pc) != (H_rgb, W_rgb):
            mask_pc = cv2.resize(blob_mask, (W_pc, H_pc),
                                 interpolation=cv2.INTER_NEAREST)
        else:
            mask_pc = blob_mask

        cluster = self.latest_xyz[mask_pc > 0]
        valid   = cluster[np.all(np.isfinite(cluster), axis=1)]

        if len(valid) < MIN_CLUSTER_PTS:
            return None

        normal, _ = self.ground_plane

        # Build an orthonormal basis for the ground plane so we can work in 2D.
        # v1, v2 span the plane; normal is the out-of-plane direction.
        ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v1  = np.cross(normal, ref);  v1 /= np.linalg.norm(v1)
        v2  = np.cross(normal, v1);   v2 /= np.linalg.norm(v2)

        # Project cluster onto the plane (2D coordinates in the plane's frame)
        proj = np.column_stack([valid @ v1, valid @ v2])

        # PCA on the 2D projections → principal axes and their extents
        centered = proj - proj.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        proj_pca = centered @ Vt.T          # rotate to PCA frame

        long_m  = float(proj_pca[:, 0].max() - proj_pca[:, 0].min())
        short_m = float(proj_pca[:, 1].max() - proj_pca[:, 1].min())

        cols = max(1, min(8, round(long_m  / STUD_PITCH_M)))
        rows = max(1, min(4, round(short_m / STUD_PITCH_M)))

        return (rows, cols)

    def _shape_from_blob_2d(self, w_px: float, h_px: float,
                             cx_px: float, cy_px: float,
                             depth: np.ndarray):
        """
        Fallback shape inference from the 2D bounding-box pixel size.

        Converts pixel dimensions to meters using the brick's depth and
        the camera focal length, then snaps to the Duplo stud grid.
        """
        ci, ri  = int(cx_px), int(cy_px)
        dh, dw  = depth.shape
        window  = depth[max(0, ri-3):min(dh, ri+4),
                        max(0, ci-3):min(dw, ci+4)]
        valid   = window[(window > 0) & np.isfinite(window)]
        if len(valid) == 0:
            return (1, 1)
        Z = float(np.median(valid))

        f_avg   = (self.fx + self.fy) / 2.0
        long_m  = w_px * Z / f_avg   # w_px is guaranteed ≥ h_px (long axis)
        short_m = h_px * Z / f_avg

        cols = max(1, min(8, round(long_m  / STUD_PITCH_M)))
        rows = max(1, min(4, round(short_m / STUD_PITCH_M)))

        return (rows, cols)

    def compute_3d_pose(self, stud_centers: list, angle_deg: float,
                        depth: np.ndarray) -> Pose:
        """Back-project brick centroid to 3D camera frame."""
        if not stud_centers: return None

        u, v = (np.mean([c[0] for c in stud_centers]),
                np.mean([c[1] for c in stud_centers]))
        ui, vi = int(u), int(v)

        h, w   = depth.shape
        half   = 2
        window = depth[max(0,vi-half):min(h,vi+half+1),
                       max(0,ui-half):min(w,ui+half+1)]
        valid  = window[(window > 0) & np.isfinite(window)]
        if len(valid) == 0:
            self.get_logger().warn(f'Invalid depth near ({ui}, {vi})')
            return None
        Z = float(np.median(valid))

        pose = Pose()
        pose.position.x = float((u - self.cx) * Z / self.fx)
        pose.position.y = float((v - self.cy) * Z / self.fy)
        pose.position.z = Z

        q = Rotation.from_euler('z', angle_deg, degrees=True).as_quat()
        pose.orientation.x, pose.orientation.y = q[0], q[1]
        pose.orientation.z, pose.orientation.w = q[2], q[3]
        return pose

    # -----------------------------------------------------------------------
    # Baseplate detection (2D — for debug overlay + TF broadcast)
    # -----------------------------------------------------------------------

    def detect_baseplate(self, rgb: np.ndarray, depth: np.ndarray):
        """Detect baseplate via HSV. Returns (X,Y,Z,angle,box_pts) or None."""
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(BASEPLATE_HSV_LOWER),
                               np.array(BASEPLATE_HSV_UPPER))
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k),
                                cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < BASEPLATE_MIN_AREA_PX:
            self.get_logger().warn('Baseplate blob too small', once=True)
            return None

        rect = cv2.minAreaRect(largest)
        (cx_px, cy_px), (w_px, h_px), angle_deg = rect
        if w_px < h_px: angle_deg += 90.0
        box_pts = np.int0(cv2.boxPoints(rect))

        ci, ri  = int(cx_px), int(cy_px)
        ih, iw  = depth.shape
        window  = depth[max(0,ri-5):min(ih,ri+6), max(0,ci-5):min(iw,ci+6)]
        valid   = window[(window > 0) & np.isfinite(window)]
        if len(valid) == 0: return None
        Z = float(np.median(valid))
        X = (cx_px - self.cx) * Z / self.fx
        Y = (cy_px - self.cy) * Z / self.fy
        return X, Y, Z, angle_deg, box_pts

    def publish_baseplate_tf(self, X, Y, Z, angle_deg):
        """Broadcast baseplate_frame as a static TF relative to base_link."""
        camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        pose_cam = Pose()
        pose_cam.position.x, pose_cam.position.y, pose_cam.position.z = X, Y, Z
        q = Rotation.from_euler('z', angle_deg, degrees=True).as_quat()
        pose_cam.orientation.x, pose_cam.orientation.y = q[0], q[1]
        pose_cam.orientation.z, pose_cam.orientation.w = q[2], q[3]

        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', camera_frame,
                rclpy.time.Time(), timeout=Duration(seconds=TF_TIMEOUT_SEC))
            pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam, tf)
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

    # -----------------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------------

    def publish_poses(self, bricks: list) -> list:
        """
        Transform detected bricks to base_link, publish PoseArray + meta JSON.
        Returns list of bricks enriched with 'pose_base' for publish_markers.
        Meta JSON now includes 'height_m' (measured float) alongside 'height_type'.
        """
        camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        pose_array             = PoseArray()
        pose_array.header      = Header()
        pose_array.header.stamp    = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'base_link'

        meta_list   = []
        bricks_base = []

        for brick in bricks:
            try:
                tf        = self.tf_buffer.lookup_transform(
                    'base_link', camera_frame,
                    rclpy.time.Time(), timeout=Duration(seconds=TF_TIMEOUT_SEC))
                pose_base = tf2_geometry_msgs.do_transform_pose(brick['pose'], tf)

                pose_array.poses.append(pose_base)
                meta_list.append({
                    'color':       brick['color'],
                    'shape':       list(brick['shape']),
                    'height_type': brick['height_type'],
                    'height_m':    round(brick['height_m'] * 1000, 1),  # mm, 1 dp
                })
                bricks_base.append({**brick, 'pose_base': pose_base})

            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f'TF lookup failed: {e}')

        self.pose_pub.publish(pose_array)
        meta_msg      = String()
        meta_msg.data = json.dumps(meta_list)
        self.meta_pub.publish(meta_msg)
        plane_status = 'fitted' if self.ground_plane else 'fallback depth'
        self.get_logger().info(
            f'Published {len(pose_array.poses)} bricks. Ground plane: {plane_status}'
        )
        return bricks_base

    def publish_markers(self, bricks_base: list):
        """Publish RViz MarkerArray: baseplate box, stud grid, brick boxes, labels."""
        now     = self.get_clock().now().to_msg()
        markers = MarkerArray()

        clear              = Marker()
        clear.action       = Marker.DELETEALL
        clear.header.frame_id = 'base_link'
        clear.header.stamp = now
        markers.markers.append(clear)

        bp_w = BASEPLATE_COLS * STUD_PITCH_M
        bp_d = BASEPLATE_ROWS * STUD_PITCH_M

        # Baseplate flat box
        bp              = Marker()
        bp.header.frame_id = 'baseplate_frame'
        bp.header.stamp = now
        bp.ns, bp.id    = 'baseplate', 0
        bp.type         = Marker.CUBE
        bp.action       = Marker.ADD
        bp.pose.position.x = bp_w / 2.0
        bp.pose.position.y = bp_d / 2.0
        bp.pose.position.z = -0.005
        bp.pose.orientation.w = 1.0
        bp.scale.x, bp.scale.y, bp.scale.z = bp_w, bp_d, 0.008
        bp.color.r, bp.color.g, bp.color.b, bp.color.a = 0.10, 0.45, 0.15, 0.85
        markers.markers.append(bp)

        # Stud grid lines
        grid                 = Marker()
        grid.header.frame_id = 'baseplate_frame'
        grid.header.stamp    = now
        grid.ns, grid.id     = 'baseplate', 1
        grid.type            = Marker.LINE_LIST
        grid.action          = Marker.ADD
        grid.pose.orientation.w = 1.0
        grid.scale.x         = 0.0008
        grid.color.r, grid.color.g, grid.color.b, grid.color.a = 0.05, 0.25, 0.05, 0.70
        Z_GRID = 0.001
        for row in range(BASEPLATE_ROWS + 1):
            y = row * STUD_PITCH_M
            p0 = Point(); p0.x = 0.0;  p0.y = y; p0.z = Z_GRID
            p1 = Point(); p1.x = bp_w; p1.y = y; p1.z = Z_GRID
            grid.points += [p0, p1]
        for col in range(BASEPLATE_COLS + 1):
            x = col * STUD_PITCH_M
            p0 = Point(); p0.x = x; p0.y = 0.0;  p0.z = Z_GRID
            p1 = Point(); p1.x = x; p1.y = bp_d; p1.z = Z_GRID
            grid.points += [p0, p1]
        markers.markers.append(grid)

        for i, brick in enumerate(bricks_base):
            br, bc      = brick['shape']
            height_type = brick['height_type']
            height_m    = brick['height_m']
            pose_base   = brick['pose_base']
            brick_h     = BRICK_HEIGHTS.get(height_type, BRICK_HEIGHTS['normal'])
            rgba        = BRICK_COLORS_RGBA.get(brick['color'], (0.8, 0.8, 0.8, 1.0))

            box                 = Marker()
            box.header.frame_id = 'base_link'
            box.header.stamp    = now
            box.ns, box.id      = 'bricks', i
            box.type            = Marker.CUBE
            box.action          = Marker.ADD
            box.pose            = pose_base
            box.scale.x         = bc * STUD_PITCH_M
            box.scale.y         = br * STUD_PITCH_M
            box.scale.z         = brick_h
            box.color.r, box.color.g = rgba[0], rgba[1]
            box.color.b, box.color.a = rgba[2], rgba[3]
            markers.markers.append(box)

            txt                 = Marker()
            txt.header.frame_id = 'base_link'
            txt.header.stamp    = now
            txt.ns, txt.id      = 'brick_labels', i
            txt.type            = Marker.TEXT_VIEW_FACING
            txt.action          = Marker.ADD
            txt.pose.position.x    = pose_base.position.x
            txt.pose.position.y    = pose_base.position.y
            txt.pose.position.z    = pose_base.position.z + brick_h + 0.025
            txt.pose.orientation.w = 1.0
            txt.scale.z            = 0.020
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = f'{brick["color"]}\n{br}x{bc} [{height_type}] {height_m*1000:.0f}mm'
            markers.markers.append(txt)

        self.marker_pub.publish(markers)

    def draw_detection(self, img: np.ndarray, blob_mask: np.ndarray,
                       color_name: str, brick_shape: tuple,
                       height_type: str, height_m: float):
        """
        Annotate the 2D debug image with:
          - Rotated bounding box of the detected blob (green)
          - Label: colour, shape, height type, measured height in mm
        """
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        largest = max(contours, key=cv2.contourArea)
        rect    = cv2.minAreaRect(largest)
        box     = np.int0(cv2.boxPoints(rect))

        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

        cx, cy = int(rect[0][0]), int(rect[0][1])
        br, bc = brick_shape
        label  = f'{color_name} {br}x{bc} [{height_type}] {height_m*1000:.0f}mm'
        cv2.putText(img, label, (cx - 30, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        # Small dot at the centroid so the TCP target point is visible
        cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)


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
