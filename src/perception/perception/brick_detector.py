import json

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import cv2
import numpy as np
import open3d as o3d
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Pose, PoseArray, Point, TransformStamped
from std_msgs.msg import Bool, Header, String
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation


COLOR_RANGES = {
    'red':          [(( 42, 149, 130), (136, 195, 173))],
    'orange':       [(( 87, 146, 156), (202, 183, 183))],
    'yellow':       [((122, 113, 151), (207, 146, 191))],
    'light_green':  [(( 97,  85, 128), (161, 110, 166))],
    'blue':         [(( 88, 106,  80), (172, 126, 103))],
    'mint':         [((111,  87, 117), (228, 114, 134))],
    'white':        [((167, 113,  93), (209, 140, 118))],
    'purple':       [(( 89, 129,  77), (169, 151, 116))],
    'brown':        [(( 52, 126, 124), (154, 155, 146))],
    'pink':         [((117, 159, 119), (184, 189, 152))],
    'light_blue':   [((132, 105,  85), (255, 140, 130))],
}

BRICK_COLORS_RGBA = {
    'red':         (1.00, 0.15, 0.15, 1.0),
    'orange':      (1.00, 0.50, 0.00, 1.0),
    'yellow':      (1.00, 0.90, 0.00, 1.0),
    'light_green': (0.50, 1.00, 0.20, 1.0),
    'blue':        (0.30, 0.70, 1.00, 1.0),
    'mint':        (0.60, 1.00, 0.82, 1.0),
    'white':       (0.95, 0.95, 0.95, 1.0),
    'purple':      (0.60, 0.20, 0.85, 1.0),
    'brown':       (0.55, 0.27, 0.07, 1.0),
    'pink':        (1.00, 0.08, 0.58, 1.0),
    'light_blue':  (0.68, 0.85, 0.90, 1.0),
}

BASEPLATE_HSV_LOWER   = (55,  80,  30)
BASEPLATE_HSV_UPPER   = (80, 200, 100)
BASEPLATE_MIN_AREA_PX = 5000

ARUCO_DICT_ID       = cv2.aruco.DICT_4X4_50
ARUCO_MARKER_ID     = 0
ARUCO_MARKER_SIZE_M = 0.05

STUD_PITCH_M           = 0.016
MAX_BRICK_FOOTPRINT_M2 = (2 * STUD_PITCH_M) * (6 * STUD_PITCH_M) * 2.0
MIN_BRICK_FOOTPRINT_M2 = (1 * STUD_PITCH_M) * (2 * STUD_PITCH_M) * 0.4

VALID_BRICK_SHAPES = {(1, 2), (2, 2), (2, 4), (2, 6)}

BASEPLATE_ROWS = 16
BASEPLATE_COLS = 16

ARUCO_TO_BP_OFFSET = np.array([
    -(BASEPLATE_COLS * STUD_PITCH_M - ARUCO_MARKER_SIZE_M / 2.0),
    -(BASEPLATE_ROWS * STUD_PITCH_M - ARUCO_MARKER_SIZE_M / 2.0),
    0.0,
], dtype=np.float64)

BRICK_HEIGHTS = {
    'half':   0.0096,
    'normal': 0.0192,
    'tall':   0.0384,
}

HEIGHT_THRESHOLDS = {
    'half':   (0.000, 0.013),
    'normal': (0.013, 0.030),
    'tall':   (0.030, 9.999),
}

HEIGHT_PERCENTILE    = 90
MIN_CLUSTER_PTS      = 10
TABLE_MASK_MARGIN_M  = 0.005
MAX_BRICK_HEIGHT_M   = 0.060
VOXEL_SIZE_M         = 0.002
DBSCAN_EPS_M         = 0.012
DBSCAN_MIN_PTS       = 10
COLOR_MATCH_MIN_FRAC = 0.40
TF_TIMEOUT_SEC       = 1.0
ARUCO_WINDOW         = 3


class BrickDetectorNode(Node):

    def __init__(self):
        super().__init__('brick_detector')
        self.bridge = CvBridge()

        self.fx = self.fy = self.cx = self.cy = None
        self.dist_coeffs   = np.zeros(5, dtype=np.float32)
        self.latest_rgb    = None
        self.latest_depth  = None
        self.latest_xyz    = None
        self.latest_pc_bgr = None
        self.baseplate_z_cam    = None
        self._aruco_window: list = []
        self._last_baseplate_tf  = None

        self.create_subscription(
            Image,       '/camera/camera/color/image_raw',
            self.rgb_callback, 10)
        self.create_subscription(
            Image,       '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback, 10)
        self.create_subscription(
            CameraInfo,  '/camera/camera/color/camera_info',
            self.camera_info_callback, 10)
        self.create_subscription(
            PointCloud2, '/camera/camera/depth/color/points',
            self.pointcloud_callback, 10)

        self.pose_pub        = self.create_publisher(PoseArray,   '/detected_bricks',      10)
        self.meta_pub        = self.create_publisher(String,      '/detected_bricks_meta', 10)
        self.debug_pub       = self.create_publisher(Image,       '/brick_debug_image',    10)
        self.aruco_debug_pub = self.create_publisher(Image,       '/aruco_debug_image',    10)
        self.marker_pub      = self.create_publisher(MarkerArray, '/duplo_markers',        10)

        self.tf_buffer             = tf2_ros.Buffer()
        self.tf_listener           = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster        = tf2_ros.TransformBroadcaster(self)
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')

        self._enabled          = False
        self._baseplate_locked = False

        self.create_subscription(
            Bool, '/brick_detection_enabled',
            lambda msg: setattr(self, '_enabled', msg.data), 10)

        self._frame_validated = False
        self.create_timer(0.5, self.process)
        self.create_timer(2.0, self._validate_camera_frame)
        self.get_logger().info('BrickDetectorNode initialized.')

    def _validate_camera_frame(self):
        if self._frame_validated:
            return
        camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        try:
            self.tf_buffer.lookup_transform('base_link', camera_frame, rclpy.time.Time())
            self.get_logger().info(f"Camera frame '{camera_frame}' validated OK.")
            self._frame_validated = True
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            self.get_logger().warn(
                f"Camera frame '{camera_frame}' not found in TF tree.",
                throttle_duration_sec=5.0)

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]; self.fy = msg.k[4]
        self.cx = msg.k[2]; self.cy = msg.k[5]
        if msg.d:
            self.dist_coeffs = np.array(msg.d, dtype=np.float32)

    def rgb_callback(self, msg: Image):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg: Image):
        depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_depth = depth_mm.astype(np.float32) / 1000.0

    def pointcloud_callback(self, msg: PointCloud2):
        self.latest_xyz, self.latest_pc_bgr = self._unpack_pointcloud(msg)

    def _cam_to_baseplate(self) -> np.ndarray | None:
        camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                'baseplate_frame', camera_frame,
                rclpy.time.Time(), timeout=Duration(seconds=TF_TIMEOUT_SEC))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'cam→baseplate_frame TF failed: {e}', throttle_duration_sec=5.0)
            return None
        t     = tf_msg.transform.translation
        q     = tf_msg.transform.rotation
        R_mat = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T     = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3]  = [t.x, t.y, t.z]
        return T

    def process(self):
        if self.latest_rgb is None:
            return

        rgb   = self.latest_rgb.copy()
        debug = rgb.copy()

        if self.fx is not None and not self._baseplate_locked:
            bp = self.detect_baseplate_aruco(rgb)
            if bp is None and self.latest_depth is not None:
                bp = self.detect_baseplate(rgb, self.latest_depth.copy())
            if bp is not None:
                X, Y, Z, angle_deg, box_pts = bp
                self.baseplate_z_cam = Z
                self.publish_baseplate_tf(X, Y, Z, angle_deg)
                cv2.drawContours(debug, [box_pts], 0, (0, 200, 0), 2)
                cv2.putText(debug, 'baseplate', tuple(box_pts[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            else:
                self.get_logger().warn(
                    'Baseplate not detected (ArUco + HSV both failed)',
                    throttle_duration_sec=5.0)
                if self._last_baseplate_tf is not None:
                    avg_t, avg_q = self._last_baseplate_tf
                    t                         = TransformStamped()
                    t.header.stamp            = self.get_clock().now().to_msg()
                    t.header.frame_id         = 'base_link'
                    t.child_frame_id          = 'baseplate_frame'
                    t.transform.translation.x = float(avg_t[0])
                    t.transform.translation.y = float(avg_t[1])
                    t.transform.translation.z = float(avg_t[2])
                    t.transform.rotation.x    = float(avg_q[0])
                    t.transform.rotation.y    = float(avg_q[1])
                    t.transform.rotation.z    = float(avg_q[2])
                    t.transform.rotation.w    = float(avg_q[3])
                    self.tf_broadcaster.sendTransform(t)

        if not self._enabled:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
            return

        if not self._baseplate_locked:
            self.get_logger().warn(
                'Detection enabled but baseplate not yet locked.',
                throttle_duration_sec=5.0)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
            return

        if self.latest_xyz is None or self.latest_pc_bgr is None:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
            return

        cam_to_bp = self._cam_to_baseplate()
        if cam_to_bp is None:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
            return

        clusters   = self._cluster_above_table(self.latest_xyz, self.latest_pc_bgr.copy(), cam_to_bp)
        self.get_logger().info(
            f'[detect] {len(clusters)} cluster(s) above table',
            throttle_duration_sec=2.0)
        all_bricks = []

        for ci_idx, (cluster_pts_bp, cluster_bgr) in enumerate(clusters):
            color_name = self._classify_cluster_color(cluster_bgr)
            if color_name is None:
                self.get_logger().info(
                    f'  cluster {ci_idx}: {len(cluster_pts_bp)} pts — color unclassified, skipping',
                    throttle_duration_sec=2.0)
                continue

            footprint = self._cluster_footprint_m2(cluster_pts_bp)
            if footprint < MIN_BRICK_FOOTPRINT_M2:
                self.get_logger().info(
                    f'  cluster {ci_idx}: {color_name} — footprint {footprint*1e4:.1f}cm² < min {MIN_BRICK_FOOTPRINT_M2*1e4:.1f}cm², skipping',
                    throttle_duration_sec=2.0)
                continue

            rows, cols, angle_deg = self._shape_from_pointcloud_extent(cluster_pts_bp)
            shape_key = (min(rows, cols), max(rows, cols))
            if shape_key not in VALID_BRICK_SHAPES:
                self.get_logger().info(
                    f'  cluster {ci_idx}: {color_name} — shape {rows}x{cols} not valid, skipping',
                    throttle_duration_sec=2.0)
                continue

            heights_z   = cluster_pts_bp[:, 2]
            heights_pos = heights_z[heights_z > 0]
            height_m    = (float(np.percentile(heights_pos, HEIGHT_PERCENTILE))
                           if len(heights_pos) >= MIN_CLUSTER_PTS // 2
                           else BRICK_HEIGHTS['normal'])
            height_type = self._classify_height(height_m)

            top_mask    = cluster_pts_bp[:, 2] >= (height_m - 0.003)
            top_pts     = cluster_pts_bp[top_mask]
            centroid_bp = np.median(
                top_pts if len(top_pts) >= MIN_CLUSTER_PTS else cluster_pts_bp,
                axis=0)

            pose_bp = Pose()
            pose_bp.position.x = float(centroid_bp[0])
            pose_bp.position.y = float(centroid_bp[1])
            pose_bp.position.z = float(centroid_bp[2])
            q = Rotation.from_euler('z', angle_deg, degrees=True).as_quat()
            pose_bp.orientation.x = float(q[0])
            pose_bp.orientation.y = float(q[1])
            pose_bp.orientation.z = float(q[2])
            pose_bp.orientation.w = float(q[3])

            try:
                tf_to_base = self.tf_buffer.lookup_transform(
                    'base_link', 'baseplate_frame',
                    rclpy.time.Time(), timeout=Duration(seconds=TF_TIMEOUT_SEC))
                pose_base = tf2_geometry_msgs.do_transform_pose(pose_bp, tf_to_base)
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f'base_link←baseplate_frame TF failed: {e}')
                continue

            all_bricks.append({
                'color':       color_name,
                'shape':       (rows, cols),
                'height_type': height_type,
                'height_m':    height_m,
                'pose':        pose_base,
            })

        bricks_out = self.publish_poses(all_bricks)
        self.publish_markers(bricks_out)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

    def _unpack_pointcloud(self, msg: PointCloud2):
        H, W     = msg.height, msg.width
        step     = msg.point_step
        n_floats = step // 4
        fields   = {f.name: f.offset // 4 for f in msg.fields}
        data     = np.frombuffer(msg.data, dtype=np.float32).reshape(H * W, n_floats)

        x = data[:, fields['x']].reshape(H, W)
        y = data[:, fields['y']].reshape(H, W)
        z = data[:, fields['z']].reshape(H, W)
        xyz = np.stack([x, y, z], axis=2)
        xyz[~np.isfinite(xyz)] = np.nan

        rgb_u32 = data[:, fields['rgb']].view(np.uint32)
        r = ((rgb_u32 >> 16) & 0xFF).astype(np.uint8).reshape(H, W)
        g = ((rgb_u32 >>  8) & 0xFF).astype(np.uint8).reshape(H, W)
        b = (rgb_u32         & 0xFF).astype(np.uint8).reshape(H, W)
        bgr = np.stack([b, g, r], axis=2)
        return xyz, bgr

    def _cluster_above_table(self, xyz_cam: np.ndarray, bgr: np.ndarray,
                             cam_to_bp: np.ndarray) -> list:
        pts_cam = xyz_cam.reshape(-1, 3)
        colors  = bgr.reshape(-1, 3)

        valid     = np.all(np.isfinite(pts_cam), axis=1)
        pts_h     = np.hstack([pts_cam[valid], np.ones((int(valid.sum()), 1))])
        pts_bp    = (cam_to_bp @ pts_h.T).T[:, :3]
        col_valid = colors[valid]

        z    = pts_bp[:, 2]
        keep = (z > TABLE_MASK_MARGIN_M) & (z < MAX_BRICK_HEIGHT_M)

        n_keep = int(np.sum(keep))
        self.get_logger().info(
            f'[table filter] z range [{z.min():.3f}, {z.max():.3f}]m  '
            f'keep ({TABLE_MASK_MARGIN_M:.3f}<z<{MAX_BRICK_HEIGHT_M:.3f}): {n_keep}/{len(z)} pts',
            throttle_duration_sec=2.0)

        if n_keep < DBSCAN_MIN_PTS:
            return []

        above_pts = pts_bp[keep]
        above_col = col_valid[keep]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(above_pts)
        pcd.colors = o3d.utility.Vector3dVector(above_col.astype(np.float64) / 255.0)

        pcd_down    = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE_M)
        pts_down    = np.asarray(pcd_down.points)
        colors_down = (np.asarray(pcd_down.colors) * 255).astype(np.uint8)

        labels = np.array(pcd_down.cluster_dbscan(
            eps=DBSCAN_EPS_M, min_points=DBSCAN_MIN_PTS, print_progress=False))

        unique_lbls = [lbl for lbl in np.unique(labels) if lbl >= 0]
        self.get_logger().info(
            f'[dbscan] {len(pts_down)} pts after voxel → {len(unique_lbls)} cluster(s)',
            throttle_duration_sec=2.0)
        return [
            (pts_down[labels == lbl], colors_down[labels == lbl])
            for lbl in unique_lbls
        ]

    def _cluster_footprint_m2(self, pts_bp: np.ndarray) -> float:
        if len(pts_bp) < 3:
            return 0.0
        proj = pts_bp[:, :2].astype(np.float32)
        hull = cv2.convexHull(proj.reshape(-1, 1, 2))
        return float(cv2.contourArea(hull))

    def _classify_cluster_color(self, cluster_bgr: np.ndarray):
        lab_pts = cv2.cvtColor(
            cluster_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
        ).reshape(-1, 3).astype(np.float32)

        l_vals  = lab_pts[:, 0]
        valid   = (l_vals > 30) & (l_vals < 240)
        lab_pts = lab_pts[valid]
        if len(lab_pts) < 5:
            return None

        def match_mask(pts, ranges):
            mask = np.zeros(len(pts), dtype=bool)
            for lower, upper in ranges:
                mask |= np.all(
                    (pts >= np.array(lower, np.float32)) &
                    (pts <= np.array(upper, np.float32)), axis=1)
            return mask

        orange_mask = match_mask(lab_pts, COLOR_RANGES['orange'])
        pink_mask   = match_mask(lab_pts, COLOR_RANGES['pink'])

        best_color = None
        best_count = 0
        for color_name, ranges in COLOR_RANGES.items():
            m = match_mask(lab_pts, ranges)
            if color_name == 'red':
                m = m & ~orange_mask & ~pink_mask
            count = int(np.sum(m))
            if count > best_count:
                best_count = count
                best_color = color_name

        if best_count < len(lab_pts) * COLOR_MATCH_MIN_FRAC:
            return None
        return best_color

    def _shape_from_pointcloud_extent(self, pts_bp: np.ndarray) -> tuple:
        if len(pts_bp) < MIN_CLUSTER_PTS:
            return 1, 1, 0.0

        xy       = pts_bp[:, :2]
        centered = xy - xy.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        proj_pca = centered @ Vt.T

        long_m  = float(np.percentile(proj_pca[:, 0], 95) - np.percentile(proj_pca[:, 0], 5))
        short_m = float(np.percentile(proj_pca[:, 1], 95) - np.percentile(proj_pca[:, 1], 5))

        cols = max(1, min(8, round(long_m  / STUD_PITCH_M)))
        rows = max(1, min(4, round(short_m / STUD_PITCH_M)))

        long_vec  = np.array([Vt[0, 0], Vt[0, 1]])
        angle_deg = 0.0 if rows == cols else float(np.degrees(np.arctan2(long_vec[1], long_vec[0])))
        return rows, cols, angle_deg

    def _classify_height(self, height_m: float) -> str:
        for label, (lo, hi) in HEIGHT_THRESHOLDS.items():
            if lo <= height_m < hi:
                return label
        return 'normal'

    def detect_baseplate_aruco(self, rgb: np.ndarray):
        if self.fx is None:
            return None

        gray            = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        clahe           = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_clahe      = clahe.apply(gray)
        gray_blur_clahe = clahe.apply(cv2.GaussianBlur(gray, (5, 5), 1.0))

        try:
            self.aruco_debug_pub.publish(
                self.bridge.cv2_to_imgmsg(gray_clahe, encoding='mono8'))
        except Exception:
            pass

        attempts = [
            ('raw+permissive',        gray,            0.01, 3,   53,  4),
            ('clahe+permissive',      gray_clahe,      0.01, 3,   53,  4),
            ('raw+default',           gray,            0.03, 3,   23, 10),
            ('clahe+default',         gray_clahe,      0.03, 3,   23, 10),
            ('blur_clahe+permissive', gray_blur_clahe, 0.01, 3,   53,  4),
            ('raw+wide',              gray,            0.01, 3,  103,  8),
        ]

        for label, img, min_perim, amin, amax, astep in attempts:
            result = self._aruco_detect_on(img, label, min_perim, amin, amax, astep)
            if result is not None:
                return result

        self.get_logger().warn(
            'ArUco: no markers detected after all attempts — '
            'check DICT_4X4_50 marker ID 0 is visible and well-lit',
            throttle_duration_sec=5.0)
        return None

    def _aruco_detect_on(self, gray: np.ndarray, label: str,
                         min_perim: float = 0.01,
                         adapt_min: int = 3, adapt_max: int = 53, adapt_step: int = 4):
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
            params     = cv2.aruco.DetectorParameters()
            params.minMarkerPerimeterRate    = min_perim
            params.adaptiveThreshWinSizeMin  = adapt_min
            params.adaptiveThreshWinSizeMax  = adapt_max
            params.adaptiveThreshWinSizeStep = adapt_step
            params.errorCorrectionRate       = 0.8
            try:
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            except AttributeError:
                pass
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_ID)
            params     = cv2.aruco.DetectorParameters_create()
            params.minMarkerPerimeterRate    = min_perim
            params.adaptiveThreshWinSizeMin  = adapt_min
            params.adaptiveThreshWinSizeMax  = adapt_max
            params.adaptiveThreshWinSizeStep = adapt_step
            params.errorCorrectionRate       = 0.8
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        if ids is None:
            return None

        self.get_logger().info(
            f'ArUco [{label}]: found markers {ids.flatten().tolist()} — '
            f'looking for ID {ARUCO_MARKER_ID}',
            throttle_duration_sec=2.0)

        cam_mat = np.array([[self.fx, 0, self.cx],
                            [0, self.fy, self.cy],
                            [0,       0,       1]], dtype=np.float32)

        for i, mid in enumerate(ids.flatten()):
            if mid != ARUCO_MARKER_ID:
                continue
            half    = ARUCO_MARKER_SIZE_M / 2.0
            obj_pts = np.array([
                [-half,  half, 0],
                [ half,  half, 0],
                [ half, -half, 0],
                [-half, -half, 0],
            ], dtype=np.float32)
            _, rvec, tvec = cv2.solvePnP(
                obj_pts, corners[i].reshape(4, 2), cam_mat, self.dist_coeffs)
            rvec     = rvec.flatten()
            tvec     = tvec.flatten()
            R_mat, _ = cv2.Rodrigues(rvec)
            origin_cam = tvec + R_mat @ ARUCO_TO_BP_OFFSET
            X         = float(origin_cam[0])
            Y         = float(origin_cam[1])
            Z         = float(origin_cam[2])
            angle_deg = float(np.degrees(np.arctan2(R_mat[1, 0], R_mat[0, 0])))
            box_pts   = np.int0(corners[i].reshape(4, 2))
            self.get_logger().info(f'Baseplate detected via ArUco [{label}].')
            return X, Y, Z, angle_deg, box_pts

        return None

    def detect_baseplate(self, rgb: np.ndarray, depth: np.ndarray):
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(BASEPLATE_HSV_LOWER),
                               np.array(BASEPLATE_HSV_UPPER))
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k),
                                cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.get_logger().warn(
                'HSV baseplate: no contours found — HSV range may not match baseplate color',
                throttle_duration_sec=5.0)
            return None

        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        if area < BASEPLATE_MIN_AREA_PX:
            self.get_logger().warn(
                f'HSV baseplate: largest contour {area:.0f}px < {BASEPLATE_MIN_AREA_PX}px minimum',
                throttle_duration_sec=5.0)
            return None

        rect = cv2.minAreaRect(largest)
        (cx_px, cy_px), (w_px, h_px), angle_deg = rect
        if w_px < h_px:
            angle_deg += 90.0
        box_pts = np.int0(cv2.boxPoints(rect))

        ci, ri = int(cx_px), int(cy_px)
        ih, iw = depth.shape
        window = depth[max(0, ri-5):min(ih, ri+6), max(0, ci-5):min(iw, ci+6)]
        valid  = window[(window > 0) & np.isfinite(window)]
        if len(valid) == 0:
            return None
        Z       = float(np.median(valid))
        cam_mat = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float32)
        undist  = cv2.undistortPoints(
            np.array([[[cx_px, cy_px]]], dtype=np.float32), cam_mat, self.dist_coeffs, P=cam_mat)
        cx_u, cy_u = float(undist[0, 0, 0]), float(undist[0, 0, 1])
        X = (cx_u - self.cx) * Z / self.fx
        Y = (cy_u - self.cy) * Z / self.fy
        return X, Y, Z, angle_deg, box_pts

    def publish_baseplate_tf(self, X, Y, Z, angle_deg):
        pose_cam = Pose()
        pose_cam.position.x, pose_cam.position.y, pose_cam.position.z = X, Y, Z
        q = Rotation.from_euler('z', angle_deg, degrees=True).as_quat()
        pose_cam.orientation.x, pose_cam.orientation.y = q[0], q[1]
        pose_cam.orientation.z, pose_cam.orientation.w = q[2], q[3]

        for frame in ('camera_depth_optical_frame',
                      self.get_parameter('camera_frame').get_parameter_value().string_value):
            try:
                tf = self.tf_buffer.lookup_transform(
                    'base_link', frame,
                    rclpy.time.Time(), timeout=Duration(seconds=0.1))
                pose_base = tf2_geometry_msgs.do_transform_pose(pose_cam, tf)
                break
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                pass
        else:
            self.get_logger().warn(
                'publish_baseplate_tf: no TF from base_link to color frame — '
                'check camera_transform node is running',
                throttle_duration_sec=5.0)
            return

        quat = np.array([pose_base.orientation.x, pose_base.orientation.y,
                         pose_base.orientation.z, pose_base.orientation.w])
        txyz = np.array([pose_base.position.x, pose_base.position.y, pose_base.position.z])
        self._aruco_window.append((txyz, quat))
        if len(self._aruco_window) > ARUCO_WINDOW:
            self._aruco_window.pop(0)

        avg_t = np.mean([s[0] for s in self._aruco_window], axis=0)
        quats = np.array([s[1] for s in self._aruco_window])
        ref   = quats[0]
        quats[np.dot(quats, ref) < 0] *= -1
        avg_q = quats.mean(axis=0)
        avg_q /= np.linalg.norm(avg_q)

        t                         = TransformStamped()
        t.header.stamp            = self.get_clock().now().to_msg()
        t.header.frame_id         = 'base_link'
        t.child_frame_id          = 'baseplate_frame'
        t.transform.translation.x = float(avg_t[0])
        t.transform.translation.y = float(avg_t[1])
        t.transform.translation.z = float(avg_t[2])
        t.transform.rotation.x    = float(avg_q[0])
        t.transform.rotation.y    = float(avg_q[1])
        t.transform.rotation.z    = float(avg_q[2])
        t.transform.rotation.w    = float(avg_q[3])
        self._last_baseplate_tf = (avg_t, avg_q)

        if len(self._aruco_window) >= ARUCO_WINDOW and not self._baseplate_locked:
            self.static_tf_broadcaster.sendTransform(t)
            self._baseplate_locked = True
            self.get_logger().info('Baseplate TF locked as static transform.')
        else:
            self.tf_broadcaster.sendTransform(t)

    def publish_poses(self, bricks: list) -> list:
        pose_array                 = PoseArray()
        pose_array.header          = Header()
        pose_array.header.stamp    = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'base_link'

        meta_list = []
        for brick in bricks:
            pose_array.poses.append(brick['pose'])
            meta_list.append({
                'color':       brick['color'],
                'shape':       list(brick['shape']),
                'height_type': brick['height_type'],
                'height_m':    round(brick['height_m'] * 1000, 1),
            })

        self.pose_pub.publish(pose_array)
        meta_msg      = String()
        meta_msg.data = json.dumps(meta_list)
        self.meta_pub.publish(meta_msg)
        self.get_logger().info(
            f'Published {len(pose_array.poses)} bricks.',
            throttle_duration_sec=1.0)
        return bricks

    def publish_markers(self, bricks: list):
        now     = self.get_clock().now().to_msg()
        markers = MarkerArray()

        clear              = Marker()
        clear.action       = Marker.DELETEALL
        clear.header.frame_id = 'base_link'
        clear.header.stamp = now
        markers.markers.append(clear)

        bp_w = BASEPLATE_COLS * STUD_PITCH_M
        bp_d = BASEPLATE_ROWS * STUD_PITCH_M

        bp                 = Marker()
        bp.header.frame_id = 'baseplate_frame'
        bp.header.stamp    = now
        bp.ns, bp.id       = 'baseplate', 0
        bp.type            = Marker.CUBE
        bp.action          = Marker.ADD
        bp.pose.position.x = bp_w / 2.0
        bp.pose.position.y = bp_d / 2.0
        bp.pose.position.z = -0.005
        bp.pose.orientation.w = 1.0
        bp.scale.x, bp.scale.y, bp.scale.z = bp_w, bp_d, 0.008
        bp.color.r, bp.color.g, bp.color.b, bp.color.a = 0.10, 0.45, 0.15, 0.85
        markers.markers.append(bp)

        grid                 = Marker()
        grid.header.frame_id = 'baseplate_frame'
        grid.header.stamp    = now
        grid.ns, grid.id     = 'baseplate', 1
        grid.type            = Marker.LINE_LIST
        grid.action          = Marker.ADD
        grid.pose.orientation.w = 1.0
        grid.scale.x            = 0.0008
        grid.color.r, grid.color.g, grid.color.b, grid.color.a = 0.05, 0.25, 0.05, 0.70
        Z_GRID = 0.001
        for row in range(BASEPLATE_ROWS + 1):
            y  = row * STUD_PITCH_M
            p0 = Point(); p0.x = 0.0;  p0.y = y; p0.z = Z_GRID
            p1 = Point(); p1.x = bp_w; p1.y = y; p1.z = Z_GRID
            grid.points += [p0, p1]
        for col in range(BASEPLATE_COLS + 1):
            x  = col * STUD_PITCH_M
            p0 = Point(); p0.x = x; p0.y = 0.0;  p0.z = Z_GRID
            p1 = Point(); p1.x = x; p1.y = bp_d; p1.z = Z_GRID
            grid.points += [p0, p1]
        markers.markers.append(grid)

        for i, brick in enumerate(bricks):
            br, bc      = brick['shape']
            height_type = brick['height_type']
            height_m    = brick['height_m']
            pose_base   = brick['pose']
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

            txt                    = Marker()
            txt.header.frame_id    = 'base_link'
            txt.header.stamp       = now
            txt.ns, txt.id         = 'brick_labels', i
            txt.type               = Marker.TEXT_VIEW_FACING
            txt.action             = Marker.ADD
            txt.pose.position.x    = pose_base.position.x
            txt.pose.position.y    = pose_base.position.y
            txt.pose.position.z    = pose_base.position.z + brick_h + 0.025
            txt.pose.orientation.w = 1.0
            txt.scale.z            = 0.020
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = f'{brick["color"]}\n{br}x{bc} [{height_type}] {height_m*1000:.0f}mm'
            markers.markers.append(txt)

        self.marker_pub.publish(markers)


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
