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
from std_msgs.msg import Header, String
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

COLOR_RANGES = {
    'red':          [(( 42, 149, 130), (136, 195, 173))],
    'orange':       [(( 87, 146, 156), (202, 183, 183))],
    'yellow':       [((122, 113, 151), (207, 146, 191))],
    'light_green':  [(( 97,  85, 128), (161, 110, 166))],
    'blue':     [(( 88, 106,  80), (172, 126, 103))],
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
    'blue':    (0.30, 0.70, 1.00, 1.0),
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


STUD_PITCH_M   = 0.016
STUD_MIN_RADIUS_PX     = 5
STUD_MAX_RADIUS_PX     = 20
STUD_MIN_DIST_PX       = 15
MAX_BRICK_FOOTPRINT_M2 = (2 * STUD_PITCH_M) * (6 * STUD_PITCH_M) * 2.0

BASEPLATE_ROWS = 16
BASEPLATE_COLS = 16

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

PC_SUBSAMPLE          = 8       
RANSAC_INLIER_THRESH  = 0.008  
PLANE_NORMAL_MIN_Z    = 0.85    
HEIGHT_PERCENTILE     = 90      
MIN_CLUSTER_PTS       = 10      


TABLE_MASK_MARGIN_M   = 0.006   
MAX_BRICK_HEIGHT_M    = 0.060   

VOXEL_SIZE_M          = 0.004   
DBSCAN_EPS_M          = 0.012
DBSCAN_MIN_PTS        = 10      


COLOR_MATCH_MIN_FRAC  = 0.40

TF_TIMEOUT_SEC = 1.0


GROUND_PLANE_REFIT_SECS = 2.0



EMA_ALPHA         = 0.35
TRACK_MAX_STALE   = 3
TRACK_MATCH_DIST_M = 0.025   


class BrickDetectorNode(Node):

    def __init__(self):
        super().__init__('brick_detector')
        self.bridge = CvBridge()

        self.fx = self.fy = self.cx = self.cy = None
        self.latest_rgb   = None
        self.latest_depth = None

        self.latest_xyz   = None
        
        self.latest_pc_bgr = None

  
        self.ground_plane = None

        self._gp_last_fit: float = None

        self.baseplate_z_cam = None

        self._tracks: list = []

        self.create_subscription(
            Image,        '/camera/camera/color/image_raw',
            self.rgb_callback, 10)
        self.create_subscription(
            Image,        '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback, 10)
        self.create_subscription(
            CameraInfo,   '/camera/camera/color/camera_info',
            self.camera_info_callback, 10)
        self.create_subscription(
            PointCloud2,  '/camera/camera/depth/color/points',
            self.pointcloud_callback, 10)

        self.pose_pub   = self.create_publisher(PoseArray,   '/detected_bricks',      10)
        self.meta_pub   = self.create_publisher(String,      '/detected_bricks_meta', 10)
        self.debug_pub  = self.create_publisher(Image,       '/brick_debug_image',    10)
        self.marker_pub = self.create_publisher(MarkerArray, '/duplo_markers',        10)

        self.tf_buffer             = tf2_ros.Buffer()
        self.tf_listener           = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster        = tf2_ros.TransformBroadcaster(self)
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')

        self._frame_validated = False
        self.create_timer(0.2, self.process)
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
                f"Camera frame '{camera_frame}' not found in TF tree. "
                f"Common values: camera_color_optical_frame, camera_depth_optical_frame.",
                throttle_duration_sec=5.0
            )

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0];  self.fy = msg.k[4]
        self.cx = msg.k[2];  self.cy = msg.k[5]

    def rgb_callback(self, msg: Image):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg: Image):
        depth_mm = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_depth = depth_mm.astype(np.float32) / 1000.0

    def pointcloud_callback(self, msg: PointCloud2):
        """Unpack the RealSense organised XYZRGB point cloud."""
        self.latest_xyz, self.latest_pc_bgr = self._unpack_pointcloud(msg)


    def process(self):
        
        if self.latest_xyz is None or self.latest_pc_bgr is None:
            return
        if self.latest_rgb is None:
            return

        seg_bgr = self.latest_pc_bgr.copy()
        rgb     = self.latest_rgb.copy()
        depth   = self.latest_depth.copy() if self.latest_depth is not None else None
        debug   = rgb.copy()

        now_s = self.get_clock().now().nanoseconds * 1e-9
        if (self._gp_last_fit is None or
                now_s - self._gp_last_fit > GROUND_PLANE_REFIT_SECS):
            gp = self._fit_ground_plane(self.latest_xyz)
            if gp is not None:
                self.ground_plane = gp
                self._gp_last_fit = now_s

        if self.ground_plane is None:
            self.get_logger().warn('Waiting for ground plane fit...', once=True)
            return

        if depth is not None and self.fx is not None:
            bp = self.detect_baseplate(rgb, depth)
            if bp is not None:
                X, Y, Z, angle_deg, box_pts = bp
                self.baseplate_z_cam = Z
                self.publish_baseplate_tf(X, Y, Z, angle_deg)
                cv2.drawContours(debug, [box_pts], 0, (0, 200, 0), 2)
                cv2.putText(debug, 'baseplate', tuple(box_pts[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

        clusters   = self._cluster_above_table(self.latest_xyz, seg_bgr)
        all_bricks = []
        normal, d  = self.ground_plane

        for cluster_pts, cluster_bgr in clusters:
            color_name = self._classify_cluster_color(cluster_bgr)
            if color_name is None:
                continue

            rows, cols, angle_deg = self._shape_and_orientation_from_cluster(cluster_pts)

            centroid  = cluster_pts.mean(axis=0)
            heights   = -(cluster_pts @ normal + d)
            heights   = heights[heights > 0]
            height_m  = (float(np.percentile(heights, HEIGHT_PERCENTILE))
                         if len(heights) >= MIN_CLUSTER_PTS // 2
                         else BRICK_HEIGHTS['normal'])
            height_type = self._classify_height(height_m)

            pose_cam = Pose()
            pose_cam.position.x = float(centroid[0])
            pose_cam.position.y = float(centroid[1])
            pose_cam.position.z = float(centroid[2])
            q = Rotation.from_euler('z', angle_deg, degrees=True).as_quat()
            pose_cam.orientation.x, pose_cam.orientation.y = q[0], q[1]
            pose_cam.orientation.z, pose_cam.orientation.w = q[2], q[3]

            all_bricks.append({
                'color':       color_name,
                'shape':       (rows, cols),
                'height_type': height_type,
                'height_m':    height_m,
                'pose':        pose_cam,
            })
            self._draw_cluster_debug(debug, cluster_pts, color_name,
                                     (rows, cols), height_type, height_m)

        bricks_base = self.publish_poses(all_bricks)
        self.publish_markers(bricks_base)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))


    def _unpack_pointcloud(self, msg: PointCloud2):
        
        H, W     = msg.height, msg.width
        step     = msg.point_step
        n_floats = step // 4

        fields = {f.name: f.offset // 4 for f in msg.fields}

        data = np.frombuffer(msg.data, dtype=np.float32).reshape(H * W, n_floats)

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

    def _fit_ground_plane(self, xyz: np.ndarray):
        
        pts   = xyz.reshape(-1, 3)
        valid = pts[np.all(np.isfinite(pts), axis=1)]
        if len(valid) < 100:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid[::PC_SUBSAMPLE])

        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=RANSAC_INLIER_THRESH,
                ransac_n=3,
                num_iterations=500,
            )
        except Exception as e:
            self.get_logger().warn(f'Open3D plane fit failed: {e}')
            return None

        if len(inliers) < 50:
            return None

        a, b, c, d = plane_model
        norm_len = np.sqrt(a*a + b*b + c*c)
        if norm_len < 1e-6:
            return None
        normal = np.array([a, b, c]) / norm_len
        d      = d / norm_len

        if abs(normal[2]) < PLANE_NORMAL_MIN_Z:
            return None

        if normal[2] < 0:
            normal, d = -normal, -d

        self.get_logger().debug(
            f'Ground plane (Open3D): normal={normal.round(3)}, d={d:.4f}, '
            f'inliers={len(inliers)}'
        )
        return normal, d

    def _cluster_above_table(self, xyz: np.ndarray, bgr: np.ndarray) -> list:
       
        normal, d = self.ground_plane
        pts       = xyz.reshape(-1, 3)
        colors    = bgr.reshape(-1, 3)

        valid  = np.all(np.isfinite(pts), axis=1)
        h      = -(pts @ normal + d)
        keep   = valid & (h > TABLE_MASK_MARGIN_M) & (h < MAX_BRICK_HEIGHT_M)

        if np.sum(keep) < DBSCAN_MIN_PTS:
            return []

        above_pts = pts[keep]
        above_col = colors[keep]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(above_pts)
        pcd.colors = o3d.utility.Vector3dVector(above_col.astype(np.float64) / 255.0)

        pcd_down    = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE_M)
        pts_down    = np.asarray(pcd_down.points)
        colors_down = (np.asarray(pcd_down.colors) * 255).astype(np.uint8)

        labels = np.array(pcd_down.cluster_dbscan(
            eps=DBSCAN_EPS_M,
            min_points=DBSCAN_MIN_PTS,
            print_progress=False,
        ))

        clusters = []
        for label in np.unique(labels):
            if label < 0:
                continue
            mask = labels == label
            clusters.append((pts_down[mask], colors_down[mask]))

        result = []
        for pts_c, col_c in clusters:
            if self._cluster_footprint_m2(pts_c) > MAX_BRICK_FOOTPRINT_M2:
                self.get_logger().debug('Oversized cluster detected — attempting depth split.')
                result.extend(self._split_large_cluster(pts_c, col_c))
            else:
                result.append((pts_c, col_c))
        return result

    def _cluster_footprint_m2(self, pts: np.ndarray) -> float:
        if self.ground_plane is None or len(pts) < 3:
            return 0.0
        normal, _ = self.ground_plane
        ref = (np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        v1 = np.cross(normal, ref);  v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1);   v2 /= np.linalg.norm(v2)
        proj = np.column_stack([pts @ v1, pts @ v2]).astype(np.float32)
        hull = cv2.convexHull(proj.reshape(-1, 1, 2))
        return float(cv2.contourArea(hull))

    def _find_depth_seam(self, pts: np.ndarray):
        if self.ground_plane is None or len(pts) < MIN_CLUSTER_PTS * 2:
            return None

        normal, _ = self.ground_plane
        ref = (np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        v1 = np.cross(normal, ref);  v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1);   v2 /= np.linalg.norm(v2)

        proj_2d  = np.column_stack([pts @ v1, pts @ v2])
        centered = proj_2d - proj_2d.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        long_2d  = Vt[0]

        long_3d  = long_2d[0] * v1 + long_2d[1] * v2
        long_3d /= np.linalg.norm(long_3d)

        proj_1d = pts @ long_3d
        lo, hi  = proj_1d.min(), proj_1d.max()
        span    = hi - lo

        if span < STUD_PITCH_M * 3:
            return None

        n_bins = max(12, int(span / (STUD_PITCH_M * 0.4)))
        counts, bin_edges = np.histogram(proj_1d, bins=n_bins)
        bin_centers       = (bin_edges[:-1] + bin_edges[1:]) / 2
        smooth            = np.convolve(counts.astype(float), np.ones(3) / 3, mode='same')

        margin = n_bins // 3
        search = smooth[margin: n_bins - margin]
        if len(search) == 0:
            return None
        min_idx = margin + int(np.argmin(search))

        left_peak  = smooth[:min_idx].max() if min_idx > 0 else 0
        right_peak = smooth[min_idx + 1:].max() if min_idx < len(smooth) - 1 else 0
        avg_peak   = (left_peak + right_peak) / 2
        if avg_peak < 1 or smooth[min_idx] > avg_peak * 0.6:
            return None

        return long_3d, float(bin_centers[min_idx])

    def _split_large_cluster(self, pts: np.ndarray, colors: np.ndarray) -> list:
        seam = self._find_depth_seam(pts)
        if seam is not None:
            long_3d, split_val = seam
            side_a = (pts @ long_3d) <= split_val
            if side_a.sum() >= MIN_CLUSTER_PTS and (~side_a).sum() >= MIN_CLUSTER_PTS:
                self.get_logger().debug('Depth seam split successful.')
                return [(pts[side_a], colors[side_a]), (pts[~side_a], colors[~side_a])]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        labels = np.array(pcd.cluster_dbscan(
            eps=DBSCAN_EPS_M * 0.6,
            min_points=max(3, DBSCAN_MIN_PTS // 2),
            print_progress=False,
        ))
        sub = []
        for lbl in np.unique(labels):
            if lbl < 0:
                continue
            m = labels == lbl
            sub.append((pts[m], colors[m]))
        return sub if len(sub) > 1 else [(pts, colors)]

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

    def _shape_and_orientation_from_cluster(self, cluster_pts: np.ndarray) -> tuple:
        if self.ground_plane is None or len(cluster_pts) < MIN_CLUSTER_PTS:
            return 1, 1, 0.0

        if self.latest_rgb is not None and self.fx is not None:
            result = self._shape_from_studs(cluster_pts)
            if result is not None:
                stud_rows, stud_cols, angle_deg = result
                pc_rows, pc_cols, _ = self._shape_from_pointcloud_extent(cluster_pts)
                if pc_rows * pc_cols > stud_rows * stud_cols * 1.5:
                    return pc_rows, pc_cols, angle_deg
                return stud_rows, stud_cols, angle_deg

        return self._shape_from_pointcloud_extent(cluster_pts)

    def _shape_from_pointcloud_extent(self, cluster_pts: np.ndarray) -> tuple:
        normal, _ = self.ground_plane
        ref = (np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        v1 = np.cross(normal, ref);  v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1);   v2 /= np.linalg.norm(v2)

        proj     = np.column_stack([cluster_pts @ v1, cluster_pts @ v2])
        centered = proj - proj.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        proj_pca = centered @ Vt.T

        long_m  = float(np.percentile(proj_pca[:, 0], 95) - np.percentile(proj_pca[:, 0], 5))
        short_m = float(np.percentile(proj_pca[:, 1], 95) - np.percentile(proj_pca[:, 1], 5))

        cols = max(1, min(8, round(long_m  / STUD_PITCH_M)))
        rows = max(1, min(4, round(short_m / STUD_PITCH_M)))

        long_3d   = Vt[0, 0] * v1 + Vt[0, 1] * v2
        angle_deg = float(np.degrees(np.arctan2(long_3d[1], long_3d[0])))
        return rows, cols, angle_deg

    def _shape_from_studs(self, cluster_pts: np.ndarray):
        Z_vals = cluster_pts[:, 2]
        valid  = Z_vals > 0
        if not np.any(valid):
            return None
        pts = cluster_pts[valid]
        u = (pts[:, 0] / pts[:, 2] * self.fx + self.cx).astype(np.int32)
        v = (pts[:, 1] / pts[:, 2] * self.fy + self.cy).astype(np.int32)

        H, W = self.latest_rgb.shape[:2]
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_img):
            return None

        u_in, v_in = u[in_img], v[in_img]
        u_min, u_max = int(u_in.min()), int(u_in.max())
        v_min, v_max = int(v_in.min()), int(v_in.max())
        if (u_max - u_min) < 10 or (v_max - v_min) < 10:
            return None

        mask = np.zeros((H, W), dtype=np.uint8)
        mask[v_in, u_in] = 255
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, k)

        roi_bgr  = self.latest_rgb[v_min:v_max + 1, u_min:u_max + 1]
        roi_mask = mask[v_min:v_max + 1, u_min:u_max + 1]
        gray     = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray     = cv2.bitwise_and(gray, gray, mask=roi_mask)
        blurred  = cv2.GaussianBlur(gray, (5, 5), 1.5)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1, minDist=STUD_MIN_DIST_PX,
            param1=50, param2=18,
            minRadius=STUD_MIN_RADIUS_PX, maxRadius=STUD_MAX_RADIUS_PX,
        )
        if circles is None or len(circles[0]) < 2:
            return None

        centers = np.round(circles[0, :, :2]).astype(float)

        tree     = cKDTree(centers)
        dists, _ = tree.query(centers, k=min(2, len(centers)))
        if dists.ndim == 1 or dists.shape[1] < 2:
            return None
        pitch_px = float(np.median(dists[:, 1]))
        if pitch_px < 3:
            return None

        centered_c = centers - centers.mean(axis=0)
        _, _, Vt   = np.linalg.svd(centered_c, full_matrices=False)
        proj_pca   = centered_c @ Vt.T
        long_span  = proj_pca[:, 0].max() - proj_pca[:, 0].min()
        short_span = proj_pca[:, 1].max() - proj_pca[:, 1].min()

        cols = max(1, min(8, round(long_span  / pitch_px) + 1))
        rows = max(1, min(4, round(short_span / pitch_px) + 1))

        angle_deg = self._angle_from_pca_3d(cluster_pts)
        return rows, cols, angle_deg

    def _angle_from_pca_3d(self, cluster_pts: np.ndarray) -> float:
        if self.ground_plane is None:
            return 0.0
        normal, _ = self.ground_plane
        ref = (np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        v1 = np.cross(normal, ref);  v1 /= np.linalg.norm(v1)
        v2 = np.cross(normal, v1);   v2 /= np.linalg.norm(v2)
        proj     = np.column_stack([cluster_pts @ v1, cluster_pts @ v2])
        centered = proj - proj.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        long_3d  = Vt[0, 0] * v1 + Vt[0, 1] * v2
        return float(np.degrees(np.arctan2(long_3d[1], long_3d[0])))

    def _draw_cluster_debug(self, debug: np.ndarray, cluster_pts: np.ndarray,
                             color_name: str, shape: tuple,
                             height_type: str, height_m: float):
        if self.fx is None:
            return
        Z_v   = cluster_pts[:, 2]
        valid = Z_v > 0
        if not np.any(valid):
            return
        pts = cluster_pts[valid]
        Z   = pts[:, 2]
        u   = (pts[:, 0] / Z * self.fx + self.cx).astype(np.int32)
        v   = (pts[:, 1] / Z * self.fy + self.cy).astype(np.int32)
        H, W    = debug.shape[:2]
        in_img  = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_img):
            return
        px   = np.column_stack([u[in_img], v[in_img]])
        hull = cv2.convexHull(px.reshape(-1, 1, 2))
        cv2.drawContours(debug, [hull], 0, (0, 255, 0), 2)
        cx_px, cy_px = int(px[:, 0].mean()), int(px[:, 1].mean())
        rows, cols   = shape
        label = f'{color_name} {rows}x{cols} [{height_type}] {height_m*1000:.0f}mm'
        cv2.putText(debug, label, (cx_px - 30, cy_px - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.circle(debug, (cx_px, cy_px), 4, (0, 255, 255), -1)

    def _classify_height(self, height_m: float) -> str:
        for label, (lo, hi) in HEIGHT_THRESHOLDS.items():
            if lo <= height_m < hi:
                return label
        return 'normal'

   
    def detect_baseplate(self, rgb: np.ndarray, depth: np.ndarray):
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

    def _update_tracks(self, bricks_base: list):
        for track in self._tracks:
            track['stale'] += 1

        for brick in bricks_base:
            px        = brick['pose_base'].position
            best_idx  = None
            best_dist = TRACK_MATCH_DIST_M

            for i, track in enumerate(self._tracks):
                if track['color'] != brick['color']:
                    continue
                tp   = track['pose_base'].position
                dist = np.hypot(px.x - tp.x, px.y - tp.y)
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = i

            if best_idx is not None:
                track = self._tracks[best_idx]
                tp    = track['pose_base'].position
                a     = EMA_ALPHA
                tp.x  = a * px.x + (1 - a) * tp.x
                tp.y  = a * px.y + (1 - a) * tp.y
                tp.z  = a * px.z + (1 - a) * tp.z
                track['pose_base'].orientation = brick['pose_base'].orientation
                track['shape']       = brick['shape']
                track['height_type'] = brick['height_type']
                track['height_m']    = brick['height_m']
                track['stale']       = 0
            else:
                self._tracks.append({
                    'color':       brick['color'],
                    'shape':       brick['shape'],
                    'height_type': brick['height_type'],
                    'height_m':    brick['height_m'],
                    'pose_base':   brick['pose_base'],
                    'stale':       0,
                })

        self._tracks = [t for t in self._tracks if t['stale'] <= TRACK_MAX_STALE]

    def _snap_to_grid(self, pose_base: Pose, shape: tuple) -> Pose:
        try:
            tf_to_bp = self.tf_buffer.lookup_transform(
                'baseplate_frame', 'base_link',
                rclpy.time.Time(), timeout=Duration(seconds=0.1))
            pose_bp = tf2_geometry_msgs.do_transform_pose(pose_base, tf_to_bp)

            rows, cols = shape
            x_corner = round(pose_bp.position.x / STUD_PITCH_M - cols / 2.0)
            y_corner = round(pose_bp.position.y / STUD_PITCH_M - rows / 2.0)
            pose_bp.position.x = (x_corner + cols / 2.0) * STUD_PITCH_M
            pose_bp.position.y = (y_corner + rows / 2.0) * STUD_PITCH_M

            tf_to_base = self.tf_buffer.lookup_transform(
                'base_link', 'baseplate_frame',
                rclpy.time.Time(), timeout=Duration(seconds=0.1))
            return tf2_geometry_msgs.do_transform_pose(pose_bp, tf_to_base)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            self.get_logger().warn(
                'Grid snapping unavailable — baseplate_frame not found. Publishing unsnapped pose.',
                throttle_duration_sec=5.0
            )
            return pose_base

    def publish_poses(self, bricks: list) -> list:
       
        camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value

        bricks_base = []
        for brick in bricks:
            try:
                tf        = self.tf_buffer.lookup_transform(
                    'base_link', camera_frame,
                    rclpy.time.Time(), timeout=Duration(seconds=TF_TIMEOUT_SEC))
                pose_base = tf2_geometry_msgs.do_transform_pose(brick['pose'], tf)
                bricks_base.append({**brick, 'pose_base': pose_base})
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f'TF lookup failed: {e}')

        self._update_tracks(bricks_base)

        pose_array             = PoseArray()
        pose_array.header      = Header()
        pose_array.header.stamp    = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'base_link'

        meta_list  = []
        bricks_out = []

        for track in self._tracks:
            pose_out = self._snap_to_grid(track['pose_base'], track['shape'])
            pose_array.poses.append(pose_out)
            meta_list.append({
                'color':       track['color'],
                'shape':       list(track['shape']),
                'height_type': track['height_type'],
                'height_m':    round(track['height_m'] * 1000, 1),
            })
            bricks_out.append({**track, 'pose_base': pose_out})

        self.pose_pub.publish(pose_array)
        meta_msg      = String()
        meta_msg.data = json.dumps(meta_list)
        self.meta_pub.publish(meta_msg)
        plane_status = 'fitted' if self.ground_plane else 'fallback depth'
        self.get_logger().info(
            f'Published {len(pose_array.poses)} bricks. Ground plane: {plane_status}'
        )
        return bricks_out

    def publish_markers(self, bricks_base: list):
        now     = self.get_clock().now().to_msg()
        markers = MarkerArray()

        clear              = Marker()
        clear.action       = Marker.DELETEALL
        clear.header.frame_id = 'base_link'
        clear.header.stamp = now
        markers.markers.append(clear)

        bp_w = BASEPLATE_COLS * STUD_PITCH_M
        bp_d = BASEPLATE_ROWS * STUD_PITCH_M

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
