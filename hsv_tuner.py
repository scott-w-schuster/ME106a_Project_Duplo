"""
hsv_tuner.py

Interactive HSV colour-range tuner for Duplo brick detection.

Opens an OpenCV window showing three panels side by side:
  Left   — original camera feed
  Centre — binary mask (white = colour detected, black = not)
  Right  — original with mask applied (shows what the detector actually sees)

Trackbars let you adjust H/S/V min-max live.  Red gets a second H range
because it wraps around 0°/180° in OpenCV HSV.

Controls:
  1 – 8  — switch to a different colour (shown in window title)
  s      — save current trackbar values for the active colour
  p      — print the full updated COLOR_RANGES dict to the terminal
             (copy-paste straight into brick_detector.py)
  q      — quit

Usage:
  ros2 run perception hsv_tuner

Assumes /camera/camera/color/image_raw is publishing.
No depth or point cloud needed.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Starting HSV ranges — copied from brick_detector.py.
# Edit these to match your current brick_detector.py values before launching
# so the trackbars initialise at a sensible position.
# Format: each entry is a list of (lower, upper) tuples.
# ---------------------------------------------------------------------------
COLOR_RANGES = {
    'red':         [((0,   160,  80), (8,   255, 210)),
                    ((172, 160,  80), (179, 255, 210))],
    'brown':       [((8,  80,  40), (20, 180, 130))],
    'pink':       [((145, 150,  100), (170, 255, 230))],
    'light_blue':       [((90,  40,  185), (115, 130, 255))],
    'orange':      [((5,   200, 150), (15,  255, 235))],
    'yellow':      [((20,  150, 150), (35,  255, 240))],
    'light_green': [((35,  120,  80), (55,  255, 210))],
    'sky_blue':    [((95,  150, 140), (115, 255, 235))],
    'mint':        [((80,   50, 170), (95,  140, 235))],
    'white':       [((0,     0, 200), (179,  50, 255))],
    'purple':      [((130,  60,  90), (155, 180, 185))],

}

COLOR_KEYS = list(COLOR_RANGES.keys())   # fixed order for 1-8 key mapping

WINDOW = 'HSV Tuner'
DISPLAY_W = 320   # width of each panel (3 panels side by side)
DISPLAY_H = 240


class HSVTunerNode(Node):

    def __init__(self):
        super().__init__('hsv_tuner')
        self.bridge      = CvBridge()
        self.latest_frame = None

        # Current colour being tuned (index into COLOR_KEYS)
        self.color_idx = 0

        # Working copy of ranges — updated by 's', printed by 'p'
        import copy
        self.ranges = copy.deepcopy(COLOR_RANGES)

        self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self._image_cb, 10)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, DISPLAY_W * 3, DISPLAY_H)
        self._build_trackbars()

        # 30 ms timer — faster than 5 Hz so the UI feels responsive
        self.create_timer(0.03, self._update)
        self.get_logger().info(
            'HSV Tuner ready.  Press 1-8 to switch colour, s to save, '
            'p to print, q to quit.'
        )

    # -----------------------------------------------------------------------

    def _image_cb(self, msg: Image):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # -----------------------------------------------------------------------
    # Trackbar setup
    # -----------------------------------------------------------------------

    def _build_trackbars(self):
        """Create all trackbars for the current colour, initialised to its range."""
        # Destroy any previous trackbars by recreating the window
        cv2.destroyWindow(WINDOW)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, DISPLAY_W * 3, DISPLAY_H)

        color_name = COLOR_KEYS[self.color_idx]
        r = self.ranges[color_name]

        # Primary range (all colours)
        lo1, hi1 = r[0]
        cv2.createTrackbar('H1 low',  WINDOW, lo1[0], 179, lambda v: None)
        cv2.createTrackbar('H1 high', WINDOW, hi1[0], 179, lambda v: None)
        cv2.createTrackbar('S1 low',  WINDOW, lo1[1], 255, lambda v: None)
        cv2.createTrackbar('S1 high', WINDOW, hi1[1], 255, lambda v: None)
        cv2.createTrackbar('V1 low',  WINDOW, lo1[2], 255, lambda v: None)
        cv2.createTrackbar('V1 high', WINDOW, hi1[2], 255, lambda v: None)

        # Second range — only meaningful for red (H wrap-around).
        # Always shown so the UI layout is consistent; ignored for non-red.
        if len(r) > 1:
            lo2, hi2 = r[1]
        else:
            lo2, hi2 = (0, 0, 0), (0, 0, 0)
        cv2.createTrackbar('H2 low  (red only)', WINDOW, lo2[0], 179, lambda v: None)
        cv2.createTrackbar('H2 high (red only)', WINDOW, hi2[0], 179, lambda v: None)

        cv2.setWindowTitle(
            WINDOW,
            f'HSV Tuner  —  [{self.color_idx + 1}] {color_name}  '
            f'(1-8 switch | s save | p print | q quit)'
        )

    def _read_trackbars(self) -> list:
        """Read current trackbar positions and return a ranges list."""
        def tb(name):
            return cv2.getTrackbarPos(name, WINDOW)

        lo1 = (tb('H1 low'),  tb('S1 low'),  tb('V1 low'))
        hi1 = (tb('H1 high'), tb('S1 high'), tb('V1 high'))
        h2_lo = tb('H2 low  (red only)')
        h2_hi = tb('H2 high (red only)')

        color_name = COLOR_KEYS[self.color_idx]
        if color_name == 'red' and (h2_lo > 0 or h2_hi > 0):
            lo2 = (h2_lo, lo1[1], lo1[2])
            hi2 = (h2_hi, hi1[1], hi1[2])
            return [(lo1, hi1), (lo2, hi2)]
        return [(lo1, hi1)]

    # -----------------------------------------------------------------------
    # Main update loop
    # -----------------------------------------------------------------------

    def _update(self):
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Build mask from current trackbar values
        ranges = self._read_trackbars()
        mask   = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))

        # Optional morphological clean-up (matches brick_detector pipeline)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # Three display panels
        orig    = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        mask_3c = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                             (DISPLAY_W, DISPLAY_H))
        masked  = cv2.resize(cv2.bitwise_and(frame, frame, mask=mask),
                             (DISPLAY_W, DISPLAY_H))

        # Overlay text on panels
        def label(img, text):
            cv2.putText(img, text, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        label(orig,    'Original')
        label(mask_3c, 'Mask')
        label(masked,  'Detected')

        # Show pixel count so you know if the mask is picking up anything
        px = int(np.sum(mask > 0))
        cv2.putText(masked, f'{px} px', (6, DISPLAY_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        canvas = np.hstack([orig, mask_3c, masked])
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(1) & 0xFF

        # 1-8: switch colour
        if ord('1') <= key <= ord('8'):
            idx = key - ord('1')
            if idx < len(COLOR_KEYS):
                self.color_idx = idx
                self._build_trackbars()

        # s: save current values for this colour
        elif key == ord('s'):
            color_name = COLOR_KEYS[self.color_idx]
            self.ranges[color_name] = self._read_trackbars()
            self.get_logger().info(
                f"Saved '{color_name}': {self.ranges[color_name]}"
            )

        # p: print full dict
        elif key == ord('p'):
            self._print_ranges()

        # q: quit
        elif key == ord('q'):
            self._print_ranges()
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def _print_ranges(self):
        """Print the updated COLOR_RANGES dict, ready to paste into brick_detector.py."""
        print('\n' + '=' * 70)
        print('# ── Updated COLOR_RANGES — paste into brick_detector.py ──')
        print('COLOR_RANGES = {')
        for name, ranges in self.ranges.items():
            if len(ranges) == 1:
                lo, hi = ranges[0]
                print(f"    '{name}':{' ' * max(1, 13 - len(name))}"
                      f"[(({lo[0]:3d}, {lo[1]:3d}, {lo[2]:3d}), "
                      f"({hi[0]:3d}, {hi[1]:3d}, {hi[2]:3d}))],")
            else:
                lo1, hi1 = ranges[0]
                lo2, hi2 = ranges[1]
                print(f"    '{name}':{' ' * max(1, 13 - len(name))}"
                      f"[(({lo1[0]:3d}, {lo1[1]:3d}, {lo1[2]:3d}), "
                      f"({hi1[0]:3d}, {hi1[1]:3d}, {hi1[2]:3d})),")
                print(f"    {' ' * (13 + len(name) + 2)}"
                      f" (({lo2[0]:3d}, {lo2[1]:3d}, {lo2[2]:3d}), "
                      f"({hi2[0]:3d}, {hi2[1]:3d}, {hi2[2]:3d}))],")
        print('}')
        print('=' * 70 + '\n')


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = HSVTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
