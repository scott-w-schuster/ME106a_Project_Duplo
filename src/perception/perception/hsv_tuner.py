import copy

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


COLOR_RANGES = {
    'red':         [((40,  155, 138), (200, 200, 188))],
    'orange':      [((100, 133, 155), (230, 175, 210))],
    'yellow':      [((140, 112, 170), (255, 148, 228))],
    'light_green': [((75,   72, 158), (220, 120, 225))],
    'blue':    [((40,  102,  52), (195, 145, 108))],
    'mint':        [((128,  88,  88), (228, 128, 135))],
    'white':       [((185, 115, 115), (255, 145, 145))],
    'purple':      [((38,  125,  72), (175, 168, 118))],
    'brown':       [((28,  122, 135), (132, 165, 182))],
    'pink':        [((118, 145, 115), (238, 202, 158))],
    'light_blue':  [((132, 105,  85), (255, 140, 130))],
}

COLOR_KEYS = list(COLOR_RANGES.keys())

WINDOW   = 'LAB Tuner'
DISPLAY_W = 320
DISPLAY_H = 240


class LABTunerNode(Node):

    def __init__(self):
        super().__init__('lab_tuner')
        self.bridge        = CvBridge()
        self.latest_frame  = None
        self.color_idx     = 0
        self.ranges        = copy.deepcopy(COLOR_RANGES)
        self._panel_count  = 3

        self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self._image_cb, 10)

        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, DISPLAY_W * 3, DISPLAY_H)
        self._build_trackbars()

        self.create_timer(0.03, self._update)
        self.get_logger().info(
            'LAB Tuner ready.  Press 1-0 to switch colour, s to save, '
            'p to print, q to quit.'
        )

    def _image_cb(self, msg: Image):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    
    def _build_trackbars(self):
        try:
            w = cv2.getWindowImageRect(WINDOW)[2]
            h = cv2.getWindowImageRect(WINDOW)[3]
        except Exception:
            w, h = DISPLAY_W * self._panel_count, DISPLAY_H
        cv2.destroyWindow(WINDOW)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, w if w > 0 else DISPLAY_W * self._panel_count, h if h > 0 else DISPLAY_H)

        color_name = COLOR_KEYS[self.color_idx]
        lo, hi     = self.ranges[color_name][0]

        cv2.createTrackbar('L low',  WINDOW, lo[0], 255, lambda v: None)
        cv2.createTrackbar('L high', WINDOW, hi[0], 255, lambda v: None)
        cv2.createTrackbar('a low',  WINDOW, lo[1], 255, lambda v: None)
        cv2.createTrackbar('a high', WINDOW, hi[1], 255, lambda v: None)
        cv2.createTrackbar('b low',  WINDOW, lo[2], 255, lambda v: None)
        cv2.createTrackbar('b high', WINDOW, hi[2], 255, lambda v: None)

        cv2.setWindowTitle(
            WINDOW,
            f'LAB Tuner  —  [{self.color_idx + 1}] {color_name}  '
            f'(1-0 switch | s save | p print | q quit)'
        )

    def _read_trackbars(self) -> list:
        def tb(name):
            return cv2.getTrackbarPos(name, WINDOW)
        lo = (tb('L low'),  tb('a low'),  tb('b low'))
        hi = (tb('L high'), tb('a high'), tb('b high'))
        return [(lo, hi)]

    def _update(self):
        if self.latest_frame is None:
            return

        frame = self.latest_frame.copy()
        lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        ranges = self._read_trackbars()
        mask   = np.zeros(lab.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(lab, np.array(lo), np.array(hi)))

        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        orig    = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        mask_3c = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                             (DISPLAY_W, DISPLAY_H))
        masked  = cv2.resize(cv2.bitwise_and(frame, frame, mask=mask),
                             (DISPLAY_W, DISPLAY_H))

        def label(img, text):
            cv2.putText(img, text, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        label(orig,    'Original')
        label(mask_3c, 'Mask')
        label(masked,  'Detected')

        px = int(np.sum(mask > 0))
        cv2.putText(masked, f'{px} px', (6, DISPLAY_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        color_name = COLOR_KEYS[self.color_idx]
        if color_name == 'red':
            excl = np.zeros(lab.shape[:2], dtype=np.uint8)
            for neighbor in ('orange', 'pink'):
                for lo, hi in self.ranges[neighbor]:
                    excl = cv2.bitwise_or(excl, cv2.inRange(lab, np.array(lo), np.array(hi)))
            adj_mask   = cv2.bitwise_and(mask, cv2.bitwise_not(excl))
            adj_masked = cv2.resize(cv2.bitwise_and(frame, frame, mask=adj_mask),
                                    (DISPLAY_W, DISPLAY_H))
            label(adj_masked, 'Red (excl. orange+pink)')
            adj_px = int(np.sum(adj_mask > 0))
            cv2.putText(adj_masked, f'{adj_px} px', (6, DISPLAY_H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if self._panel_count != 4:
                cv2.resizeWindow(WINDOW, DISPLAY_W * 4, DISPLAY_H)
                self._panel_count = 4
            cv2.imshow(WINDOW, np.hstack([orig, mask_3c, masked, adj_masked]))
        else:
            if self._panel_count != 3:
                cv2.resizeWindow(WINDOW, DISPLAY_W * 3, DISPLAY_H)
                self._panel_count = 3
            cv2.imshow(WINDOW, np.hstack([orig, mask_3c, masked]))

        key = cv2.waitKey(1) & 0xFF

        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(COLOR_KEYS):
                self.color_idx = idx
                self._build_trackbars()
        elif key == ord('0') and len(COLOR_KEYS) >= 10:
            self.color_idx = 9
            self._build_trackbars()
        elif key == ord('s'):
            color_name = COLOR_KEYS[self.color_idx]
            self.ranges[color_name] = self._read_trackbars()
            self.get_logger().info(f"Saved '{color_name}': {self.ranges[color_name]}")
        elif key == ord('p'):
            self._print_ranges()
        elif key == ord('q'):
            self._print_ranges()
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def _print_ranges(self):
        print('\n' + '=' * 70)
        print('# ── Updated COLOR_RANGES (LAB) — paste into brick_detector.py ──')
        print('COLOR_RANGES = {')
        for name, ranges in self.ranges.items():
            lo, hi = ranges[0]
            print(f"    '{name}':{' ' * max(1, 13 - len(name))}"
                  f"[(({lo[0]:3d}, {lo[1]:3d}, {lo[2]:3d}), "
                  f"({hi[0]:3d}, {hi[1]:3d}, {hi[2]:3d}))],")
        print('}')
        print('=' * 70 + '\n')


def main(args=None):
    rclpy.init(args=args)
    node = LABTunerNode()
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
