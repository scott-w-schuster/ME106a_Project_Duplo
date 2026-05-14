# Code Changes — Session 2026-05-14

## `src/planning/launch/duplo.launch.py`

**Camera profile fix**
- Changed `rgb_camera.color_profile` from `'1920x1080x30'` to `'640x480x15'`
- Reason: D435 on USB 2.1 does not support 1920×1080@30fps; the invalid profile caused an ERROR log on every launch and fell back silently.

---

## `src/planning/planning/planning_node.py`

**Shutdown RuntimeError fix**
- Wrapped `rclpy.shutdown()` in `_execute_step()` (called from a daemon thread at build-complete) with `try/except Exception`.
- Updated `main()` to use `try/except/finally` so `node.destroy_node()` and `rclpy.shutdown()` are both guarded.
- Reason: calling `rclpy.shutdown()` from a daemon thread while `rclpy.spin()` runs on the main thread caused `RuntimeError: Context must be initialized before it can be shutdown` when `main()` tried to shut down a second time.

---

## `src/perception/perception/brick_detector.py`

### Pre-lock frame buffering (new feature)
Brick detection previously required `_baseplate_locked = True`, which only becomes true after the ArUco marker is seen. Because ArUco is only visible at scan pose 6 (of 8), poses 1–5 produced no detections at all.

**Changes:**
- Added `self._pre_lock_buffer: list = []` and `self._flush_pending = False` to `__init__`.
- Added `_cam_to_base_link()` method — same structure as `_cam_to_baseplate()` but targets `base_link` (always available).
- Modified the `if not self._baseplate_locked` block in `process()`: instead of warning and returning, buffers each frame as `(xyz_cam, bgr_cam, cam_to_base_link_4×4)`, capped at 16 frames.
- In `publish_baseplate_tf()`, set `self._flush_pending = True` immediately after `self._baseplate_locked = True`.
- Added `_flush_pre_lock_buffer()`: when called, looks up the now-available static `base_link→baseplate_frame` TF, computes `cam_to_bp = base_to_bp @ cam_to_base` for each buffered frame, runs `_cluster_above_table()` and `_clusters_to_bricks()` on each, and publishes any bricks found.
- Added flush trigger in `process()`: checks `self._flush_pending` after normal detection and calls `_flush_pre_lock_buffer()`.

### Cluster-to-brick extraction
- Extracted the inline brick-classification loop from `process()` into a new `_clusters_to_bricks(clusters) -> list` method, so both `process()` and `_flush_pre_lock_buffer()` share the same logic.

### Z + XY boundary filter (replaces previous color pre-filter attempts)
The prior sessions added a LAB color pre-filter to prevent the entire table surface from merging into one DBSCAN cluster. This session went through two iterations:

1. **Disabled** the color filter to diagnose clustering — confirmed 1 giant cluster of ~50,000 pts with "color unclassified".
2. **Green exclusion** (`LAB A < 118`) — only removed ~1% of points because the Z filter had already excluded most of the flat baseplate surface; remaining points were non-green clutter (robot arm, table edges, walls) that still merged into one cluster.
3. **XY boundary filter** (current) — after the Z filter, keeps only points within the baseplate footprint (`BASEPLATE_COLS × STUD_PITCH_M` × `BASEPLATE_ROWS × STUD_PITCH_M`) plus 20 mm padding. This removes all out-of-bounds clutter in a single step without needing color calibration.

The XY filter was added directly into the `keep` mask in `_cluster_above_table()`.
