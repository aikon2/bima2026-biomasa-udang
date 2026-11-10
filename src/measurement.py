"""Per-detection shrimp length measurement.

Steps: clip the box to the frame, reject empty/tiny/edge-truncated ROIs, segment
with Otsu and automatic polarity, then measure either the curved body length
along the skeleton (default) or the legacy min-area-rect long side.
"""
import logging
from collections import deque, namedtuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

MeasureResult = namedtuple("MeasureResult", "length_cm status")

STATUS_OK = "ok"
STATUS_EDGE = "edge"
STATUS_EMPTY_ROI = "empty_roi"
STATUS_NO_CONTOUR = "no_contour"

METHOD_SKELETON = "skeleton"
METHOD_MIN_AREA_RECT = "min_area_rect"
DEFAULT_METHOD = METHOD_SKELETON
DEFAULT_EDGE_MARGIN_PX = 4
DEFAULT_MIN_ROI_PX = 8
DEFAULT_MORPH_KERNEL = 3

FOREGROUND = 255
BORDER_FOREGROUND_RATIO = 0.5
MAX_THINNING_ITERATIONS = 500
PATH_SAMPLE_STEP_PX = 3
NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _clip_box(detection, frame_shape):
    """Clip to frame; returns inclusive (x1, y1, x2, y2) or None if empty."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x1, y1 = max(int(detection.x1), 0), max(int(detection.y1), 0)
    x2, y2 = min(int(detection.x2), width - 1), min(int(detection.y2), height - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _touches_edge(box, frame_shape, margin_px):
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = box
    return (x1 < margin_px or y1 < margin_px
            or x2 > width - 1 - margin_px or y2 > height - 1 - margin_px)


def _border_pixels(mask):
    return np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])


def segment_roi(roi_bgr, morph_kernel):
    """Binary mask (255 = shrimp) with Otsu and border-based polarity selection.

    The object is the Otsu class that is NOT dominant on the ROI border, so both
    bright-on-dark and dark-on-bright scenes work.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, FOREGROUND, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border_ratio = np.mean(_border_pixels(mask) == FOREGROUND)
    if border_ratio > BORDER_FOREGROUND_RATIO:
        mask = cv2.bitwise_not(mask)
    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _largest_contour(mask):
    # [-2] works for both OpenCV 3 (3 return values) and OpenCV 4 (2 values).
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _rect_endpoints(contour):
    """Midpoints of the two short sides of the min-area rectangle (ROI coords)."""
    corners = cv2.boxPoints(cv2.minAreaRect(contour))
    side_a = np.linalg.norm(corners[1] - corners[0])
    side_b = np.linalg.norm(corners[2] - corners[1])
    if side_a >= side_b:
        return np.array([(corners[1] + corners[2]) / 2.0, (corners[3] + corners[0]) / 2.0])
    return np.array([(corners[0] + corners[1]) / 2.0, (corners[2] + corners[3]) / 2.0])


def _zhang_suen_pass(image, first_pass):
    """One Zhang-Suen sub-iteration on a zero-padded 0/1 image; returns #removed."""
    center = image[1:-1, 1:-1]
    p2, p3, p4 = image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:]
    p5, p6, p7 = image[2:, 2:], image[2:, 1:-1], image[2:, :-2]
    p8, p9 = image[1:-1, :-2], image[:-2, :-2]
    ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
    neighbors = sum(ring[:-1])
    transitions = sum(((ring[i] == 0) & (ring[i + 1] == 1)).astype(np.uint8) for i in range(8))
    if first_pass:
        cond = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
    else:
        cond = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
    remove = (center == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & cond
    center[remove] = 0
    return int(np.count_nonzero(remove))


def thin_mask(mask):
    """Zhang-Suen thinning: 1-px wide, 8-connected skeleton (bool array).

    Used instead of the erode/subtract morphological skeleton because the
    latter is often disconnected, which breaks longest-path extraction.
    """
    image = np.pad((mask > 0).astype(np.uint8), 1, mode="constant")
    for _ in range(MAX_THINNING_ITERATIONS):
        removed = _zhang_suen_pass(image, True) + _zhang_suen_pass(image, False)
        if removed == 0:
            break
    return image[1:-1, 1:-1].astype(bool)


def _bfs(start, pixels):
    """BFS over 8-connected pixel set; returns (farthest pixel, parent map)."""
    parents = {start: None}
    queue = deque([start])
    farthest = start
    while queue:
        farthest = queue.popleft()
        row, col = farthest
        for d_row, d_col in NEIGHBOR_OFFSETS:
            neighbor = (row + d_row, col + d_col)
            if neighbor in pixels and neighbor not in parents:
                parents[neighbor] = farthest
                queue.append(neighbor)
    return farthest, parents


def longest_skeleton_path(skeleton):
    """Longest path (two BFS passes) through skeleton pixels as Nx2 (x, y) points."""
    rows, cols = np.nonzero(skeleton)
    if rows.size < 2:
        return np.zeros((0, 2))
    pixels = set(zip(rows.tolist(), cols.tolist()))
    end_a, _ = _bfs((int(rows[0]), int(cols[0])), pixels)
    end_b, parents = _bfs(end_a, pixels)
    path = []
    node = end_b
    while node is not None:
        path.append((node[1], node[0]))
        node = parents[node]
    sampled = path[::PATH_SAMPLE_STEP_PX]
    if sampled[-1] != path[-1]:
        sampled.append(path[-1])
    return np.asarray(sampled, dtype=np.float64)


def _object_points(mask, contour, method):
    """Points (ROI coords) whose polyline length is the shrimp length."""
    if method == METHOD_MIN_AREA_RECT:
        return _rect_endpoints(contour)
    object_mask = np.zeros_like(mask)
    cv2.drawContours(object_mask, [contour], -1, FOREGROUND, thickness=cv2.FILLED)
    path = longest_skeleton_path(thin_mask(object_mask))
    if path.shape[0] < 2:
        LOGGER.debug("Skeleton too short; falling back to min-area-rect")
        return _rect_endpoints(contour)
    return path


def measure_length(frame_bgr, detection, plane_scale, measurement_cfg):
    """Measure one detection; never raises for bad boxes (returns a status)."""
    method = measurement_cfg.get("method", DEFAULT_METHOD)
    if method not in (METHOD_SKELETON, METHOD_MIN_AREA_RECT):
        raise ValueError("Unknown measurement method: {}".format(method))
    box = _clip_box(detection, frame_bgr.shape)
    min_roi_px = measurement_cfg.get("min_roi_px", DEFAULT_MIN_ROI_PX)
    if box is None or min(box[2] - box[0], box[3] - box[1]) + 1 < min_roi_px:
        return MeasureResult(None, STATUS_EMPTY_ROI)
    if _touches_edge(box, frame_bgr.shape,
                     measurement_cfg.get("edge_margin_px", DEFAULT_EDGE_MARGIN_PX)):
        return MeasureResult(None, STATUS_EDGE)

    x1, y1, x2, y2 = box
    roi = frame_bgr[y1:y2 + 1, x1:x2 + 1]
    mask = segment_roi(roi, measurement_cfg.get("morph_kernel", DEFAULT_MORPH_KERNEL))
    contour = _largest_contour(mask)
    if contour is None:
        return MeasureResult(None, STATUS_NO_CONTOUR)
    points_frame = _object_points(mask, contour, method) + np.array([x1, y1], dtype=np.float64)
    return MeasureResult(plane_scale.polyline_length_cm(points_frame), STATUS_OK)
