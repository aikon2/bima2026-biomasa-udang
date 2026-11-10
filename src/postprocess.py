"""YOLOv8 output decoding, NMS and box rescaling (pure numpy)."""
from collections import namedtuple
from typing import List, Sequence, Tuple

import numpy as np

Detection = namedtuple("Detection", "x1 y1 x2 y2 conf")

BOX_COLUMNS = 4


def decode_yolov8(output: np.ndarray, num_classes: int,
                  conf_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Decode ultralytics YOLOv8 output (1, 4+nc, N) into xyxy boxes and scores.

    Boxes are cx, cy, w, h in network-input pixels; class score is the max over
    the class columns (YOLOv8 has no objectness term).
    """
    expected_rows = BOX_COLUMNS + num_classes
    predictions = np.asarray(output, dtype=np.float32).reshape(expected_rows, -1).T
    class_scores = predictions[:, BOX_COLUMNS:BOX_COLUMNS + num_classes]
    scores = class_scores.max(axis=1)
    mask = scores > conf_threshold
    if not np.any(mask):
        return np.zeros((0, BOX_COLUMNS), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    cxcywh = predictions[mask, :BOX_COLUMNS]
    half_w = cxcywh[:, 2] / 2.0
    half_h = cxcywh[:, 3] / 2.0
    boxes = np.stack([cxcywh[:, 0] - half_w, cxcywh[:, 1] - half_h,
                      cxcywh[:, 0] + half_w, cxcywh[:, 1] + half_h], axis=1)
    return boxes.astype(np.float32), scores[mask].astype(np.float32)


def _iou_one_to_many(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    inter_x1 = np.maximum(box[0], others[:, 0])
    inter_y1 = np.maximum(box[1], others[:, 1])
    inter_x2 = np.minimum(box[2], others[:, 2])
    inter_y2 = np.minimum(box[3], others[:, 3])
    inter_area = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    other_area = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = box_area + other_area - inter_area
    return np.where(union > 0, inter_area / np.maximum(union, 1e-12), 0.0)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Class-agnostic greedy NMS. Returns kept indices in descending score order."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, BOX_COLUMNS)
    order = np.argsort(-np.asarray(scores, dtype=np.float32), kind="stable")
    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        rest = order[1:]
        if rest.size == 0:
            break
        overlaps = _iou_one_to_many(boxes[best], boxes[rest])
        order = rest[overlaps <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def scale_boxes(boxes_xyxy: np.ndarray, scale: float, pad_xy: Tuple[int, int],
                frame_shape: Sequence[int]) -> np.ndarray:
    """Map letterboxed network boxes back to frame pixels, clipped to the frame."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, BOX_COLUMNS).copy()
    pad_x, pad_y = pad_xy
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_height - 1)
    return np.round(boxes).astype(np.int64)


def postprocess_detections(output: np.ndarray, num_classes: int, conf_threshold: float,
                           iou_threshold: float, scale: float, pad_xy: Tuple[int, int],
                           frame_shape: Sequence[int]) -> List[Detection]:
    """Decode -> NMS -> rescale raw network output into frame-space detections."""
    boxes, scores = decode_yolov8(output, num_classes, conf_threshold)
    if boxes.shape[0] == 0:
        return []
    keep = nms(boxes, scores, iou_threshold)
    frame_boxes = scale_boxes(boxes[keep], scale, pad_xy, frame_shape)
    return [Detection(int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(score))
            for box, score in zip(frame_boxes, scores[keep])]
