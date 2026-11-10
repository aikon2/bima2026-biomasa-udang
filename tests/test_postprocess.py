import numpy as np

from src.postprocess import Detection, decode_yolov8, nms, scale_boxes

NUM_ANCHORS = 8400
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45


def make_output(num_classes, rows):
    """rows: list of (anchor_idx, cx, cy, w, h, class_scores)."""
    output = np.zeros((1, 4 + num_classes, NUM_ANCHORS), dtype=np.float32)
    for anchor_idx, cx, cy, w, h, class_scores in rows:
        output[0, :4, anchor_idx] = (cx, cy, w, h)
        output[0, 4:, anchor_idx] = class_scores
    return output


def test_detection_fields():
    assert Detection._fields == ("x1", "y1", "x2", "y2", "conf")


def test_decode_single_class_two_boxes():
    output = make_output(1, [
        (10, 100, 100, 20, 40, [0.9]),
        (500, 300, 200, 60, 30, [0.5]),
        (900, 50, 50, 10, 10, [0.1]),
    ])
    boxes, scores = decode_yolov8(output, 1, CONF_THRESHOLD)
    assert boxes.shape == (2, 4)
    order = np.argsort(-scores)
    assert np.allclose(boxes[order[0]], [90, 80, 110, 120])
    assert np.allclose(boxes[order[1]], [270, 185, 330, 215])
    assert np.allclose(np.sort(scores)[::-1], [0.9, 0.5])


def test_decode_multi_class_picks_max_score():
    output = make_output(2, [(7, 100, 100, 20, 20, [0.2, 0.8])])
    boxes, scores = decode_yolov8(output, 2, CONF_THRESHOLD)
    assert boxes.shape == (1, 4)
    assert np.isclose(scores[0], 0.8)


def test_decode_nothing_above_threshold_returns_empty():
    output = make_output(1, [(3, 10, 10, 5, 5, [0.1])])
    boxes, scores = decode_yolov8(output, 1, CONF_THRESHOLD)
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)


def test_nms_suppresses_overlap_keeps_higher_score():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], dtype=np.float32)
    scores = np.array([0.6, 0.9], dtype=np.float32)
    keep = nms(boxes, scores, IOU_THRESHOLD)
    assert list(keep) == [1]


def test_nms_keeps_disjoint_boxes_in_score_order():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.5, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, IOU_THRESHOLD)
    assert list(keep) == [1, 0]


def test_nms_empty_input():
    keep = nms(np.zeros((0, 4), dtype=np.float32), np.zeros((0,)), IOU_THRESHOLD)
    assert len(keep) == 0


def test_scale_boxes_inverts_letterbox_and_clips():
    frame_shape = (720, 1280, 3)
    scale = 0.5
    pad = (0, 140)
    boxes = np.array([
        [100, 190, 200, 240],     # inside -> (200, 100, 400, 200)
        [-20, 100, 50, 150],      # negative x, y inside pad
        [600, 400, 700, 600],     # overflow beyond frame
    ], dtype=np.float32)
    scaled = scale_boxes(boxes, scale, pad, frame_shape)
    assert scaled.dtype.kind == "i"
    assert scaled[0].tolist() == [200, 100, 400, 200]
    assert scaled[1].tolist() == [0, 0, 100, 20]
    assert scaled[2].tolist() == [1200, 520, 1279, 719]


def test_postprocess_detections_full_chain():
    from src.postprocess import postprocess_detections

    output = make_output(1, [
        (10, 100, 190, 100, 50, [0.9]),
        (11, 102, 191, 100, 50, [0.6]),   # duplicate of the first, suppressed
        (900, 50, 50, 10, 10, [0.1]),     # below threshold
    ])
    detections = postprocess_detections(
        output, num_classes=1, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD,
        scale=0.5, pad_xy=(0, 140), frame_shape=(720, 1280, 3))
    assert len(detections) == 1
    detection = detections[0]
    assert isinstance(detection, Detection)
    assert (detection.x1, detection.y1, detection.x2, detection.y2) == (100, 50, 300, 150)
    assert all(isinstance(value, int) for value in detection[:4])
    assert isinstance(detection.conf, float)
    assert abs(detection.conf - 0.9) < 1e-6
