import math

import cv2
import numpy as np
import pytest

from src.geometry import PlaneScale
from src.measurement import (MeasureResult, STATUS_EDGE, STATUS_EMPTY_ROI, STATUS_NO_CONTOUR,
                             STATUS_OK, measure_length)
from src.postprocess import Detection

FRAME_HEIGHT = 360
FRAME_WIDTH = 640
BRIGHT = 220
DARK = 30
CM_PER_PIXEL = 0.1
BAR_START = (200, 180)
BAR_END = (400, 180)
BAR_THICKNESS = 16
SKELETON_CFG = {"method": "skeleton", "edge_margin_px": 4, "min_roi_px": 8, "morph_kernel": 3}
RECT_CFG = dict(SKELETON_CFG, method="min_area_rect")
ARC_CENTER = (320, 220)
ARC_RADIUS = 100


def make_frame(background, foreground, draw):
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), background, dtype=np.uint8)
    draw(frame, (foreground, foreground, foreground))
    return frame


def draw_bar(frame, color):
    cv2.line(frame, BAR_START, BAR_END, color, BAR_THICKNESS)


def draw_arc(frame, color):
    cv2.ellipse(frame, ARC_CENTER, (ARC_RADIUS, ARC_RADIUS), 0, 180, 360, color, BAR_THICKNESS)


def bar_detection():
    return Detection(180, 160, 420, 200, 0.9)


def arc_detection():
    return Detection(200, 100, 440, 240, 0.9)


def scale():
    return PlaneScale(CM_PER_PIXEL)


def test_result_fields():
    assert MeasureResult._fields == ("length_cm", "status")


def test_out_of_range_box_does_not_crash():
    frame = make_frame(DARK, BRIGHT, draw_bar)
    result = measure_length(frame, Detection(-50, -50, 2000, 2000, 0.9), scale(), SKELETON_CFG)
    assert isinstance(result, MeasureResult)


def test_empty_box_is_empty_roi():
    frame = make_frame(DARK, BRIGHT, draw_bar)
    result = measure_length(frame, Detection(300, 100, 300, 100, 0.9), scale(), SKELETON_CFG)
    assert result == MeasureResult(None, STATUS_EMPTY_ROI)


def test_inverted_box_is_empty_roi():
    frame = make_frame(DARK, BRIGHT, draw_bar)
    result = measure_length(frame, Detection(400, 200, 300, 100, 0.9), scale(), SKELETON_CFG)
    assert result.status == STATUS_EMPTY_ROI


def test_tiny_box_is_empty_roi():
    frame = make_frame(DARK, BRIGHT, draw_bar)
    result = measure_length(frame, Detection(300, 100, 303, 140, 0.9), scale(), SKELETON_CFG)
    assert result.status == STATUS_EMPTY_ROI


def test_box_touching_border_is_edge():
    frame = make_frame(DARK, BRIGHT, draw_bar)
    result = measure_length(frame, Detection(2, 160, 420, 200, 0.9), scale(), SKELETON_CFG)
    assert result == MeasureResult(None, STATUS_EDGE)


def test_uniform_roi_has_no_contour():
    frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), DARK, dtype=np.uint8)
    result = measure_length(frame, bar_detection(), scale(), SKELETON_CFG)
    assert result == MeasureResult(None, STATUS_NO_CONTOUR)


@pytest.mark.parametrize("background,foreground", [(DARK, BRIGHT), (BRIGHT, DARK)])
@pytest.mark.parametrize("cfg", [SKELETON_CFG, RECT_CFG])
def test_bar_length_both_polarities(background, foreground, cfg):
    frame = make_frame(background, foreground, draw_bar)
    result = measure_length(frame, bar_detection(), scale(), cfg)
    true_length_cm = (BAR_END[0] - BAR_START[0]) * CM_PER_PIXEL
    assert result.status == STATUS_OK
    assert result.length_cm == pytest.approx(true_length_cm, rel=0.10)


@pytest.mark.parametrize("background,foreground", [(DARK, BRIGHT), (BRIGHT, DARK)])
def test_curved_skeleton_follows_arc(background, foreground):
    frame = make_frame(background, foreground, draw_arc)
    skeleton = measure_length(frame, arc_detection(), scale(), SKELETON_CFG)
    rect = measure_length(frame, arc_detection(), scale(), RECT_CFG)
    true_arc_cm = math.pi * ARC_RADIUS * CM_PER_PIXEL
    assert skeleton.status == STATUS_OK
    assert skeleton.length_cm == pytest.approx(true_arc_cm, rel=0.15)
    assert skeleton.length_cm > rect.length_cm


def test_unknown_method_raises():
    frame = make_frame(DARK, BRIGHT, draw_bar)
    with pytest.raises(ValueError):
        measure_length(frame, bar_detection(), scale(), dict(SKELETON_CFG, method="bogus"))
