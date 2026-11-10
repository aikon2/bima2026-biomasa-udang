import csv
import subprocess
import sys

import cv2
import numpy as np
import pytest

from src.config import REPO_ROOT
from src.geometry import PlaneScale
from src.measurement import STATUS_EDGE, STATUS_OK
from src.pipeline import CSV_HEADER, Pipeline
from src.postprocess import Detection

FRAME_SHAPE = (240, 320, 3)
FRAME_COUNT = 3
BAR_START = (100, 120)
BAR_END = (220, 120)
BAR_THICKNESS = 10
BAR_DETECTION = Detection(80, 100, 240, 140, 0.9)
EDGE_DETECTION = Detection(0, 10, 60, 50, 0.8)
MEASUREMENT_CFG = {"method": "skeleton", "edge_margin_px": 4, "min_roi_px": 8, "morph_kernel": 3}
BIOMASS_CFG = {"a": 0.01, "b": 3.0}
CM_PER_PIXEL = 0.1
EXPECTED_HEADER = "timestamp,frame_idx,x1,y1,x2,y2,conf,length_cm,weight_g,status"


def make_bar_frame():
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    cv2.line(frame, BAR_START, BAR_END, (220, 220, 220), BAR_THICKNESS)
    return frame


class FakeCamera(object):
    def __init__(self, frame_count=FRAME_COUNT):
        self.remaining = frame_count
        self.released = False

    def read(self):
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        return True, make_bar_frame()

    def release(self):
        self.released = True


class FakeDetector(object):
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame_bgr):
        return list(self.detections)


class FailingDetector(object):
    def detect(self, frame_bgr):
        raise RuntimeError("inference failed")


def make_pipeline(camera, detector):
    return Pipeline(camera, detector, PlaneScale(CM_PER_PIXEL), MEASUREMENT_CFG, BIOMASS_CFG)


def read_rows(csv_path):
    with open(str(csv_path), "r", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def test_headless_run_writes_csv_and_summary(tmp_path):
    camera = FakeCamera()
    output_csv = tmp_path / "run.csv"
    summary = make_pipeline(camera, FakeDetector([BAR_DETECTION])).run(
        headless=True, output_csv=output_csv, max_frames=None)

    assert ",".join(CSV_HEADER) == EXPECTED_HEADER
    assert output_csv.read_text().splitlines()[0] == EXPECTED_HEADER
    rows = read_rows(output_csv)
    assert len(rows) == FRAME_COUNT
    assert [row["frame_idx"] for row in rows] == ["0", "1", "2"]
    assert all(row["status"] == STATUS_OK for row in rows)
    assert float(rows[0]["length_cm"]) == pytest.approx(12.0, rel=0.15)
    assert summary["n_measured"] == FRAME_COUNT
    assert summary["n_detected"] == FRAME_COUNT
    assert summary["frames"] == FRAME_COUNT
    assert "fps" in summary and summary["fps"] > 0
    assert summary["abw_g"] is not None
    assert camera.released


def test_non_ok_detections_are_logged_with_empty_values(tmp_path):
    output_csv = tmp_path / "run.csv"
    summary = make_pipeline(FakeCamera(), FakeDetector([BAR_DETECTION, EDGE_DETECTION])).run(
        headless=True, output_csv=output_csv, max_frames=2)

    rows = read_rows(output_csv)
    assert len(rows) == 4
    edge_rows = [row for row in rows if row["status"] == STATUS_EDGE]
    assert len(edge_rows) == 2
    assert edge_rows[0]["length_cm"] == "" and edge_rows[0]["weight_g"] == ""
    assert summary["frames"] == 2
    assert summary["n_detected"] == 4
    assert summary["n_measured"] == 2


def test_camera_released_when_detector_raises(tmp_path):
    camera = FakeCamera()
    with pytest.raises(RuntimeError, match="inference failed"):
        make_pipeline(camera, FailingDetector()).run(
            headless=True, output_csv=tmp_path / "run.csv", max_frames=None)
    assert camera.released


def test_run_without_csv_output(tmp_path):
    summary = make_pipeline(FakeCamera(), FakeDetector([])).run(
        headless=True, output_csv=None, max_frames=None)
    assert summary["frames"] == FRAME_COUNT
    assert summary["n_detected"] == 0
    assert summary["abw_g"] is None


def test_run_trt_help_works_without_tensorrt():
    result = subprocess.run([sys.executable, str(REPO_ROOT / "run_trt.py"), "--help"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=str(REPO_ROOT.parent), timeout=60)
    assert result.returncode == 0, result.stderr.decode()
    help_text = result.stdout.decode()
    for option in ("--source", "--headless", "--output", "--max-frames"):
        assert option in help_text


class InterruptingCamera(FakeCamera):
    def read(self):
        if self.remaining <= 1:
            raise KeyboardInterrupt
        return FakeCamera.read(self)


def test_keyboard_interrupt_stops_gracefully_and_keeps_log(tmp_path):
    camera = InterruptingCamera()
    output_csv = tmp_path / "run.csv"
    summary = make_pipeline(camera, FakeDetector([BAR_DETECTION])).run(
        headless=True, output_csv=output_csv, max_frames=None)
    assert summary["frames"] == FRAME_COUNT - 1
    assert len(read_rows(output_csv)) == FRAME_COUNT - 1
    assert camera.released
