"""Frame loop: detect -> measure -> weigh -> log, with optional on-screen overlay.

Without tracking, the same shrimp can be sampled in several frames; session
statistics (ABW, size, CV%) are therefore sample statistics and counts are
detection counts, not population counts.
"""
import csv
import datetime
import logging
import time
from collections import namedtuple
from pathlib import Path
from typing import List, Optional

import cv2

from .biomass import summarize_session, weight_g
from .measurement import STATUS_OK, measure_length

LOGGER = logging.getLogger(__name__)

CSV_HEADER = ("timestamp", "frame_idx", "x1", "y1", "x2", "y2", "conf",
              "length_cm", "weight_g", "status")
WINDOW_NAME = "Shrimp Biomass"
ESC_KEY = 27
WAIT_KEY_MS = 1
COLOR_OK = (0, 255, 0)
COLOR_REJECTED = (0, 0, 255)
COLOR_TEXT = (0, 255, 255)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
TEXT_THICKNESS = 2
TEXT_ORIGIN_X = 10
TEXT_FIRST_LINE_Y = 25
TEXT_LINE_HEIGHT = 25
LABEL_OFFSET_Y = 5
CONF_DECIMALS = 3
VALUE_DECIMALS = 3

Record = namedtuple("Record", "timestamp frame_idx detection length_cm weight_g status")


def _format_optional(value: Optional[float]) -> str:
    if value is None:
        return ""
    return "{:.{}f}".format(value, VALUE_DECIMALS)


def record_to_row(record: Record) -> List[str]:
    """CSV row; length/weight are empty for non-ok statuses."""
    detection = record.detection
    return [record.timestamp, str(record.frame_idx), str(detection.x1), str(detection.y1),
            str(detection.x2), str(detection.y2),
            "{:.{}f}".format(detection.conf, CONF_DECIMALS),
            _format_optional(record.length_cm), _format_optional(record.weight_g),
            record.status]


def _fmt_or_dash(value: Optional[float], template: str) -> str:
    return "-" if value is None else template.format(value)


def draw_overlay(frame, records: List[Record], summary: dict) -> None:
    """Boxes plus frame counts, visible (frame) biomass and session ABW/size."""
    for record in records:
        detection = record.detection
        color = COLOR_OK if record.status == STATUS_OK else COLOR_REJECTED
        cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2),
                      color, BOX_THICKNESS)
        label = _fmt_or_dash(record.length_cm, "{:.1f} cm") if record.status == STATUS_OK \
            else record.status
        cv2.putText(frame, label, (detection.x1, max(detection.y1 - LABEL_OFFSET_Y, 0)),
                    FONT, FONT_SCALE, color, TEXT_THICKNESS)
    measured = [record for record in records if record.weight_g is not None]
    visible_biomass = sum(record.weight_g for record in measured)
    lines = [
        "Detected: {}  Measured: {}".format(len(records), len(measured)),
        "Visible biomass (frame): {:.1f} g".format(visible_biomass),
        "ABW: {}  Size: {}".format(_fmt_or_dash(summary.get("abw_g"), "{:.2f} g"),
                                   _fmt_or_dash(summary.get("size_pcs_per_kg"),
                                                "{:.0f} pcs/kg")),
    ]
    for index, text in enumerate(lines):
        origin = (TEXT_ORIGIN_X, TEXT_FIRST_LINE_Y + index * TEXT_LINE_HEIGHT)
        cv2.putText(frame, text, origin, FONT, FONT_SCALE, COLOR_TEXT, TEXT_THICKNESS)


class Pipeline(object):
    """Runs the measurement loop over a camera with an injected detector."""

    def __init__(self, camera, detector, plane_scale, measurement_cfg: dict,
                 biomass_cfg: dict) -> None:
        self.camera = camera
        self.detector = detector
        self.plane_scale = plane_scale
        self.measurement_cfg = measurement_cfg
        self.a = float(biomass_cfg["a"])
        self.b = float(biomass_cfg["b"])
        self.pond_cfg = biomass_cfg.get("pond")
        self.lengths = []
        self.weights = []
        self.n_detected = 0

    def process_frame(self, frame, frame_idx: int) -> List[Record]:
        """Measure every detection in a frame and update session accumulators."""
        timestamp = datetime.datetime.now().isoformat()
        records = []
        for detection in self.detector.detect(frame):
            result = measure_length(frame, detection, self.plane_scale, self.measurement_cfg)
            weight = weight_g(result.length_cm, self.a, self.b)
            status = result.status
            if status == STATUS_OK and weight is not None:
                self.lengths.append(result.length_cm)
                self.weights.append(weight)
            records.append(Record(timestamp, frame_idx, detection, result.length_cm, weight,
                                  status))
        self.n_detected += len(records)
        return records

    def summary(self) -> dict:
        return summarize_session(self.lengths, self.weights, self.n_detected, self.pond_cfg)

    def _show(self, frame, records: List[Record]) -> bool:
        """Draw and display; returns False when the user pressed ESC."""
        draw_overlay(frame, records, self.summary())
        cv2.imshow(WINDOW_NAME, frame)
        return (cv2.waitKey(WAIT_KEY_MS) & 0xFF) != ESC_KEY

    def _step(self, headless: bool, writer, frame_idx: int) -> bool:
        """Process one frame; returns False when the loop should stop."""
        ok, frame = self.camera.read()
        if not ok or frame is None:
            return False
        records = self.process_frame(frame, frame_idx)
        if writer is not None:
            writer.writerows(record_to_row(record) for record in records)
        if not headless and not self._show(frame, records):
            LOGGER.info("Stopped by user (ESC)")
            return False
        return True

    def _loop(self, headless: bool, writer, max_frames: Optional[int]) -> int:
        frame_idx = 0
        # Ctrl+C is the normal way to stop a headless run; keep the log and summary.
        try:
            while max_frames is None or frame_idx < max_frames:
                if not self._step(headless, writer, frame_idx):
                    break
                frame_idx += 1
        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user after %d frames", frame_idx)
        return frame_idx

    def _run_with_output(self, headless: bool, output_csv: Optional[Path],
                         max_frames: Optional[int]) -> int:
        if output_csv is None:
            return self._loop(headless, None, max_frames)
        output_path = Path(str(output_csv))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CSV_HEADER)
            frames = self._loop(headless, writer, max_frames)
        LOGGER.info("Detections logged to %s", output_path)
        return frames

    def run(self, headless: bool, output_csv: Optional[Path] = None,
            max_frames: Optional[int] = None) -> dict:
        """Process frames until the source ends, max_frames or ESC; returns the summary."""
        start = time.monotonic()
        try:
            frames = self._run_with_output(headless, output_csv, max_frames)
        finally:
            self.camera.release()
            if not headless:
                cv2.destroyAllWindows()
        elapsed = time.monotonic() - start
        summary = self.summary()
        summary["frames"] = frames
        summary["fps"] = frames / elapsed if elapsed > 0 else 0.0
        return summary
