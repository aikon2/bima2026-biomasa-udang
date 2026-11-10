"""Frame sources: USB/V4L2 device index, Jetson CSI camera (GStreamer) or video file."""
import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from .config import resolve_path

LOGGER = logging.getLogger(__name__)

SOURCE_CSI = "csi"
DEFAULT_CSI_SENSOR_ID = 0
DEFAULT_CSI_FLIP_METHOD = 0
CSI_PIPELINE_TEMPLATE = (
    "nvarguscamerasrc sensor-id={sensor_id} ! "
    "video/x-raw(memory:NVMM), width={width}, height={height}, "
    "format=NV12, framerate={fps}/1 ! "
    "nvvidconv flip-method={flip_method} ! "
    "video/x-raw, width={width}, height={height}, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

SourceSpec = Union[int, str]


def build_csi_pipeline(cfg: dict) -> str:
    """GStreamer pipeline string for a CSI camera (nvarguscamerasrc -> BGR appsink)."""
    csi_cfg = cfg.get("csi") or {}
    return CSI_PIPELINE_TEMPLATE.format(
        sensor_id=int(csi_cfg.get("sensor_id", DEFAULT_CSI_SENSOR_ID)),
        flip_method=int(csi_cfg.get("flip_method", DEFAULT_CSI_FLIP_METHOD)),
        width=int(cfg["width"]), height=int(cfg["height"]), fps=int(cfg["fps"]))


def parse_source(source: Optional[SourceSpec], cfg: dict) -> SourceSpec:
    """None -> config device; digit string -> device index; 'csi'; else a file path."""
    if source is None:
        source = cfg.get("device", 0)
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if text.isdigit():
        return int(text)
    if text.lower() == SOURCE_CSI:
        return SOURCE_CSI
    return text


class Camera(object):
    """cv2.VideoCapture wrapper with clear open errors and context-manager support."""

    def __init__(self, cfg: dict, source: Optional[SourceSpec] = None) -> None:
        self.cfg = cfg
        self.source = parse_source(source, cfg)
        self.cap = self._open(self.source)

    def _open(self, source: SourceSpec):
        if source == SOURCE_CSI:
            pipeline = build_csi_pipeline(self.cfg)
            LOGGER.info("Opening CSI camera: %s", pipeline)
            capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        elif isinstance(source, int):
            LOGGER.info("Opening camera device %d", source)
            capture = cv2.VideoCapture(source)
        else:
            video_path = resolve_path(source)
            if not video_path.is_file():
                raise RuntimeError("Video file not found: {}".format(video_path))
            LOGGER.info("Opening video file %s", video_path)
            capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Cannot open camera source {!r} (check device, cable "
                               "or GStreamer support)".format(source))
        if isinstance(source, int):
            self._apply_device_settings(capture)
        return capture

    def _apply_device_settings(self, capture) -> None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg["width"]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg["height"]))
        capture.set(cv2.CAP_PROP_FPS, int(self.cfg["fps"]))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self) -> None:
        """Release the capture; safe to call more than once."""
        if self.cap is None:
            return
        self.cap.release()
        self.cap = None

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.release()
        return False
