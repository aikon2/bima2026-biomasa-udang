"""Aspect-preserving letterbox preprocessing for YOLOv8 (pure numpy + OpenCV)."""
from typing import Tuple

import cv2
import numpy as np

LETTERBOX_PAD_VALUE = 114
PIXEL_MAX_VALUE = 255.0


def letterbox(frame_bgr: np.ndarray,
              input_hw: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize with preserved aspect ratio and pad to input_hw.

    Returns (blob[1,3,H,W] float32 RGB in [0,1], scale, (pad_x, pad_y)) where
    network_xy = frame_xy * scale + pad.
    """
    input_height, input_width = int(input_hw[0]), int(input_hw[1])
    frame_height, frame_width = frame_bgr.shape[:2]
    scale = min(input_height / frame_height, input_width / frame_width)
    resized_width = int(round(frame_width * scale))
    resized_height = int(round(frame_height * scale))
    pad_x = (input_width - resized_width) // 2
    pad_y = (input_height - resized_height) // 2

    if (resized_width, resized_height) != (frame_width, frame_height):
        resized = cv2.resize(frame_bgr, (resized_width, resized_height),
                             interpolation=cv2.INTER_LINEAR)
    else:
        resized = frame_bgr

    canvas = np.full((input_height, input_width, 3), LETTERBOX_PAD_VALUE, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized

    rgb = canvas[:, :, ::-1]
    chw = rgb.transpose(2, 0, 1)[np.newaxis]
    blob = np.ascontiguousarray(chw, dtype=np.float32) / PIXEL_MAX_VALUE
    return blob, float(scale), (int(pad_x), int(pad_y))
