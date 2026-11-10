import numpy as np

from src.preprocess import LETTERBOX_PAD_VALUE, letterbox

INPUT_HW = (640, 640)


def make_frame(height, width):
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_letterbox_shape_dtype_scale_and_pad():
    blob, scale, pad = letterbox(make_frame(360, 640), INPUT_HW)
    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert blob.flags["C_CONTIGUOUS"]
    assert scale == 1.0
    assert pad == (0, 140)


def test_letterbox_pad_pixels_use_pad_value():
    blob, _, _ = letterbox(make_frame(360, 640), INPUT_HW)
    expected = LETTERBOX_PAD_VALUE / 255.0
    assert np.allclose(blob[0, :, 0, :], expected)
    assert np.allclose(blob[0, :, 639, :], expected)
    assert np.allclose(blob[0, :, 320, :], 0.0)


def test_letterbox_converts_bgr_to_rgb():
    frame = make_frame(360, 640)
    frame[:, :] = (255, 0, 0)  # pure blue in BGR
    blob, _, _ = letterbox(frame, INPUT_HW)
    center = blob[0, :, 320, 320]
    assert center[2] == 1.0
    assert center[0] == 0.0


def test_letterbox_downscales_large_frame():
    blob, scale, pad = letterbox(make_frame(720, 1280), INPUT_HW)
    assert blob.shape == (1, 3, 640, 640)
    assert scale == 0.5
    assert pad == (0, 140)
