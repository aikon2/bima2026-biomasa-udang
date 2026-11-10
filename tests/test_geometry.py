import numpy as np
import pytest

from src.geometry import PlaneScale, compute_homography

CM_PER_PIXEL = 0.1
LINE_LENGTH_PX = 100.0


def test_scalar_scale_line_length():
    scale = PlaneScale.from_config({"cm_per_pixel": CM_PER_PIXEL, "homography": None})
    points = np.array([[10.0, 20.0], [10.0 + LINE_LENGTH_PX, 20.0]])
    assert scale.polyline_length_cm(points) == pytest.approx(LINE_LENGTH_PX * CM_PER_PIXEL)


def test_scalar_scale_missing_homography_key():
    scale = PlaneScale.from_config({"cm_per_pixel": CM_PER_PIXEL})
    assert scale.homography is None


def test_homography_scale_matches_scalar():
    homography = [[CM_PER_PIXEL, 0, 0], [0, CM_PER_PIXEL, 0], [0, 0, 1]]
    scale = PlaneScale.from_config({"cm_per_pixel": 0.5, "homography": homography})
    points = np.array([[0.0, 0.0], [60.0, 80.0], [60.0, 180.0]])
    expected = (100.0 + 100.0) * CM_PER_PIXEL
    assert scale.polyline_length_cm(points) == pytest.approx(expected)


def test_to_plane_with_homography_shape():
    homography = [[CM_PER_PIXEL, 0, 5], [0, CM_PER_PIXEL, 7], [0, 0, 1]]
    scale = PlaneScale.from_config({"cm_per_pixel": 0.5, "homography": homography})
    plane = scale.to_plane(np.array([[10.0, 20.0]]))
    assert plane.shape == (1, 2)
    assert plane[0] == pytest.approx([6.0, 9.0])


def test_compute_homography_recovers_known():
    known = np.array([[0.05, 0.01, 2.0], [-0.02, 0.06, 1.0], [0.0001, 0.0002, 1.0]])
    points_px = np.array([[0, 0], [640, 0], [640, 360], [0, 360], [320, 180]], dtype=np.float64)
    ones = np.ones((points_px.shape[0], 1))
    projected = np.hstack([points_px, ones]) @ known.T
    points_cm = projected[:, :2] / projected[:, 2:3]
    recovered = np.asarray(compute_homography(points_px, points_cm))
    assert recovered.shape == (3, 3)
    assert recovered / recovered[2, 2] == pytest.approx(known, rel=1e-4, abs=1e-6)


def test_compute_homography_requires_four_points():
    with pytest.raises(ValueError):
        compute_homography(np.zeros((3, 2)), np.zeros((3, 2)))


def test_short_polyline_is_zero():
    scale = PlaneScale(CM_PER_PIXEL)
    assert scale.polyline_length_cm(np.array([[1.0, 2.0]])) == 0.0
