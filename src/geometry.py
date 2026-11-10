"""Image-pixel to measurement-plane (cm) conversion.

A 3x3 homography (image px -> plane cm) corrects perspective when the camera is
not perpendicular to the measurement plane; otherwise a scalar cm_per_pixel is used.
"""
import logging
from typing import List, Optional, Sequence

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

MIN_HOMOGRAPHY_POINTS = 4
HOMOGRAPHY_SHAPE = (3, 3)


class PlaneScale(object):
    """Maps frame pixel coordinates onto the measurement plane in centimetres."""

    def __init__(self, cm_per_pixel: float,
                 homography: Optional[Sequence[Sequence[float]]] = None) -> None:
        self.cm_per_pixel = float(cm_per_pixel)
        self.homography = None
        if homography is not None:
            matrix = np.asarray(homography, dtype=np.float64)
            if matrix.shape != HOMOGRAPHY_SHAPE:
                raise ValueError("homography must be 3x3, got {}".format(matrix.shape))
            self.homography = matrix

    @classmethod
    def from_config(cls, calibration_cfg: dict) -> "PlaneScale":
        """Build from config/calibration.yaml ({cm_per_pixel, homography|null})."""
        homography = calibration_cfg.get("homography")
        if homography is None:
            LOGGER.info("No homography configured; using scalar cm_per_pixel")
        return cls(calibration_cfg["cm_per_pixel"], homography)

    def to_plane(self, points_px: np.ndarray) -> np.ndarray:
        """Project Nx2 pixel points (x, y) to Nx2 plane points in cm."""
        points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        if self.homography is None:
            return points * self.cm_per_pixel
        projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), self.homography)
        return projected.reshape(-1, 2)

    def polyline_length_cm(self, points_px: np.ndarray) -> float:
        """Length in cm of the polyline through the given pixel points."""
        plane_points = self.to_plane(points_px)
        if plane_points.shape[0] < 2:
            return 0.0
        segments = np.diff(plane_points, axis=0)
        return float(np.sum(np.hypot(segments[:, 0], segments[:, 1])))


def compute_homography(points_px: np.ndarray, points_cm: np.ndarray) -> List[List[float]]:
    """Least-squares homography from >=4 pixel/plane point pairs, as a nested list."""
    source = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(points_cm, dtype=np.float64).reshape(-1, 2)
    if source.shape[0] != target.shape[0]:
        raise ValueError("points_px and points_cm must have the same length")
    if source.shape[0] < MIN_HOMOGRAPHY_POINTS:
        raise ValueError("At least {} point pairs are required, got {}".format(
            MIN_HOMOGRAPHY_POINTS, source.shape[0]))
    matrix, _ = cv2.findHomography(source, target, 0)
    if matrix is None:
        raise ValueError("Homography estimation failed (degenerate points?)")
    matrix = matrix / matrix[2, 2]
    return matrix.tolist()


def reprojection_errors_cm(homography: Sequence[Sequence[float]], points_px: np.ndarray,
                           points_cm: np.ndarray) -> np.ndarray:
    """Per-point distance (cm) between projected pixel points and their targets."""
    projected = PlaneScale(1.0, homography).to_plane(points_px)
    target = np.asarray(points_cm, dtype=np.float64).reshape(-1, 2)
    delta = projected - target
    return np.hypot(delta[:, 0], delta[:, 1])
