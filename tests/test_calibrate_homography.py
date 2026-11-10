import numpy as np
import pytest
import yaml

from calibration.calibrate_homography import load_point_pairs, write_homography


def test_load_point_pairs(tmp_path):
    csv_path = tmp_path / "points.csv"
    csv_path.write_text("u,v,x_cm,y_cm\n0,0,0,0\n\n100,0,10,0\n100,100,10,10\n0,100,0,10\n")
    points_px, points_cm = load_point_pairs(csv_path)
    assert points_px.shape == (4, 2)
    assert points_cm[2] == pytest.approx([10.0, 10.0])


def test_load_point_pairs_rejects_bad_value(tmp_path):
    csv_path = tmp_path / "points.csv"
    csv_path.write_text("u,v,x_cm,y_cm\n0,0,0,zero\n")
    with pytest.raises(ValueError):
        load_point_pairs(csv_path)


def test_write_homography_preserves_scale(tmp_path):
    cfg_path = tmp_path / "calibration.yaml"
    cfg_path.write_text("cm_per_pixel: 0.035\nhomography: null\n")
    homography = np.eye(3).tolist()
    write_homography(homography, cfg_path)
    written = yaml.safe_load(cfg_path.read_text())
    assert written["cm_per_pixel"] == pytest.approx(0.035)
    assert written["homography"] == homography
