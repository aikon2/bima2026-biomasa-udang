"""User-supplied CLI paths resolve against the current directory; defaults against the repo."""
import subprocess
import sys
from pathlib import Path

import pytest

from src.config import REPO_ROOT, resolve_cli_path

PYTHON = sys.executable


def test_resolve_cli_path_relative_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(str(tmp_path))
    assert resolve_cli_path("data/x.csv") == tmp_path.resolve() / "data" / "x.csv"


def test_resolve_cli_path_keeps_absolute():
    absolute = Path("/tmp/some_file.csv")
    assert resolve_cli_path(str(absolute)) == absolute


def test_run_trt_args_resolve_against_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(str(tmp_path))
    sys.path.insert(0, str(REPO_ROOT))
    import run_trt

    args = run_trt.parse_args(["--output", "out.csv", "--engine", "e.trt",
                               "--source", "clip.mp4"])
    assert Path(args.output) == tmp_path.resolve() / "out.csv"
    assert Path(args.engine) == tmp_path.resolve() / "e.trt"
    assert Path(args.source) == tmp_path.resolve() / "clip.mp4"


@pytest.mark.parametrize("source", ["0", "2", "csi"])
def test_run_trt_device_sources_untouched(source):
    import run_trt

    assert run_trt.parse_args(["--source", source]).source == source


def test_evaluate_reads_csv_relative_to_cwd(tmp_path):
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text("weight_gt_g,weight_pred_g\n8.0,8.2\n9.0,8.7\n10.0,10.4\n")
    result = subprocess.run([PYTHON, str(REPO_ROOT / "evaluation" / "evaluate.py"),
                             "--csv", "eval.csv"], cwd=str(tmp_path),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr.decode()


def test_homography_points_relative_to_cwd(tmp_path):
    points = "u,v,x_cm,y_cm\n0,0,0,0\n100,0,10,0\n100,100,10,10\n0,100,0,10\n"
    (tmp_path / "pts.csv").write_text(points)
    result = subprocess.run([PYTHON, str(REPO_ROOT / "calibration" / "calibrate_homography.py"),
                             "--points", "pts.csv", "--dry-run"], cwd=str(tmp_path),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr.decode()
