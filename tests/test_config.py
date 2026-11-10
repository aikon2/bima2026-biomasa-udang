from pathlib import Path

from src.config import REPO_ROOT, load_yaml, resolve_path


def test_repo_root_contains_config_dir():
    assert (REPO_ROOT / "config").is_dir()


def test_load_paths_yaml_engine_path():
    paths_cfg = load_yaml("config/paths.yaml")
    assert paths_cfg["engine"].endswith("shrimp_fp16.trt")
    assert paths_cfg["onnx"] == "models/yolo/shrimp.onnx"
    assert paths_cfg["weights_pt"] == "models/yolo/shrimp.pt"


def test_load_detector_yaml_defaults():
    detector_cfg = load_yaml("config/detector.yaml")
    assert detector_cfg["imgsz"] == 640
    assert 0.0 < detector_cfg["conf_threshold"] < 1.0
    assert 0.0 < detector_cfg["iou_threshold"] < 1.0


def test_resolve_path_is_absolute_under_repo_root():
    resolved = resolve_path("models/yolo/shrimp.onnx")
    assert resolved.is_absolute()
    assert resolved == REPO_ROOT / "models" / "yolo" / "shrimp.onnx"


def test_resolve_path_keeps_absolute_input():
    absolute = Path("/tmp/some_file.txt")
    assert resolve_path(str(absolute)) == absolute
