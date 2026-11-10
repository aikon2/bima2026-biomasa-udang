import json
import sys
import types

import run_trt
from tests.test_pipeline import BAR_DETECTION, FakeCamera, FakeDetector

MAX_FRAMES = 2


class ContextFakeCamera(FakeCamera):
    def __init__(self, cfg, source=None):
        FakeCamera.__init__(self)
        self.source = source

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return False


def fake_detector_module():
    module = types.ModuleType("src.detector_trt")

    def build(engine_path, conf_threshold, iou_threshold, num_classes):
        return FakeDetector([BAR_DETECTION])

    module.ShrimpDetectorTRT = build
    return module


def test_run_wires_configs_and_writes_csv_and_summary(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "src.detector_trt", fake_detector_module())
    monkeypatch.setattr("src.camera.Camera", ContextFakeCamera)
    output_csv = tmp_path / "session.csv"
    args = run_trt.parse_args(["--headless", "--source", "0", "--output", str(output_csv),
                               "--max-frames", str(MAX_FRAMES)])
    summary = run_trt.run(args)

    assert summary["frames"] == MAX_FRAMES
    assert summary["n_measured"] == MAX_FRAMES
    assert output_csv.is_file()
    saved = json.loads((tmp_path / "session_summary.json").read_text())
    assert saved["frames"] == MAX_FRAMES
    assert "fps" in saved and "pond_biomass_kg" in saved
