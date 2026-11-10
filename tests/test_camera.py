import pytest

from src.camera import Camera, build_csi_pipeline, parse_source

CAMERA_CFG = {"device": 0, "width": 640, "height": 360, "fps": 15,
              "csi": {"sensor_id": 0, "flip_method": 0}}


def test_build_csi_pipeline_uses_config_values():
    pipeline = build_csi_pipeline(CAMERA_CFG)
    assert pipeline.startswith("nvarguscamerasrc sensor-id=0")
    assert "width=640" in pipeline
    assert "height=360" in pipeline
    assert "framerate=15/1" in pipeline
    assert "flip-method=0" in pipeline
    assert pipeline.rstrip().endswith("appsink")


def test_parse_source_device_index_csi_and_file():
    assert parse_source(None, CAMERA_CFG) == 0
    assert parse_source("1", CAMERA_CFG) == 1
    assert parse_source("csi", CAMERA_CFG) == "csi"
    assert parse_source("clip.mp4", CAMERA_CFG) == "clip.mp4"


def test_missing_video_file_raises_clear_error(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(RuntimeError, match="Video file not found"):
        Camera(CAMERA_CFG, source=str(missing))


def test_camera_is_context_manager_and_release_is_idempotent(tmp_path, monkeypatch):
    released = []

    class FakeCapture(object):
        def isOpened(self):
            return True

        def set(self, prop, value):
            return True

        def read(self):
            return False, None

        def release(self):
            released.append(True)

    monkeypatch.setattr("src.camera.cv2.VideoCapture", lambda *args: FakeCapture())
    with Camera(CAMERA_CFG, source="0") as camera:
        assert camera.read() == (False, None)
    camera.release()
    assert released == [True]


def test_unopened_device_raises_clear_error(monkeypatch):
    class ClosedCapture(object):
        def isOpened(self):
            return False

        def release(self):
            return None

    monkeypatch.setattr("src.camera.cv2.VideoCapture", lambda *args: ClosedCapture())
    with pytest.raises(RuntimeError, match="Cannot open camera"):
        Camera(CAMERA_CFG, source="3")
