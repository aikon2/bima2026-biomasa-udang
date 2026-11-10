from src.camera import Camera
from src.config import load_yaml, resolve_path
from src.detector_trt import ShrimpDetectorTRT
from src.pipeline import Pipeline

cfg_paths = load_yaml("config/paths.yaml")
cfg_cam = load_yaml("config/camera.yaml")
cfg_cal = load_yaml("config/calibration.yaml")
cfg_bio = load_yaml("config/biomass.yaml")

cam = Camera(cfg_cam)
detector = ShrimpDetectorTRT(str(resolve_path(cfg_paths["engine"])))

Pipeline(cam, detector, cfg_cal["cm_per_pixel"],
         cfg_bio["a"], cfg_bio["b"]).run()
