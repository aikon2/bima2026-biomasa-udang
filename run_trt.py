
import yaml
from src.camera import Camera
from src.detector_trt import ShrimpDetectorTRT
from src.pipeline import Pipeline

cfg_cam = yaml.safe_load(open("config/camera.yaml"))
cfg_cal = yaml.safe_load(open("config/calibration.yaml"))
cfg_bio = yaml.safe_load(open("config/biomass.yaml"))

cam = Camera(cfg_cam)
detector = ShrimpDetectorTRT("models/shrimp_fp16.trt")

Pipeline(cam, detector, cfg_cal["cm_per_pixel"],
         cfg_bio["a"], cfg_bio["b"]).run()
