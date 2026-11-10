"""Run shrimp length & biomass estimation with a TensorRT YOLOv8 engine.

Examples:
  python3 run_trt.py                      # camera.yaml device, with display
  python3 run_trt.py --source csi --headless --max-frames 500
  python3 run_trt.py --source video.mp4 --output outputs/test.csv
"""
import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_yaml, resolve_path  # noqa: E402

LOGGER = logging.getLogger("run_trt")

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
SUMMARY_SUFFIX = "_summary.json"
JSON_INDENT = 2


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shrimp length & biomass estimation (TensorRT YOLOv8, Jetson Nano).")
    parser.add_argument("--source", default=None,
                        help="Device index (e.g. 0), 'csi' for a Jetson CSI camera, or a "
                             "video file path. Default: 'device' in config/camera.yaml.")
    parser.add_argument("--headless", action="store_true",
                        help="Do not open a display window (SSH / no monitor).")
    parser.add_argument("--output", default=None,
                        help="Per-detection CSV log path. Default: "
                             "<output_dir>/run_<timestamp>.csv (output_dir from "
                             "config/paths.yaml). The summary JSON is written next to it.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many frames (default: until source ends/ESC).")
    parser.add_argument("--engine", default=None,
                        help="TensorRT engine path. Default: 'engine' in config/paths.yaml.")
    return parser.parse_args(argv)


def default_output_csv(paths_cfg: dict) -> Path:
    stamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
    return resolve_path(paths_cfg["output_dir"]) / "run_{}.csv".format(stamp)


def summary_path_for(output_csv: Path) -> Path:
    return output_csv.with_name(output_csv.stem + SUMMARY_SUFFIX)


def write_summary(summary: dict, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(summary_path), "w") as summary_file:
        json.dump(summary, summary_file, indent=JSON_INDENT, sort_keys=True)


def print_summary(summary: dict) -> None:
    print("=== Session summary ===")
    for key in sorted(summary):
        print("{:<20} {}".format(key, summary[key]))


def run(args: argparse.Namespace) -> dict:
    # Heavy/GPU imports are deferred so --help works without TensorRT or a camera.
    from src.camera import Camera
    from src.detector_trt import ShrimpDetectorTRT
    from src.geometry import PlaneScale
    from src.pipeline import Pipeline

    paths_cfg = load_yaml("config/paths.yaml")
    detector_cfg = load_yaml("config/detector.yaml")
    output_csv = resolve_path(args.output) if args.output else default_output_csv(paths_cfg)
    engine_path = resolve_path(args.engine or paths_cfg["engine"])
    plane_scale = PlaneScale.from_config(load_yaml("config/calibration.yaml"))
    with Camera(load_yaml("config/camera.yaml"), source=args.source) as camera:
        detector = ShrimpDetectorTRT(str(engine_path), detector_cfg["conf_threshold"],
                                     detector_cfg["iou_threshold"], detector_cfg["num_classes"])
        pipeline = Pipeline(camera, detector, plane_scale, load_yaml("config/measurement.yaml"),
                            load_yaml("config/biomass.yaml"))
        summary = pipeline.run(headless=args.headless, output_csv=output_csv,
                               max_frames=args.max_frames)
    summary_path = summary_path_for(output_csv)
    write_summary(summary, summary_path)
    LOGGER.info("Summary written to %s", summary_path)
    return summary


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        summary = run(args)
    except (RuntimeError, OSError, ImportError, KeyError) as error:
        LOGGER.error("Run failed: %s", error)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
