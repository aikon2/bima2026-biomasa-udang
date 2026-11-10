"""Export trained YOLOv8 weights to ONNX. Runs on the PC/GPU workstation, not on Jetson."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_yaml, resolve_cli_path, resolve_path  # noqa: E402

DEFAULT_OPSET = 12
LOGGER = logging.getLogger("export_to_onnx")


def parse_args() -> argparse.Namespace:
    paths_cfg = load_yaml("config/paths.yaml")
    detector_cfg = load_yaml("config/detector.yaml")
    parser = argparse.ArgumentParser(description="Export YOLOv8 .pt weights to ONNX.")
    parser.add_argument("--weights", type=resolve_cli_path,
                        default=str(resolve_path(paths_cfg["weights_pt"])),
                        help="Path to .pt weights (relative to the current directory).")
    parser.add_argument("--imgsz", type=int, default=detector_cfg["imgsz"],
                        help="Square network input size in pixels.")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET,
                        help="ONNX opset version (12 is compatible with TensorRT 8.2).")
    return parser.parse_args()


def export_onnx(weights_path: Path, imgsz: int, opset: int) -> Path:
    if not weights_path.is_file():
        raise FileNotFoundError("Weights not found: {}".format(weights_path))
    from ultralytics import YOLO  # imported lazily so --help works without ultralytics

    model = YOLO(str(weights_path))
    exported = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True)
    return Path(str(exported))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    weights_path = resolve_path(args.weights)
    try:
        onnx_path = export_onnx(weights_path, args.imgsz, args.opset)
    except FileNotFoundError as error:
        LOGGER.error("Export failed: %s", error)
        return 1
    expected_onnx = resolve_path(load_yaml("config/paths.yaml")["onnx"])
    LOGGER.info("Exported ONNX: %s", onnx_path)
    if onnx_path.resolve() != expected_onnx.resolve():
        LOGGER.warning("ONNX written to %s but config/paths.yaml expects %s",
                       onnx_path, expected_onnx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
