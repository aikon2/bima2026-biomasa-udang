# Configuration reference

All configuration lives in `config/*.yaml`. Relative paths **inside these files** (and the
built-in defaults) are resolved against the **repository root**. Relative paths **typed on the
command line** (`--source <video>`, `--output`, `--engine`, `--weights`, `--csv`,
`--calibration-csv`, `--points`, `--out`, `--json-out`) are resolved against the **current
directory**, like any normal CLI tool.

## `config/paths.yaml`

| Key | Default | Meaning |
|-----|---------|---------|
| `weights_pt` | `models/yolo/shrimp.pt` | Trained YOLOv8 weights; default `--weights` of `export_to_onnx.py`. |
| `onnx` | `models/yolo/shrimp.onnx` | Expected ONNX location. Ultralytics writes the ONNX next to the `.pt`; the exporter warns if it differs from this key. `build_trt_engine.sh` uses the same path as its own default. |
| `engine` | `models/yolo/shrimp_fp16.trt` | TensorRT engine loaded by `run_trt.py` (override with `--engine`). |
| `output_dir` | `outputs` | Directory for `run_<YYYYmmdd_HHMMSS>.csv` and its `_summary.json` when `--output` is not given. |

## `config/detector.yaml`

| Key | Default | Meaning |
|-----|---------|---------|
| `conf_threshold` | `0.4` | Minimum class score; candidates with score `>` this value are kept. |
| `iou_threshold` | `0.45` | Class-agnostic greedy NMS IoU threshold. |
| `num_classes` | `1` | Number of classes the model was trained with (output rows = 4 + `num_classes`). |
| `imgsz` | `640` | Square input size used by `export_to_onnx.py` (`--imgsz` default). At runtime the input size is read from the engine binding, not from this key. |

## `config/camera.yaml`

| Key | Default | Meaning |
|-----|---------|---------|
| `device` | `0` | Source used when `run_trt.py --source` is omitted. An integer is a V4L2/USB device index; the string `csi` or a video path also work. |
| `width` | `640` | Capture width. Applied to device-index and CSI sources; ignored for video files. |
| `height` | `360` | Capture height (same scope as `width`). |
| `fps` | `15` | Capture frame rate (same scope as `width`). |
| `csi.sensor_id` | `0` | `nvarguscamerasrc sensor-id` for the Jetson CSI camera. |
| `csi.flip_method` | `0` | `nvvidconv flip-method` (0 = none, 2 = rotate 180°, see NVIDIA docs). |

## `config/calibration.yaml`

| Key | Default | Meaning |
|-----|---------|---------|
| `cm_per_pixel` | `0.035` | Scalar scale on the measurement plane. Used only when `homography` is `null`. Required (also as fallback). |
| `homography` | `null` | Optional 3×3 matrix mapping image pixels to plane centimetres. Written by `calibration/calibrate_homography.py`, which keeps the other keys. |

Scalar example: a 10 cm marker spanning 285 px gives `10 / 285 ≈ 0.035` cm/px.

## `config/measurement.yaml`

| Key | Default | Meaning |
|-----|---------|---------|
| `method` | `skeleton` | `skeleton`: length along the longest path of the body skeleton (handles bent shrimp). `min_area_rect`: legacy long side of the minimum-area rectangle (straight shrimp only). Any other value raises an error. |
| `edge_margin_px` | `4` | Boxes closer than this to the frame border get status `edge` and are not measured (truncated shrimp). |
| `min_roi_px` | `8` | Boxes whose clipped width or height is below this get status `empty_roi`. |
| `morph_kernel` | `3` | Elliptical kernel size for morphological open + close after Otsu. `<= 1` disables it. |

## `config/biomass.yaml`

Written by `calibration/calibrate_ab.py`. Only `a`, `b` and the optional `pond` section are read
at runtime; the other keys document the fit.

| Key | Default | Meaning |
|-----|---------|---------|
| `a` | `0.012` (placeholder) | Bias-corrected coefficient of `W = a·L^b` (`a = a_raw · cf`). W in g, L in cm. |
| `b` | `3.1` (placeholder) | Allometric exponent. |
| `a_raw` | written by fit | Back-transformed intercept before bias correction. |
| `cf` | written by fit | Sprugel (1983) correction factor `exp(SEE²/2)`. |
| `r2` | written by fit | R² of the log–log regression. |
| `n` | written by fit | Number of valid samples used. |
| `pond.stocked_pcs` | absent | Optional. Number of post-larvae stocked. |
| `pond.survival_rate` | absent | Optional. Estimated survival as a fraction (e.g. `0.8`). |

The `pond` section is optional and is preserved when `calibrate_ab.py` rewrites the file. When
present, the session summary reports `pond_biomass_kg = ABW · stocked_pcs · survival_rate / 1000`.

```yaml
a: 0.0123
b: 3.05
pond:
  stocked_pcs: 100000
  survival_rate: 0.8
```
