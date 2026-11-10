# Spec: Review Fixes — Shrimp Length & Biomass Estimation (Jetson Nano)

- Date: 2026-11-10
- Status: Approved for execution
- Source: full code review of commit `416827a`

## 1. Goal

Make the pipeline (camera → TensorRT YOLOv8 → ROI measurement → pixel→cm → W = a·L^b → biomass)
**run end-to-end correctly on Jetson Nano**, produce **scientifically defensible** numbers,
and be **documented in English** so a third party can reproduce it.

## 2. Hard constraints

| ID | Constraint |
|----|-----------|
| C1 | Runtime code under `src/`, `run_trt.py`, `calibration/`, `evaluation/` must run on **Python 3.6** (JetPack 4.6 default). No dataclasses, no walrus, no `from __future__ import annotations`, no `typing.Literal`/`Protocol`, no f-string `=` specifier, no positional-only params. f-strings are allowed. |
| C2 | `tensorrt` and `pycuda` are imported **only** inside `src/detector_trt.py`. All pre/post-processing (letterbox, decode, NMS, rescale) lives in pure-numpy modules testable without a GPU. |
| C3 | Tests run on a dev machine with `pytest` using only numpy, opencv-python-headless, pyyaml, pandas, scikit-learn. |
| C4 | No magic numbers: thresholds, sizes, paths live in `config/*.yaml` or named module constants. |
| C5 | All paths resolve relative to the repo root (`Path(__file__)`), never the current working directory. |
| C6 | Functions ≤ 50 lines, nesting ≤ 3, files ≤ 300 LOC. Code/comments in English. |
| C7 | Every behavior change gets a test written first (RED → GREEN). |
| C8 | Conventional commits. **No AI co-author trailer.** No push. |

## 3. Requirements

### P0 — Pipeline correctness

- **R1 Detector output decoding.** Support the ultralytics YOLOv8 ONNX output `(1, 4+nc, N)`
  (transpose to `(N, 4+nc)`; boxes are `cx, cy, w, h` in network-input pixels; class score = max
  over class columns, no objectness). Confidence threshold from config.
- **R2 NMS.** Class-agnostic NMS with IoU threshold from config (pure numpy).
- **R3 Letterbox + rescale.** Preprocess with aspect-preserving letterbox (pad value 114) to the
  engine input size, BGR→RGB, CHW, float32/255. Map boxes back to original frame coordinates
  and clip to frame bounds. Input size read from the engine binding shape.
- **R4 Safe ROI.** `length_cm` must never crash: clip box, return `None` for empty/too-small ROI.
- **R5 Path consistency.** One source of truth `config/paths.yaml` (or a `paths` section) for
  `.pt`, `.onnx`, `.trt`; build script, export script and `run_trt.py` all agree
  (`models/yolo/shrimp.{pt,onnx}`, `models/yolo/shrimp_fp16.trt`).
- **R6 Calibration script paths.** `calibrate_ab.py` reads/writes relative to repo root, never `/`.
- **R7 Shell script.** Shebang on line 1, `set -euo pipefail`, executable bit, args overridable.
- **R8 Dependencies.** Split `requirements-train.txt` (PC/GPU: ultralytics, torch, onnx) and
  `requirements-jetson.txt` (numpy, pyyaml, pycuda note; OpenCV/TensorRT come from JetPack — do
  NOT pip install opencv-python on Jetson) and `requirements-dev.txt` (pytest, headless OpenCV,
  pandas, scikit-learn). Keep `requirements.txt` as a pointer/comment or remove with README note.

### P1 — Measurement validity

- **R9 Perspective-correct scale.** Support an optional 3×3 **homography** (image px → plane cm)
  in `config/calibration.yaml`; length is computed by projecting the measured endpoints/skeleton
  to the plane. Fallback to scalar `cm_per_pixel` when no homography. Provide
  `calibration/calibrate_homography.py` that computes it from 4+ marker point pairs (CLI input or
  CSV of `u,v,x_cm,y_cm`).
- **R10 Edge-truncated shrimp.** Exclude detections whose box is within `edge_margin_px` of the
  frame border (configurable) — they are counted as "detected" but not "measured".
- **R11 Segmentation polarity.** Otsu threshold with automatic polarity: the object is the class
  that is **not** dominant on the ROI border. Morphological open/close with configurable kernel.
- **R12 Curved-body length.** Length method configurable: `min_area_rect` (legacy) or
  `skeleton` (default): longest path along a morphological skeleton (pure OpenCV/numpy, no
  scikit-image), measured in plane coordinates when a homography exists.
- **R13 Length–weight model.** `weight_g(length_cm, a, b)` validates input (None/≤0/NaN → None).
  Calibration fits log-log OLS, reports a, b, R², n, SE and 95% CI of b, and applies the
  **Sprugel correction factor** `CF = exp(SEE²/2)` to `a` (stored as `a`, raw value as `a_raw`,
  plus `cf`). Refuse to write config with n < `MIN_CALIBRATION_SAMPLES` (=10) unless `--force`;
  warn below 30. Drop blank/invalid rows.
- **R14 Farmer-relevant outputs.** Per session aggregate over measured shrimp: count detected,
  count measured, **ABW** (g), **size** (pcs/kg = 1000/ABW), **CV%** of weight, length
  mean/std, and an optional **pond biomass estimate** = ABW × stocked × survival_rate (from
  `config/biomass.yaml` `pond:` section, optional). Frame-sum biomass is shown but labeled
  "visible biomass (frame)". Document that without tracking, the same shrimp may be sampled in
  multiple frames (ABW/size remain valid as sample statistics; counts are not population counts).

### P1 — Research methodology

- **R15 Evaluation.** `evaluation/evaluate.py` with CLI: input CSV with `length_gt_cm,
  length_pred_cm, weight_gt_g, weight_pred_g` (columns optional per metric). Reports MAE, RMSE,
  MAPE, R², bias, n — for length and weight separately. Guard: division by zero, n < 2.
  Optional `--calibration-csv` to warn if evaluation rows duplicate calibration rows (leakage).
- **R16 Placeholder data.** Mark `calibration_data.csv` / `evaluation/data.csv` as examples
  (rename to `*.example.csv` or add header comment in README) and remove blank lines.

### P2 — Engineering

- **R17 Pipeline.** CLI `run_trt.py` with `--headless`, `--source` (device index / video file /
  `csi`), `--output` CSV path, `--max-frames`. Per-detection CSV log
  (`timestamp, frame_idx, x1,y1,x2,y2, conf, length_cm, weight_g, status`). FPS measurement.
  `try/finally` releases camera and windows. Summary (R14) printed and saved as JSON at end.
- **R18 Camera.** Use `cv2.CAP_PROP_*`, check `isOpened()` and raise a clear error, support CSI
  via GStreamer `nvarguscamerasrc` pipeline string built from config, and video files.
- **R19 Repo hygiene.** `.gitignore` (models binaries, `.venv`, `__pycache__`, outputs, `.omc/`),
  `pytest.ini`/`tests/`, logging via `logging` module instead of print in library code.
- **R20 Export.** `export_to_onnx.py` with argparse (imgsz default from config, `simplify=True`,
  opset 12) — runs on PC, not Jetson.

### P3 — Documentation

- **R21 README in English**, complete: overview + architecture diagram (ASCII/mermaid),
  hardware (incl. waterproof housing, lighting/backscatter, heat, power, lens biofouling),
  software matrix (PC training vs Jetson runtime), step-by-step: dataset → train → export ONNX →
  build engine on Jetson → camera setup → scale/homography calibration (in water, on the
  measurement plane e.g. feeding tray) → L–W calibration (30–50 samples) → run → evaluate;
  config reference table; output format; metrics; limitations (no tracking, turbidity, depth
  variance); troubleshooting; project structure; testing; citation/license placeholder.

## 4. Out of scope

Multi-object tracking (ByteTrack/SORT), training script and dataset, YOLOv8-seg/keypoint model,
web dashboard. Listed as future work in README.

## 5. Acceptance

- `.venv/bin/python -m pytest -q` passes.
- `python3 -m py_compile` passes for all files; static grep finds no py3.7+ only syntax (C1).
- No file > 300 LOC; README in English.
- Git history: one commit per plan task, conventional message, no AI trailer.
