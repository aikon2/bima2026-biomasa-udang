# Plan: Review Fixes — Implementation

- Spec: [`docs/specs/2026-11-10-review-fixes.md`](../specs/2026-11-10-review-fixes.md) (requirement IDs R1–R21, constraints C1–C8)
- Executor: Sonnet subagents, sequential (tasks share files), one commit per task.
- Test command: `.venv/bin/python -m pytest -q` (dev venv already exists at `.venv`, Python 3.11,
  with numpy, opencv-python-headless, pyyaml, pytest, pandas, scikit-learn).
- Py3.6 check: `grep -rnE "dataclass|:=|from __future__ import annotations|Literal\[|Protocol" src calibration evaluation run_trt.py` must be empty.

## Shared interfaces (all tasks MUST follow)

```text
src/config.py
  REPO_ROOT: Path
  load_yaml(relative_path: str) -> dict          # relative to REPO_ROOT
  resolve_path(relative_path: str) -> Path

src/postprocess.py   (pure numpy, no cv2 needed except optional)
  Detection = namedtuple("Detection", "x1 y1 x2 y2 conf")   # ints for coords, float conf
  decode_yolov8(output, num_classes, conf_threshold) -> (boxes_xyxy[N,4] float, scores[N])
  nms(boxes_xyxy, scores, iou_threshold) -> np.ndarray[int] keep indices (score-desc order)
  scale_boxes(boxes_xyxy, scale, pad_xy, frame_shape) -> boxes int, clipped to frame

src/preprocess.py
  LETTERBOX_PAD_VALUE = 114
  letterbox(frame_bgr, input_hw) -> (blob[1,3,H,W] float32 contiguous RGB/255, scale, (pad_x, pad_y))

src/detector_trt.py  (only file importing tensorrt/pycuda)
  ShrimpDetectorTRT(engine_path, conf_threshold, iou_threshold, num_classes)
  .detect(frame_bgr) -> list[Detection]

src/geometry.py
  PlaneScale.from_config(calibration_cfg: dict) -> PlaneScale
  PlaneScale.to_plane(points_px[N,2]) -> points_cm[N,2]
  PlaneScale.polyline_length_cm(points_px[N,2]) -> float
  compute_homography(points_px, points_cm) -> 3x3 list   # cv2.findHomography, >=4 pairs

src/measurement.py
  MeasureResult = namedtuple("MeasureResult", "length_cm status")
  STATUS_OK, STATUS_EDGE, STATUS_EMPTY_ROI, STATUS_NO_CONTOUR  (string constants)
  measure_length(frame_bgr, detection, plane_scale, measurement_cfg) -> MeasureResult

src/biomass.py
  weight_g(length_cm, a, b) -> Optional[float]
  summarize_session(lengths_cm, weights_g, n_detected, pond_cfg=None) -> dict
    keys: n_detected, n_measured, abw_g, size_pcs_per_kg, cv_weight_pct,
          length_mean_cm, length_std_cm, sampled_biomass_g, pond_biomass_kg (None if no pond cfg)

src/camera.py
  Camera(cfg: dict, source: Optional[str]=None)   # source: "0", "csi", or video path
  build_csi_pipeline(cfg) -> str
  .read() -> (ok, frame); .release(); context manager support

src/pipeline.py
  Pipeline(camera, detector, plane_scale, measurement_cfg, biomass_cfg)
  .run(headless: bool, output_csv: Optional[Path], max_frames: Optional[int]) -> dict summary
```

## Config files (target state)

| File | Keys |
|------|------|
| `config/paths.yaml` | `weights_pt: models/yolo/shrimp.pt`, `onnx: models/yolo/shrimp.onnx`, `engine: models/yolo/shrimp_fp16.trt`, `output_dir: outputs` |
| `config/detector.yaml` | `conf_threshold: 0.4`, `iou_threshold: 0.45`, `num_classes: 1`, `imgsz: 640` |
| `config/camera.yaml` | `device: 0`, `width`, `height`, `fps`, `csi: {sensor_id: 0, flip_method: 0}` |
| `config/calibration.yaml` | `cm_per_pixel: 0.035`, `homography: null` |
| `config/measurement.yaml` | `method: skeleton`, `edge_margin_px: 4`, `min_roi_px: 8`, `morph_kernel: 3` |
| `config/biomass.yaml` | `a`, `b` (+ `a_raw`, `cf`, `n`, `r2` written by calibration), optional `pond: {stocked_pcs, survival_rate}` |

---

## Task 1 — `chore: repo hygiene, dependency split, consistent paths` (R5, R7, R8, R19, R20)

Files: `.gitignore`, `requirements*.txt`, `pytest.ini`, `tests/__init__.py`, `tests/conftest.py`,
`src/__init__.py`, `src/config.py`, `config/paths.yaml`, `config/detector.yaml`,
`scripts/build_trt_engine.sh`, `scripts/export_to_onnx.py`, `tests/test_config.py`.

1. RED: `tests/test_config.py` — `REPO_ROOT` contains `config/`; `load_yaml("config/paths.yaml")`
   returns engine path ending `shrimp_fp16.trt`; `resolve_path` returns absolute path.
2. GREEN: implement `src/config.py`, add yaml files.
3. `.gitignore`: `.venv/`, `__pycache__/`, `*.pyc`, `models/**/*.pt`, `*.onnx`, `*.trt`, `*.engine`,
   `outputs/`, `.omc/`, `.pytest_cache/`. Keep `models/yolo/.gitkeep`.
4. Requirements split per R8; `requirements.txt` becomes a comment pointing at the three files.
5. `build_trt_engine.sh`: shebang line 1, `set -euo pipefail`, `ONNX=${1:-models/yolo/shrimp.onnx}`,
   `ENGINE=${2:-models/yolo/shrimp_fp16.trt}`, `cd` to repo root, check `trtexec` exists and ONNX
   exists with clear messages, `chmod +x` (use `git update-index --chmod=+x`).
6. `export_to_onnx.py`: argparse (`--weights`, `--imgsz` default from detector.yaml, `--opset 12`),
   `simplify=True`, paths via `src.config`.
7. Verify tests, commit.

## Task 2 — `fix(detector): decode YOLOv8 output, add NMS, letterbox and rescale` (R1, R2, R3)

Files: `src/preprocess.py`, `src/postprocess.py`, `src/detector_trt.py`,
`tests/test_preprocess.py`, `tests/test_postprocess.py`.

1. RED tests:
   - letterbox of 360×640 frame to 640×640 → blob shape `(1,3,640,640)`, dtype float32,
     scale 1.0, pad `(0,140)`; pad pixels == 114/255; channel order RGB (pure blue BGR pixel →
     channel 2 high).
   - decode: synthetic `(1,5,8400)` array with 2 boxes above threshold → 2 results, correct xyxy.
   - decode multi-class `(1,6,N)` picks max class score.
   - nms: two overlapping boxes (IoU>0.45) → keeps higher score; disjoint boxes → both kept.
   - scale_boxes: inverse of letterbox maps back and clips negatives/overflow to frame.
2. GREEN implement modules.
3. Refactor `detector_trt.py`: read input shape from engine binding (TRT 8.2 API
   `get_binding_shape`, keep it), find output binding by name/not-input, reshape output to binding
   shape, use preprocess/postprocess, return `list[Detection]`. Context: lazy
   `import pycuda.autoinit` stays. Constructor takes thresholds (from detector.yaml).
4. Verify, commit.

## Task 3 — `fix(measurement): safe ROI, edge filtering, polarity-aware segmentation, skeleton length` (R4, R10, R11, R12, R9-runtime)

Files: `src/geometry.py`, `src/measurement.py`, `config/measurement.yaml`,
`config/calibration.yaml`, `tests/test_geometry.py`, `tests/test_measurement.py`.

1. RED tests (synthetic images drawn with cv2):
   - `PlaneScale` scalar: 100 px horizontal line → `100*cm_per_pixel` cm.
   - `PlaneScale` homography: identity×0.1 scale homography → same result as scalar 0.1;
     `compute_homography` recovers known homography from 4 points.
   - Box with negative/out-of-range coords → no exception; empty → `STATUS_EMPTY_ROI`.
   - Box touching border within margin → `STATUS_EDGE`, length None.
   - Bright straight bar on dark bg AND dark bar on bright bg → both give length within 10% of bar
     length (polarity).
   - Curved arc (thick polyline, half circle) → `skeleton` length within 15% of true arc length and
     larger than `min_area_rect` length.
2. GREEN: skeleton via iterative morphological thinning (`cv2.erode/dilate/subtract` loop),
   longest path via two BFS passes over skeleton pixels (8-connectivity), endpoints-to-path points
   in **frame** coordinates, then `plane_scale.polyline_length_cm`. Keep functions ≤50 lines.
3. Verify, commit.

## Task 4 — `feat(calibration): robust L–W fit with bias correction, homography tool, validated weight` (R6, R9-tool, R13)

Files: `calibration/__init__.py`, `calibration/calibrate_ab.py`, `calibration/calibrate_homography.py`,
`calibration/calibration_data.example.csv`, remove old `calibration_data.csv` (git mv),
`src/biomass.py`, `tests/test_biomass.py`, `tests/test_calibrate_ab.py`.

1. RED tests:
   - `weight_g(10, 0.01, 3.0) == 10.0`; None / 0 / negative / NaN → None.
   - `fit_length_weight(df)` on synthetic data from `W = 0.008·L^3.05` with multiplicative noise
     (seeded, n=40) → b within 0.1, returns keys `a, a_raw, b, cf, r2, n, b_se, b_ci95`, `cf >= 1`.
   - blank/non-positive rows dropped; n < 10 raises `ValueError` unless `force=True`.
   - `summarize_session([...])` ABW, size = 1000/ABW, CV%, pond biomass when pond cfg given,
     empty input → zeros/None without exception.
2. GREEN. `calibrate_ab.py`: argparse `--csv` (default example file), `--out config/biomass.yaml`,
   `--force`, `--dry-run`; preserves existing `pond:` section when rewriting yaml. Use numpy
   polyfit (no hard sklearn dependency so it runs on Jetson).
   `calibrate_homography.py`: `--points CSV(u,v,x_cm,y_cm)` → writes `homography` into
   `config/calibration.yaml`, prints reprojection error.
3. Example CSV: header + 2 example rows, no blank lines, comment in README that ≥30 real samples
   are required.
4. Verify, commit.

## Task 5 — `feat(pipeline): headless CLI, CSV logging, farmer metrics, robust camera` (R14, R17, R18)

Files: `src/camera.py`, `src/pipeline.py`, `run_trt.py`, `config/camera.yaml`,
`tests/test_camera.py`, `tests/test_pipeline.py`.

1. RED tests:
   - `build_csi_pipeline(cfg)` contains `nvarguscamerasrc sensor-id=0`, width/height/fps, ends with `appsink`.
   - `Camera` with nonexistent video path raises `RuntimeError` with clear message.
   - Pipeline with fake camera (yields 3 synthetic frames with a bright bar) + fake detector
     (returns fixed Detection) runs headless, writes CSV with header
     `timestamp,frame_idx,x1,y1,x2,y2,conf,length_cm,weight_g,status` and 3 rows, returns summary
     with `n_measured == 3`, `fps` key, and camera `release()` called even if detector raises.
2. GREEN. Pipeline draws overlay only when not headless; overlay shows count detected/measured,
   ABW, size. `run_trt.py`: argparse `--source`, `--headless`, `--output` (default
   `outputs/run_<timestamp>.csv`), `--max-frames`; saves summary JSON next to CSV; lazy-imports
   `detector_trt` so `--help` works without TensorRT. Use `logging`.
3. Verify, commit.

## Task 6 — `feat(evaluation): length and weight metrics with leakage check` (R15, R16)

Files: `evaluation/__init__.py`, `evaluation/evaluate.py`, `evaluation/data.example.csv`
(git mv from `data.csv`, new columns), `tests/test_evaluate.py`.

1. RED tests: perfect predictions → MAE 0, R² 1; known small set → hand-computed MAE/RMSE/MAPE/bias;
   zero ground-truth row excluded from MAPE; n<2 → R² None; leakage detection flags identical rows.
2. GREEN with argparse `--csv`, `--calibration-csv`, `--json-out`.
3. Verify, commit.

## Task 7 — `docs: rewrite README in English with full reproduction guide` (R21)

File: `README.md` (may add `docs/architecture.md` only if README > 300 lines).
Must document every CLI, config key table, outputs, metrics, field-deployment guidance
(measurement plane on feeding tray, calibrate in water, turbidity/backscatter, housing, heat,
power, lens cleaning), limitations & future work, testing, structure, citation placeholder.
Commands must match the actual code (read `--help` of every script). Commit.

## Final verification (lead)

pytest green, py3.6 grep clean, `wc -l` ≤300 per file, `git log` = 1 docs commit + 7 task commits.
