#!/usr/bin/env bash
# Build an FP16 TensorRT engine from the YOLOv8 ONNX model. Run on the Jetson.
# Usage: scripts/build_trt_engine.sh [ONNX_PATH] [ENGINE_PATH]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ONNX="${1:-models/yolo/shrimp.onnx}"
ENGINE="${2:-models/yolo/shrimp_fp16.trt}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
WORKSPACE_MB="${WORKSPACE_MB:-1024}"

if [ ! -x "${TRTEXEC}" ] && ! command -v "${TRTEXEC}" >/dev/null 2>&1; then
  echo "ERROR: trtexec not found at '${TRTEXEC}'. Install TensorRT (JetPack) or set TRTEXEC." >&2
  exit 1
fi

if [ ! -f "${ONNX}" ]; then
  echo "ERROR: ONNX model not found: '${ONNX}'. Export it first with scripts/export_to_onnx.py." >&2
  exit 1
fi

mkdir -p "$(dirname "${ENGINE}")"
echo "Building engine: ${ONNX} -> ${ENGINE} (fp16, workspace ${WORKSPACE_MB} MB)"

"${TRTEXEC}" \
  --onnx="${ONNX}" \
  --saveEngine="${ENGINE}" \
  --fp16 \
  --workspace="${WORKSPACE_MB}"

echo "Engine written to ${ENGINE}"
