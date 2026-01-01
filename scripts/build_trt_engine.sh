
#!/bin/bash
ONNX=models/yolo/shrimp.onnx
ENGINE=models/yolo/shrimp_fp16.trt

/usr/src/tensorrt/bin/trtexec \
  --onnx=$ONNX \
  --saveEngine=$ENGINE \
  --fp16 \
  --workspace=1024
