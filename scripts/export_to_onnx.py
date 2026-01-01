
from ultralytics import YOLO

model = YOLO("models/yolo/shrimp.pt")
model.export(format="onnx", imgsz=640, opset=12)
