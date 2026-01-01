# Riset BIMA 2026 - Jurnal proses submit - Estimasi Panjang & Biomassa Udang Vaname Untuk kamera bawah air, TensorRT FP16 dioptimasi untuk Jetson Nano

---

## 1. System Overview

Alur:
→ Kamera
→ TensorRT YOLO (deteksi udang)
→ ROI crop
→ Segmentasi ringan (threshold + contour)
→ Panjang tubuh (longest axis)
→ Kalibrasi pixel → cm
→ Model regresi udang W = a·L^b
→ Estimasi biomassa udang

---

## 2. Hardware Requirements

- NVIDIA Jetson Nano (4GB)
- Kamera USB atau CSI
- Marker kalibrasi (10 cm)

---

## 3. Software Requirements

- JetPack 4.6
- Python 3.8
- TensorRT

---

## 4. Installation

- camera.yml = Kemiringan kamera 30-45 derajat
- Pencahayaan dengan led putih
- calibration.yml = siapkan marker PVC 10cm dan letakkan di dasar, lalu hitung cm per pixel
  marker 10cm = 285 px => 10/285 = 0.035
- export_to_onnx.py = training YOLO nano lalu export ke ONNX
- calibrate_ab.py = ambil sample 30-50 ukur manual dan timbang lalu masukkan update file calibration_data.csv

```bash
pip3 install -r requirements.txt
```

---

## 5. Configuration

`config/camera.yaml`
`config/calibration.yaml`
`config/biomass.yaml` => diisi dari calibrate_ab.py

---

## 6. Inference (TensorRT FP16)

```bash
python3 scripts/export_to_onnx.py
python3 run_trt.py
```

Output:

- menghitung jumlah udang yang terdeteksi
- estimasi panjang (cm)
- estimasi berat (g)
- Total biomassa (g)

---

## 8. Evaluation

```bash
python3 evaluation/evaluate.py
```

untuk mengukur metrik:

- MAE
- MAPE
- R²
