# ML Training Scripts

This directory contains scripts for collecting training data and training the sign recognition model.

## 📋 Requirements

### For Data Collection (Python 3.11 required)

MediaPipe Python doesn't support Python 3.14 yet, so data collection requires Python 3.11:

```powershell
# Use Python 3.11 for data collection
py -3.11 -m pip install opencv-python mediapipe numpy
```

### For Training (Python 3.14 works!)

Training only needs PyTorch and NumPy:

```powershell
# Works with Python 3.14
py -m pip install torch numpy
```

## 🎬 Data Collection

```powershell
# Use Python 3.11 (MediaPipe requirement)
py -3.11 collect_data.py --output_dir data --label hello --num_sequences 20
```

## 🧠 Training

### PyTorch Version (Python 3.14+)

```powershell
# Works with Python 3.14
py train_model_pytorch.py --data_dir data --output_model ../backend/model/sign_model.pt
```

### TensorFlow Version (Python 3.8-3.11)

```powershell
# Requires Python 3.11 or earlier
py -3.11 train_model.py --data_dir data --output_model ../backend/model/sign_model.h5
```

## 📝 Summary

| Task | Python Version | Dependencies |
|------|---------------|---------------|
| **Data Collection** | 3.11 required | opencv-python, mediapipe, numpy |
| **Training (PyTorch)** | 3.14+ ✅ | torch, numpy |
| **Training (TensorFlow)** | 3.8-3.11 | tensorflow, numpy |



