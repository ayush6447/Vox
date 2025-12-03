# ✅ Using SignSpeak with Python 3.14

Great news! You can use **Python 3.14** with SignSpeak by using the **PyTorch** version instead of TensorFlow.

**Note:** MediaPipe (used for data collection) doesn't support Python 3.14 yet. You have two options:
1. **Collect data with Python 3.11**, then train/run backend with Python 3.14
2. **Use the frontend** (which uses MediaPipe JS) to collect data instead

## 🚀 Quick Setup for Python 3.14

### Step 1: Install PyTorch Dependencies

```powershell
cd backend
py -m pip install -r requirements_pytorch.txt
```

Or install manually:
```powershell
py -m pip install fastapi uvicorn torch numpy gtts python-multipart mediapipe opencv-python
```

### Step 2: Install Training Dependencies

```powershell
cd ml_training
py -m pip install torch numpy
```

**Note:** MediaPipe is NOT needed for training (only for data collection). The PyTorch training script only needs `torch` and `numpy`, both of which work with Python 3.14.

### Step 3: Collect Training Data

**⚠️ Important:** MediaPipe Python doesn't support Python 3.14 yet. You have two options:

**Option A: Use Python 3.11 for data collection** (recommended)
```powershell
# Use Python 3.11 specifically for data collection
py -3.11 collect_data.py --output_dir data --label hello --num_sequences 20
py -3.11 collect_data.py --output_dir data --label thanks --num_sequences 20
```

**Option B: Use frontend to collect data** (alternative)
- The frontend uses MediaPipe JS which works fine
- You can modify the frontend to save landmark sequences

### Step 4: Train Model (PyTorch Version)

```powershell
py train_model_pytorch.py --data_dir data --output_model ../backend/model/sign_model.pt
```

**Note:** PyTorch saves models as `.pt` files, not `.h5` files.

### Step 5: Start Backend (PyTorch Version)

```powershell
cd backend
py main_pytorch.py
```

Or with uvicorn:
```powershell
py -m uvicorn main_pytorch:app --reload --port 8000
```

### Step 6: Start Frontend

```powershell
cd frontend
npm install
npm run dev
```

---

## 📝 Key Differences: PyTorch vs TensorFlow

| Feature | TensorFlow Version | PyTorch Version |
|---------|-------------------|-----------------|
| **Python Support** | 3.8-3.11 only | 3.8-3.14+ ✅ |
| **Model File** | `sign_model.h5` | `sign_model.pt` |
| **Training Script** | `train_model.py` | `train_model_pytorch.py` |
| **Backend File** | `main.py` | `main_pytorch.py` |
| **Dependencies** | `requirements.txt` | `requirements_pytorch.txt` |

---

## 🔄 Switching Between Versions

### To Use PyTorch (Python 3.14):
1. Use `train_model_pytorch.py` for training
2. Use `main_pytorch.py` for backend
3. Model will be saved as `sign_model.pt`

### To Use TensorFlow (Python 3.11):
1. Use `train_model.py` for training
2. Use `main.py` for backend
3. Model will be saved as `sign_model.h5`

---

## ✅ Advantages of PyTorch Version

- ✅ **Works with Python 3.14** (and all newer versions)
- ✅ **Better GPU support** (if you have CUDA)
- ✅ **More modern API** (easier to debug)
- ✅ **Same accuracy** (LSTM architecture is identical)

---

## 🎯 Complete Workflow (Python 3.14)

```powershell
# 1. Setup backend
cd backend
py -m pip install -r requirements_pytorch.txt

# 2. Setup training
cd ../ml_training
py -m pip install -r requirements_pytorch.txt

# 3. Collect data
py collect_data.py --output_dir data --label hello --num_sequences 20
py collect_data.py --output_dir data --label thanks --num_sequences 20

# 4. Train model
py train_model_pytorch.py --data_dir data

# 5. Start backend (Terminal 1)
cd ../backend
py -m uvicorn main_pytorch:app --reload --port 8000

# 6. Start frontend (Terminal 2)
cd ../frontend
npm install
npm run dev
```

---

**That's it!** You can now use SignSpeak with Python 3.14! 🎉

