# 🎯 SignSpeak - Complete Setup & Training Guide

This guide walks you through **collecting data**, **training the model**, and **running the entire system**.

---

## 📋 Prerequisites

- **Python 3.8 - 3.11** ⚠️ **REQUIRED** (TensorFlow doesn't support Python 3.12+)
  - Check version: `py --version`
  - If you have Python 3.12+, install Python 3.11: https://www.python.org/downloads/
  - See [PYTHON_VERSION_NOTE.md](./PYTHON_VERSION_NOTE.md) for details
- **Node.js 16+** and npm
- **Webcam** connected to your computer
- **Git** (optional, for cloning)

---

## 🔧 Step 1: Backend Setup

### 1.1 Install Python Dependencies

**⚠️ IMPORTANT:** Make sure you're using Python 3.8-3.11!

```powershell
# Check Python version
py --version

# If you have Python 3.11, use it specifically:
cd backend
py -3.11 -m pip install -r requirements.txt

# Or if Python 3.11 is your default:
py -m pip install -r requirements.txt
```

**Note:** If you encounter issues with TensorFlow:
- Make sure Python version is 3.8-3.11 (not 3.12+)
- Requirements.txt uses `tensorflow-cpu` which works on all systems

### 1.2 Verify Installation

```bash
python -c "import tensorflow; import fastapi; print('✅ Backend dependencies installed')"
```

---

## 🎬 Step 2: Collect Training Data

### 2.1 Navigate to Training Directory

```bash
cd ml_training
```

### 2.2 Collect Data for Each Sign

For each sign you want to recognize, run the collection script:

**Example: Collect 20 sequences for "hello" sign**
```bash
python collect_data.py --output_dir data --label hello --num_sequences 20
```

**Example: Collect 20 sequences for "thanks" sign**
```bash
python collect_data.py --output_dir data --label thanks --num_sequences 20
```

**Example: Collect 20 sequences for "yes" sign**
```bash
python collect_data.py --output_dir data --label yes --num_sequences 20
```

### 2.3 How Data Collection Works

1. **Script starts** → Opens your webcam
2. **2-second countdown** → Get ready to perform the sign
3. **Recording** → Performs the sign for ~30 frames (1 second)
4. **Auto-saves** → Saves as `data/<label>_000.npy`, `data/<label>_001.npy`, etc.
5. **Repeats** → Does this `num_sequences` times

**Tips:**
- Perform the sign **clearly** and **consistently**
- Keep your hand **visible** in the camera frame
- **More data = better accuracy** (aim for 15-30 sequences per sign minimum)
- Press `q` to quit early if needed

### 2.4 Verify Collected Data

After collecting, check your `ml_training/data/` folder:
```
ml_training/data/
  ├── hello_000.npy
  ├── hello_001.npy
  ├── hello_002.npy
  ├── thanks_000.npy
  ├── thanks_001.npy
  └── ...
```

---

## 🧠 Step 3: Train the Model

### 3.1 Run Training Script

```bash
python train_model.py --data_dir data --output_model ../backend/model/sign_model.h5 --epochs 25
```

**Parameters:**
- `--data_dir`: Directory with your `.npy` files (default: `data`)
- `--output_model`: Where to save the trained model (default: `../backend/model/sign_model.h5`)
- `--epochs`: Number of training epochs (default: 25)
- `--batch_size`: Batch size (default: 16)

### 3.2 Training Output

You'll see output like:
```
Loaded 60 sequences, 3 classes
Classes: ['hello', 'thanks', 'yes']
Epoch 1/25
...
Model saved to ../backend/model/sign_model.h5
```

### 3.3 Verify Model Created

Check that the model file exists:
```bash
ls ../backend/model/sign_model.h5
```

---

## 🚀 Step 4: Run the Backend Server

### 4.1 Start FastAPI Server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 4.2 Test Backend (Optional)

Open browser: http://localhost:8000

You should see: `{"status":"ok","message":"SignSpeak backend is running"}`

**Keep this terminal running!** The backend must stay active.

---

## 🎨 Step 5: Run the Frontend

### 5.1 Install Frontend Dependencies

Open a **new terminal** (keep backend running):

```bash
cd frontend
npm install
```

### 5.2 Start Development Server

```bash
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### 5.3 Open in Browser

Navigate to: **http://localhost:5173**

---

## ✅ Step 6: Test the System

### 6.1 Allow Camera Access

When you open the frontend, your browser will ask for camera permission. **Click "Allow"**.

### 6.2 Perform Signs

1. **Show your hand** to the webcam
2. **Perform a sign** you trained (e.g., "hello")
3. **Watch the prediction** appear in real-time:
   - Prediction text (e.g., "hello")
   - Confidence score (e.g., 0.95)

### 6.3 Test Speech

1. Click **"Speak Prediction"** button
2. You should hear the text spoken aloud

---

## 🔍 Troubleshooting

### Problem: "Model not found" error

**Solution:** Make sure you've trained the model:
```bash
cd ml_training
python train_model.py --data_dir data
```

### Problem: Camera not working

**Solution:** 
- Check camera permissions in browser
- Make sure no other app is using the camera
- Try refreshing the page

### Problem: CORS errors

**Solution:** Backend CORS is already configured. Make sure backend is running on port 8000.

### Problem: Low accuracy predictions

**Solution:**
- Collect **more training data** (30+ sequences per sign)
- Ensure consistent sign performance during collection
- Try training with more epochs: `--epochs 50`

### Problem: TensorFlow/GPU issues

**Solution:** Install CPU-only version:
```bash
pip uninstall tensorflow
pip install tensorflow-cpu
```

---

## 📊 Quick Reference Commands

### Complete Workflow (Copy-Paste Ready)

```bash
# 1. Setup backend
cd backend
pip install -r requirements.txt

# 2. Collect data (repeat for each sign)
cd ../ml_training
python collect_data.py --output_dir data --label hello --num_sequences 20
python collect_data.py --output_dir data --label thanks --num_sequences 20

# 3. Train model
python train_model.py --data_dir data --output_model ../backend/model/sign_model.h5

# 4. Run backend (Terminal 1)
cd ../backend
uvicorn main:app --reload --port 8000

# 5. Run frontend (Terminal 2)
cd ../frontend
npm install
npm run dev
```

---

## 🎓 Training Tips

1. **More Data = Better**: Aim for 20-30 sequences per sign minimum
2. **Consistency**: Perform signs the same way each time
3. **Lighting**: Good lighting helps MediaPipe detect hands better
4. **Background**: Plain backgrounds work best
5. **Multiple Angles**: Collect data from slightly different angles for robustness

---

## 📝 Next Steps

- Add more signs by collecting more data
- Fine-tune model hyperparameters in `train_model.py`
- Customize UI in `frontend/src/App.jsx`
- Deploy backend to cloud (Heroku, Railway, etc.)
- Deploy frontend to Vercel/Netlify

---

**Happy Signing! 🎉**

