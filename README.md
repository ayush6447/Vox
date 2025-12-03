# 🎯 SignSpeak — Real-time Sign Language to Text/Speech

SignSpeak is an end-to-end system that translates sign language gestures into text and speech in real-time.

## ✨ Features

- 🎥 **Real-time webcam capture** using WebRTC
- 🤲 **Hand tracking** with MediaPipe Hands (21 landmarks)
- 🧠 **Sign recognition** using TensorFlow LSTM model
- 🔊 **Text-to-speech** conversion with gTTS
- ⚡ **Low latency** predictions (~200ms updates)

## 📂 Project Structure

```
SignSpeak/
├── frontend/          # React + Vite web app
│   ├── src/
│   │   ├── components/Webcam.jsx
│   │   ├── mediapipe/handTracker.js
│   │   └── api/api.js
├── backend/           # FastAPI server
│   ├── main.py
│   ├── model/sign_model.h5
│   └── utils/
├── ml_training/       # Data collection & training
│   ├── collect_data.py
│   └── train_model.py
```

## 🚀 Quick Start

**See [QUICK_START.md](./QUICK_START.md) for a 5-minute setup guide.**

**See [SETUP_GUIDE.md](./SETUP_GUIDE.md) for detailed instructions.**

### Basic Workflow

1. **Install dependencies** (backend + frontend)
2. **Collect training data** for your signs
3. **Train the model**
4. **Start backend server** (port 8000)
5. **Start frontend dev server** (port 5173)
6. **Open browser** and start signing!

## 📖 Documentation

- **[SETUP_GUIDE.md](./SETUP_GUIDE.md)** - Complete step-by-step guide
- **[QUICK_START.md](./QUICK_START.md)** - Fast setup instructions
- **[PYTHON_314_GUIDE.md](./PYTHON_314_GUIDE.md)** - Using Python 3.14 with PyTorch

## 🛠️ Tech Stack

- **Frontend:** React, Vite, MediaPipe Hands (JS)
- **Backend:** FastAPI, PyTorch/TensorFlow, gTTS
- **ML:** LSTM neural network for sequence classification
- **Data:** MediaPipe hand landmarks (21 points × 3D = 63 features)

## 🐍 Python Version Support

- **PyTorch version:** Works with Python 3.8-3.14+ ✅ (Recommended for Python 3.14)
- **TensorFlow version:** Works with Python 3.8-3.11 only
- See [PYTHON_314_GUIDE.md](./PYTHON_314_GUIDE.md) for Python 3.14 setup

## 📝 License

MIT License - feel free to use and modify!



