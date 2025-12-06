# 🎯 Vox — Real-time Sign Language to Text/Speech

Vox is an end-to-end system that translates sign language gestures into text and speech in real-time.

## ✨ Features

- 🎥 **Real-time webcam capture** using WebRTC
- 🤲 **Hand tracking** with MediaPipe Hands (21 landmarks)
- 🧠 **Sign recognition** using TensorFlow LSTM model
- 🔊 **Text-to-speech** conversion with gTTS
- ⚡ **Low latency** predictions (~200ms updates)

## 🏗️ Project Structure

```
Vox/
├── frontend/              # React + Vite web app
│   ├── src/
│   │   ├── components/
│   │   │   └── Webcam.jsx
│   │   ├── mediapipe/
│   │   │   └── handTracker.js
│   │   ├── api/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.js
├── backend/               # FastAPI server (Python 3.11)
│   ├── main.py
│   ├── requirements.txt
│   ├── model/
│   │   ├── sign_model.h5    # Trained TensorFlow model
│   │   └── sign_model.json  # Class names metadata
│   └── utils/
│       ├── preprocess.py    # Sequence buffer for landmarks
│       └── tts.py           # Text-to-speech with gTTS
└── ml_training/            # Data collection & training
    ├── collect_data.py      # Capture sign sequences
    ├── train_model.py       # Train LSTM model
    ├── requirements.txt
    ├── README.md
    └── data/                # Training data (.npy files)
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.11** (Required - TensorFlow compatibility)
- **Node.js 18+** and npm
- **Webcam** for data collection and real-time inference

### Install Dependencies

**Backend:**
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### 3. Collect & Train Data

1.  **Collect Data**: (Run for each sign you want to recognize)
    ```bash
    cd ml_training
    python collect_data.py --output_dir data --label hello --num_sequences 20
    python collect_data.py --output_dir data --label bye --num_sequences 20
    ```
    *Each sequence captures 30 frames of hand landmarks. Recommended: 20-30 sequences per sign.*

2.  **Train Model**:
    ```bash
    python train_model.py --data_dir data
    ```
    *This automatically saves to `../backend/model/sign_model.h5` and `sign_model.json` (class names).*
    
    **Optional parameters:**
    ```bash
    python train_model.py --data_dir data --epochs 50 --batch_size 32
    ```

### 4. Run the Application

**Terminal 1 - Start Backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Backend will be available at: http://localhost:8000

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm run dev
```
Frontend will be available at: http://localhost:5173

**Step 3: Use the Application**
1. Open browser to http://localhost:5173
2. Allow camera access when prompted
3. Show sign language gestures to the webcam
4. See real-time predictions appear
5. Click "Speak Prediction" to hear text-to-speech

## 🛠️ Tech Stack

- **Frontend:**
  - React 18+ with Vite
  - MediaPipe Hands (JavaScript) for hand tracking
  - WebRTC for webcam access
  - Real-time API communication

- **Backend:**
  - FastAPI (Python 3.11)
  - TensorFlow 2.13+ for LSTM model inference
  - gTTS for text-to-speech
  - NumPy for data processing

- **Machine Learning:**
  - TensorFlow/Keras for model training
  - LSTM neural network architecture
  - MediaPipe Python for data collection
  - OpenCV for webcam capture

- **Data Format:**
  - 21 hand landmarks per frame
  - 3D coordinates (x, y, z) = 63 values per frame
  - 30-frame sequences for temporal modeling

## 📖 API Endpoints

### Backend (FastAPI)

- **POST `/predict`** - Accepts 63-value landmark array, returns prediction and confidence
  ```json
  {
    "landmarks": [0.1, 0.2, 0.3, ...]  // 63 values (21 landmarks × 3D)
  }
  ```
  Response:
  ```json
  {
    "prediction": "hello",
    "confidence": 0.95
  }
  ```

- **POST `/speak`** - Converts text to speech, returns MP3 audio
  ```json
  {
    "text": "hello"
  }
  ```

- **GET `/`** - Health check endpoint

## 🛠️ Troubleshooting

- **Model Loading Error?** 
  - Ensure you trained the model after collecting data
  - Check that `backend/model/sign_model.h5` exists
  - Verify Python 3.11 is being used

- **CORS Errors?**
  - Backend CORS is configured for `http://localhost:5173`
  - Ensure backend is running on port 8000
  - Check browser console for specific errors

- **Wrong Predictions?** 
  - Collect more training data (20-30 sequences per sign)
  - Ensure consistent sign performance during collection
  - Retrain the model with more data

- **Frontend Port?** 
  - Default is 5173, Vite will use next available port if busy
  - Check terminal output for actual port number

- **Camera Not Working?**
  - Grant browser permissions for camera access
  - Use HTTPS or localhost (required for getUserMedia)
  - Try a different browser (Chrome, Firefox, Edge)

## 📚 Additional Documentation

- **`ml_training/README.md`** - Detailed guide for data collection and training
- Check backend terminal for model loading status and errors
- Check browser console (F12) for frontend debugging

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows Python 3.11 compatibility
- All tests pass
- Documentation is updated

## 📜 License

MIT License - feel free to use and modify!
