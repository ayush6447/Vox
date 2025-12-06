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
├── frontend/          # React + Vite web app
├── backend/           # FastAPI server (Python 3.11)
│   ├── main.py
│   ├── model/         # Trained model + class names JSON
│   └── utils/
├── ml_training/       # Data collection & training
│   ├── collect_data.py
│   ├── train_model.py
│   ├── augment_data.py # Data augmentation script
│   └── data/          # Training samples
```

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.11** (Required)
- **Node.js 18+**

### 2. Install Dependencies

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

1.  **Collect Data**: (Run for each sign, e.g., 'hello', 'bye')
    ```bash
    cd ml_training
    python collect_data.py --output_dir data --label hello --num_sequences 20
    python collect_data.py --output_dir data --label bye --num_sequences 20
    ```

2.  **Augment Data** (Optional but recommended):
    ```bash
    # Generates synthetic variations (noise, scale, shift) to improve accuracy
    python augment_data.py
    ```

3.  **Train Model**:
    ```bash
    python train_model.py --data_dir data --output_model backend/model/sign_model.h5
    ```

   *Note: This saves both `sign_model.h5` and `sign_model.json` (class names).*

### 4. Run the Application

**Step 1: Start Backend**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Step 2: Start Frontend**
```bash
cd frontend
npm run dev
```

**Step 3: Open Browser**
Go to [http://localhost:5173](http://localhost:5173) (or port 5174 if 5173 is busy).

## 🛠️ Troubleshooting

- **Model Loading Error?** Ensure you trained the model *after* collecting data. The backend needs `backend/model/sign_model.h5`.
- **Wrong Predictions?** Try augmenting your data (`augment_data.py`) and retraining.
- **Frontend Port?** If 5173 is taken, Vite uses the next available port (check terminal).

## 📜 License
MIT License
