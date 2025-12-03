"""
FastAPI backend for SignSpeak using PyTorch.

This version works with Python 3.14+ since PyTorch supports newer Python versions.

Exposes:
- POST /predict  → sign language recognition from hand landmarks
- POST /speak    → text-to-speech using gTTS
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from utils.preprocess import LandmarkSequenceBuffer
from utils.tts import synthesize_speech

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "sign_model.pt"
METADATA_PATH = BASE_DIR / "model" / "sign_model.json"

app = FastAPI(title="SignSpeak Backend (PyTorch)", version="1.0.0")

# Allow local dev origins; adjust as needed for production.
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    landmarks: List[float]


class PredictResponse(BaseModel):
    prediction: str
    confidence: float


class SpeakRequest(BaseModel):
    text: str


sequence_buffer = LandmarkSequenceBuffer(sequence_length=30)


class SignLSTMClassifier(nn.Module):
    """LSTM-based sign language classifier (must match training script)."""

    def __init__(self, sequence_length: int, feature_dim: int, num_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = None
class_names: List[str] = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Load PyTorch model and metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    # Load model checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Extract metadata
    sequence_length = checkpoint.get("sequence_length", 30)
    feature_dim = checkpoint.get("feature_dim", 63)
    num_classes = checkpoint.get("num_classes")
    class_names_list = checkpoint.get("class_names", [])

    # If metadata not in checkpoint, try JSON file
    if not class_names_list and METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
            class_names_list = metadata.get("class_names", [])

    if not num_classes:
        num_classes = len(class_names_list) if class_names_list else 2

    # Create model and load weights
    model = SignLSTMClassifier(sequence_length, feature_dim, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names_list


def init_model():
    """Initialize model on first request."""
    global model, class_names
    if model is None:
        model, class_names = load_model()
        if not class_names:
            num_classes = len(class_names) if class_names else model.fc2.out_features
            class_names = [f"Sign {i}" for i in range(num_classes)]


@app.on_event("startup")
def startup_event():
    """Lazy-load model on startup."""
    try:
        init_model()
    except Exception:
        # Fail silently here; /predict will surface a clear error.
        pass


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Accept a single-frame set of landmarks (63 values) and run the
    sign recognition model on the rolling sequence buffer.
    """
    if len(req.landmarks) != 63:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 63 landmark values, got {len(req.landmarks)}",
        )

    try:
        init_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    sequence_buffer.add_frame(req.landmarks)
    seq_batch = sequence_buffer.to_batch()  # (1, seq_len, 63)

    # Convert to PyTorch tensor
    seq_tensor = torch.FloatTensor(seq_batch).to(device)

    # Model inference
    with torch.no_grad():
        logits = model(seq_tensor)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs_np = probs.cpu().numpy()[0]

    class_idx = int(np.argmax(probs_np))
    confidence = float(probs_np[class_idx])
    prediction = class_names[class_idx] if class_names else f"Sign {class_idx}"

    return PredictResponse(prediction=prediction, confidence=confidence)


@app.post("/speak")
async def speak(req: SpeakRequest):
    """
    Convert text to speech using gTTS and stream the resulting audio.
    Returns an MP3 file.
    """
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty")

    try:
        audio_path, mime_type = synthesize_speech(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

    return FileResponse(
        path=audio_path,
        media_type=mime_type,
        filename="speech.mp3",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/")
async def root():
    return {"status": "ok", "message": "SignSpeak backend is running (PyTorch)"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


