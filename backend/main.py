"""
FastAPI backend for SignSpeak.

Exposes:
- POST /predict  → sign language recognition from hand landmarks
- POST /speak    → text-to-speech using gTTS
"""

from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from utils.preprocess import LandmarkSequenceBuffer
from utils.tts import synthesize_speech

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "sign_model.h5"

app = FastAPI(title="SignSpeak Backend", version="1.0.0")

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


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


model = None
class_names: List[str] = []


def init_model():
    global model, class_names
    if model is None:
        model = load_model()
        # Try to infer class names from model metadata, if set during training.
        # Otherwise, default to generic labels.
        if hasattr(model, "class_names"):
            class_names = list(model.class_names)
        else:
            num_classes = model.output_shape[-1]
            class_names = [f"Sign {i}" for i in range(num_classes)]


@app.on_event("startup")
def startup_event():
    """Lazy-load model on startup so first request is fast for the client."""
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

    # Model outputs logits / probabilities over classes
    logits = model.predict(seq_batch, verbose=0)
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    prediction = class_names[class_idx]

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

    # FileResponse will handle streaming the file to the client.
    return FileResponse(
        path=audio_path,
        media_type=mime_type,
        filename="speech.mp3",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/")
async def root():
    return {"status": "ok", "message": "SignSpeak backend is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


