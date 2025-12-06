"""
FastAPI backend for Vox - Real-time Sign Language to Text/Speech translation.

Exposes:
- POST /predict  → sign language recognition from hand landmarks
- POST /speak    → text-to-speech using gTTS
"""

from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from utils.preprocess import LandmarkSequenceBuffer
from utils.tts import synthesize_speech

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "sign_model.h5"

app = FastAPI(title="Vox Backend", version="1.0.0")

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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure CORS headers are always sent, even on errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
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
        
        # Try to load class names from JSON
        import json
        json_path = MODEL_PATH.with_suffix(".json")
        if json_path.exists():
            with open(json_path, "r") as f:
                class_names = json.load(f)
        elif hasattr(model, "class_names"):
            class_names = list(model.class_names)
        else:
            num_classes = model.output_shape[-1]
            class_names = [f"Sign {i}" for i in range(num_classes)]


@app.on_event("startup")
def startup_event():
    """Lazy-load model on startup so first request is fast for the client."""
    try:
      init_model()
      print(f"✅ Model loaded successfully. Classes: {class_names}")
    except Exception as e:
      # Log error but don't fail - /predict will surface a clear error.
      print(f"⚠️  Model not loaded on startup: {e}")
      print("   Will attempt to load on first /predict request.")


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model initialization failed: {str(e)}")

    try:
        sequence_buffer.add_frame(req.landmarks)
        seq_batch = sequence_buffer.to_batch()  # (1, seq_len, 63)

        # Model outputs probabilities (since last layer is softmax)
        probs = model.predict(seq_batch, verbose=0)[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        prediction = class_names[class_idx] if class_names else f"Sign {class_idx}"

        return PredictResponse(prediction=prediction, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


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
    return {"status": "ok", "message": "Vox backend is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)



