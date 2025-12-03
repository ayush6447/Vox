SignSpeak — Real-time Sign Language to Text/Speech
=================================================

SignSpeak is an end-to-end prototype system that:

- Captures webcam video in the browser
- Uses MediaPipe Hands in the frontend to extract 3D hand landmarks
- Streams landmark sequences to a FastAPI backend
- Runs a TensorFlow LSTM sign recognition model
- Returns the predicted text and confidence
- Optionally converts text to speech via gTTS and streams audio back to the browser

This repository is organized into three main parts:

- `frontend` — Vite + React web client
- `backend` — FastAPI inference + TTS server
- `ml_training` — offline data collection & training scripts

See each folder for more details and how to run individual parts.


