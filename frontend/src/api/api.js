// Centralized API client for Vox frontend
// All endpoints are relative to the backend FastAPI server.

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Send a single-frame landmark vector to the backend for prediction.
// landmarks: array of 63 numbers (21 keypoints * (x, y, z))
export const sendLandmarks = async (landmarks) => {
  const res = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ landmarks })
  });

  if (!res.ok) {
    throw new Error(`Prediction request failed with ${res.status}`);
  }

  return res.json();
};

// Request TTS audio for a given text string.
export const speakText = async (text) => {
  const res = await fetch(`${BASE_URL}/speak`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ text })
  });

  if (!res.ok) {
    throw new Error(`TTS request failed with ${res.status}`);
  }

  const blob = await res.blob();
  return blob;
};



