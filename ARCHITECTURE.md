# 🏗️ System Architecture & Workflow

## 🛠️ Tech Stack

### Frontend (The Visuals)
*   **React + Vite**: Fast, modern framework for building the UI.
*   **MediaPipe Hands (JS)**: Google's library that runs in the browser to detect hands and extract **21 skeletal landmarks** (joints) from the webcam feed in real-time.
*   **WebRTC**: Used to access the user's webcam.

### Backend (The Brain)
*   **FastAPI (Python)**: A high-performance web framework that handles API requests.
*   **TensorFlow / Keras**: Runs the actual AI model (`sign_model.h5`) to classify gestures from the landmark data.
*   **gTTS (Google Text-to-Speech)**: Converts the predicted text into spoken audio files.

---

## 🔄 System Workflow

### 1. Capture & Track (Frontend)
*   The browser accesses the webcam via **WebRTC**.
*   **MediaPipe** detects your hand in every video frame.
*   It calculates the `(x, y, z)` coordinates for 21 points on your hand (Total: 63 values per frame).
*   This happens continuously (~30 frames per second).

### 2. Send Data (API)
*   The frontend acts as a client.
*   It sends these 63-value coordinate arrays to the backend API via a **POST request** to `/predict`.

### 3. Process & Predict (Backend)
*   The backend maintains a **"Sequence Buffer"** (a rolling window of the last 30 frames). This gives the model context about movement over time, not just a static pose.
*   When a new frame arrives, it's added to the buffer.
*   The buffer is fed into the **LSTM (Long Short-Term Memory)** neural network.
*   The model analyzes the temporal sequence and outputs a probability score for each known class (e.g., `Hello: 99%`, `Bye: 1%`).

### 4. Feedback Loop
*   **Prediction**: The backend returns the text (e.g., "Hello") to the frontend.
*   **Display**: The frontend updates the UI to show the detected sign.
*   **Speech**: If the user clicks "Speak", the backend uses **gTTS** to generate an MP3 file and streams it back to the browser for playback.
