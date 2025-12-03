import React, { useEffect, useRef, useState } from 'react';
import { createHandTracker } from '../mediapipe/handTracker';
import { sendLandmarks } from '../api/api';

// Webcam component: handles WebRTC video, MediaPipe tracking, and polling backend.
const Webcam = ({ onPrediction }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const trackerRef = useRef(null);
  const lastSentRef = useRef(0);
  const [status, setStatus] = useState('Initializing camera…');

  useEffect(() => {
    let stream;
    let cancelled = false;

    const setup = async () => {
      try {
        setStatus('Requesting camera access…');
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480 }
        });
        if (!videoRef.current) return;

        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        if (!canvasRef.current) return;
        canvasRef.current.width = videoRef.current.videoWidth || 640;
        canvasRef.current.height = videoRef.current.videoHeight || 480;

        if (cancelled) return;

        trackerRef.current = createHandTracker(
          videoRef.current,
          canvasRef.current,
          handleLandmarks
        );
        trackerRef.current.start();
        setStatus('Tracking hand landmarks');
      } catch (err) {
        console.error('Error setting up webcam', err);
        setStatus('Unable to access camera');
      }
    };

    setup();

    return () => {
      cancelled = true;
      if (trackerRef.current) {
        trackerRef.current.stop();
      }
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const handleLandmarks = async (flatLandmarks) => {
    const now = performance.now();
    // Throttle sending to backend to once every ~200 ms
    if (now - lastSentRef.current < 200) return;
    lastSentRef.current = now;

    try {
      const res = await sendLandmarks(flatLandmarks);
      if (onPrediction) {
        onPrediction(res.prediction, res.confidence);
      }
    } catch (err) {
      console.error('Prediction error', err);
    }
  };

  return (
    <div>
      <div className="video-container">
        <video
          ref={videoRef}
          className="video-element"
          playsInline
          muted
        />
        <canvas ref={canvasRef} className="overlay-canvas" />
      </div>
      <p style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: '#9ca3af' }}>
        {status}
      </p>
    </div>
  );
};

export default Webcam;


