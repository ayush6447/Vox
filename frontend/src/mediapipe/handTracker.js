// MediaPipe Hands wrapper for tracking hand landmarks in the browser.
// Exposes a simple function to start tracking on a video element.

import { Hands } from '@mediapipe/hands';
import { Camera } from '@mediapipe/camera_utils';
import * as drawingUtils from '@mediapipe/drawing_utils';

// Initialize MediaPipe Hands instance
export const createHandTracker = (videoElement, canvasElement, onLandmarks) => {
  const canvasCtx = canvasElement.getContext('2d');

  const hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6
  });

  // Called by MediaPipe for each processed frame
  hands.onResults((results) => {
    // Draw video frame to canvas
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(
      results.image,
      0,
      0,
      canvasElement.width,
      canvasElement.height
    );

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      for (const landmarks of results.multiHandLandmarks) {
        drawingUtils.drawConnectors(
          canvasCtx,
          landmarks,
          Hands.HAND_CONNECTIONS,
          { color: '#22d3ee', lineWidth: 3 }
        );
        drawingUtils.drawLandmarks(canvasCtx, landmarks, {
          color: '#f97316',
          lineWidth: 1,
          radius: 2
        });

        // Flatten (x, y, z) for 21 landmarks → 63-length vector
        const flat = [];
        for (const lm of landmarks) {
          flat.push(lm.x, lm.y, lm.z ?? 0);
        }

        onLandmarks(flat);
      }
    }

    canvasCtx.restore();
  });

  const camera = new Camera(videoElement, {
    onFrame: async () => {
      await hands.send({ image: videoElement });
    },
    width: 640,
    height: 480
  });

  return {
    start: () => camera.start(),
    stop: () => camera.stop()
  };
};



