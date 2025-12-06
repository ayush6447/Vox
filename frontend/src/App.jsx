import React, { useState } from 'react';
import Webcam from './components/Webcam';
import { speakText } from './api/api';

// Root application component composing webcam, prediction, and TTS
const App = () => {
  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const handlePredictionUpdate = (text, conf) => {
    setPrediction(text);
    setConfidence(conf);
  };

  const handleSpeak = async () => {
    if (!prediction) return;
    try {
      setIsSpeaking(true);
      const audioBlob = await speakText(prediction);
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      audio.play();
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
        setIsSpeaking(false);
      };
    } catch (err) {
      console.error('Error playing TTS audio', err);
      setIsSpeaking(false);
    }
  };

  return (
    <div className="app-root">
      <header className="app-header">
        <h1>Vox</h1>
        <p>Real-time Sign Language to Text &amp; Speech</p>
      </header>
      <main className="app-main">
        <section className="video-section">
          <Webcam onPrediction={handlePredictionUpdate} />
        </section>
        <section className="prediction-section">
          <div className="prediction-card">
            <h2>Predicted Text</h2>
            <p className="prediction-text">{prediction || 'Waiting for sign...'}</p>
            <p className="prediction-confidence">
              Confidence:{' '}
              <span>
                {confidence ? `${(confidence * 100).toFixed(1)}%` : '--'}
              </span>
            </p>
            <button
              className="speak-button"
              onClick={handleSpeak}
              disabled={!prediction || isSpeaking}
            >
              {isSpeaking ? 'Speaking…' : 'Speak Prediction'}
            </button>
          </div>
        </section>
      </main>
      <footer className="app-footer">
        <span>Vox &copy; {new Date().getFullYear()}</span>
      </footer>
    </div>
  );
};

export default App;



