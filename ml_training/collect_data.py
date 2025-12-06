"""
Data collection script for Vox.

Uses OpenCV + MediaPipe Hands to capture 30-frame sequences of
21 hand landmarks (x, y, z) per frame, and saves them as NumPy arrays.

Usage:
    python collect_data.py --output_dir data --label hello --num_sequences 20
"""

import argparse
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def collect_sequences(output_dir: Path, label: str, num_sequences: int, seq_len: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: cv2.VideoCapture(0) returned False (cap not opened)")
        raise RuntimeError("Unable to access webcam")
    print("✅ Webcam accessed successfully")

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        for seq_idx in range(num_sequences):
            sequence = []

            print(f"Get ready for sequence {seq_idx + 1}/{num_sequences} for label '{label}'")
            time.sleep(2.0)
            print("Recording...")

            while len(sequence) < seq_len:
                ret, frame = cap.read()
                if not ret:
                    print("❌ ERROR: cap.read() returned False (failed to read frame)")
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    flat = []
                    for lm in hand_landmarks.landmark:
                        flat.extend([lm.x, lm.y, lm.z])
                    if len(flat) == 63:
                        sequence.append(flat)

                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                cv2.putText(
                    image,
                    f"{label} seq {seq_idx+1}/{num_sequences} frame {len(sequence)}/{seq_len}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Vox Data Collection", image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            sequence = np.asarray(sequence, dtype=np.float32)
            if sequence.shape != (seq_len, 63):
                print(f"Skipping invalid sequence shape {sequence.shape}")
                continue

            save_path = output_dir / f"{label}_{seq_idx:03d}.npy"
            np.save(save_path, sequence)
            print(f"Saved {save_path}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to store .npy sequences")
    parser.add_argument("--label", type=str, required=True, help="Name of the sign label")
    parser.add_argument("--num_sequences", type=int, default=20, help="Number of sequences to capture")
    parser.add_argument("--sequence_length", type=int, default=30, help="Frames per sequence")
    args = parser.parse_args()

    collect_sequences(
        output_dir=Path(args.output_dir),
        label=args.label,
        num_sequences=args.num_sequences,
        seq_len=args.sequence_length,
    )


if __name__ == "__main__":
    main()



