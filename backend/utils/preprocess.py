"""
Preprocessing utilities for SignSpeak.

Converts raw 63-length landmark vectors from the frontend into
model-ready sequences of shape (1, sequence_length, 63).
"""

from collections import deque
from typing import Deque, List
import numpy as np


class LandmarkSequenceBuffer:
    """
    Maintains a fixed-length sequence of landmark frames.

    The model expects (sequence_length, 63) as input. We keep a rolling
    window of frames and expose a method to export the current buffer
    as a batch of shape (1, sequence_length, 63).
    """

    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self._buffer: Deque[np.ndarray] = deque(maxlen=sequence_length)

    def add_frame(self, landmarks: List[float]) -> None:
        """Append a single frame (63-length list) to the buffer."""
        arr = np.asarray(landmarks, dtype=np.float32)
        if arr.shape != (63,):
            raise ValueError(f"Expected landmarks of shape (63,), got {arr.shape}")
        self._buffer.append(arr)

    def is_ready(self) -> bool:
        """Return True if we currently have a full sequence."""
        return len(self._buffer) == self.sequence_length

    def to_batch(self) -> np.ndarray:
        """
        Export the current buffer as a batch of shape (1, sequence_length, 63).
        If the buffer is not yet full, left-pad with zeros.
        """
        frames = list(self._buffer)
        if len(frames) < self.sequence_length:
            pad_count = self.sequence_length - len(frames)
            pad_frames = [np.zeros(63, dtype=np.float32) for _ in range(pad_count)]
            frames = pad_frames + frames
        seq = np.stack(frames, axis=0)  # (sequence_length, 63)
        return seq[np.newaxis, ...]  # (1, sequence_length, 63)



