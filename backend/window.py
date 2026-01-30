# backend/window.py
from collections import deque
import numpy as np

class SlidingWindow:
    def __init__(self, window_size: int, n_features: int):
        self.window_size = window_size
        self.n_features = n_features
        self.buf = deque(maxlen=window_size)
        self.last_timestamp = None

    def push(self, x: np.ndarray, timestamp=None):
        x = np.asarray(x, dtype=np.float32)
        assert x.shape == (self.n_features,), f"expected {(self.n_features,)}, got {x.shape}"
        self.buf.append(x)
        self.last_timestamp = timestamp

    def ready(self) -> bool:
        return len(self.buf) == self.window_size

    def get_window(self) -> np.ndarray:
        # (L, K)
        assert self.ready()
        return np.stack(self.buf, axis=0)
