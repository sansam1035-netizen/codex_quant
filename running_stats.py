from collections import defaultdict, deque
import numpy as np


class RunningStats:
    def __init__(self, maxlen=5000):
        self.buf = defaultdict(lambda: deque(maxlen=maxlen))  # (name, key) -> deque
        self.ema = {}  # (name, key) -> (value)

    def push(self, name, key, x):
        self.buf[(name, key)].append(float(x))

    def quantile(self, name, key, q, fallback=0.0):
        arr = np.array(self.buf.get((name, key), []), dtype=float)
        if arr.size < 50:
            return fallback
        return float(np.quantile(arr, q))

    def ema_update(self, name, key, x, half_life_sec):
        alpha = 1.0 - np.exp(-np.log(2.0) / max(half_life_sec, 1e-6))
        k = (name, key)
        v = self.ema.get(k, float(x))
        v = (1 - alpha) * v + alpha * float(x)
        self.ema[k] = v
        return v

    def robust_z(self, name, key, x, fallback=0.0):
        arr = np.array(self.buf.get((name, key), []), dtype=float)
        if arr.size < 200:
            return fallback
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) + 1e-9
        return float((x - med) / (1.4826 * mad))
