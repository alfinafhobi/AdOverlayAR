import time
from collections import deque

class FPSTracker:
    def __init__(self, window=30):
        self.times = deque(maxlen=window)
        self.last = None

    def tick(self):
        now = time.time()
        if self.last is not None:
            self.times.append(now - self.last)
        self.last = now

    @property
    def fps(self):
        if not self.times:
            return 0.0
        avg = sum(self.times) / len(self.times)
        return 1.0 / avg if avg > 0 else 0.0
