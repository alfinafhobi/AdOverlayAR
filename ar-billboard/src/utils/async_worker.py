import threading, queue, time

class AsyncWorker:
    def __init__(self, fn, max_queue=1, daemon=True):
        self.fn = fn
        self.q = queue.Queue(maxsize=max_queue)
        self.thread = threading.Thread(target=self._loop, daemon=daemon)
        self._stop = False
        self._latest_result = None

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop = True
        try:
            self.q.put_nowait(None)
        except Exception:
            pass
        self.thread.join(timeout=1.0)

    def submit(self, item):
        """Drop older request if queue is full (keep freshest)."""
        try:
            self.q.put_nowait(item)
        except queue.Full:
            try:
                _ = self.q.get_nowait()
            except Exception:
                pass
            self.q.put_nowait(item)

    def _loop(self):
        while not self._stop:
            item = self.q.get()
            if item is None or self._stop:
                break
            try:
                self._latest_result = self.fn(item)
            except Exception as e:
                self._latest_result = {"error": str(e)}

    def latest_result(self):
        return self._latest_result
