import cv2
import numpy as np

class Camera:
    def __init__(self, index=0, backend="cv2", width=1280, height=720,
                 autofocus=True, auto_exposure=True):
        if backend == "dshow":
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(index)

        # Drop camera buffer to 1 to reduce latency and CPU copies
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        except Exception: pass
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto_exposure else 0.25)
        except Exception: pass
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera.")
        return frame

    def release(self):
        if self.cap:
            self.cap.release()


class QualityEnhancer:
    def __init__(self, cfg):
        q = cfg.get("quality", {})
        self.resize_to = tuple(q.get("resize_to", [1280, 720]))
        self.denoise_cfg = q.get("denoise", {"enable": True})
        self.sharpen_cfg = q.get("sharpen", {"enable": True})
        self.sr_cfg = q.get("superres", {"enable": False})

        self._sr = None
        self._frame_count = 0
        self._last_denoised = None
        self._apply_every = int(self.denoise_cfg.get("apply_every", 1))

        if self.sr_cfg.get("enable", False):
            self._init_superres()

    def _init_superres(self):
        try:
            from cv2 import dnn_superres
            self._sr = dnn_superres.DnnSuperResImpl_create()
            model = self.sr_cfg.get("model", "EDSR").upper()
            scale = int(self.sr_cfg.get("scale", 2))
            weights = self.sr_cfg["weights_path"]
            self._sr.readModel(weights)
            self._sr.setModel(model, scale)
        except Exception as e:
            print(f"[WARN] SuperRes init failed ({e}). Disabling.")
            self.sr_cfg["enable"] = False
            self._sr = None

    def toggle_denoise(self):
        self.denoise_cfg["enable"] = not self.denoise_cfg.get("enable", True)

    def toggle_sharpen(self):
        self.sharpen_cfg["enable"] = not self.sharpen_cfg.get("enable", True)

    def toggle_superres(self):
        if self._sr is None and self.sr_cfg.get("enable", False) is False:
            self.sr_cfg["enable"] = True
            self._init_superres()
        else:
            self.sr_cfg["enable"] = not self.sr_cfg.get("enable", False)

    def _denoise_once(self, img):
        p = self.denoise_cfg
        # Much faster than calling every frame
        return cv2.fastNlMeansDenoisingColored(
            img, None,
            h=p.get("hLuma", 6),
            hColor=p.get("hColor", 4),
            templateWindowSize=p.get("templateWindowSize", 7),
            searchWindowSize=p.get("searchWindowSize", 21),
        )

    def _denoise(self, img):
        if not self.denoise_cfg.get("enable", True):
            return img
        # apply every N frames, reuse cached result in-between
        if (self._frame_count % max(1, self._apply_every)) == 0 or self._last_denoised is None:
            self._last_denoised = self._denoise_once(img)
            return self._last_denoised
        else:
            return self._last_denoised

    def _sharpen(self, img):
        if not self.sharpen_cfg.get("enable", True):
            return img
        amt = float(self.sharpen_cfg.get("amount", 0.4))
        blur = cv2.GaussianBlur(img, (0,0), sigmaX=2.0)
        sharp = cv2.addWeighted(img, 1 + amt, blur, -amt, 0)
        return sharp

    def _superres(self, img):
        if not self.sr_cfg.get("enable", False) or self._sr is None:
            return img
        try:
            return self._sr.upsample(img)
        except Exception:
            return img

    def process(self, frame):
        self._frame_count += 1
        out = self._denoise(frame)
        out = self._superres(out)
        if self.resize_to is not None:
            out = cv2.resize(out, self.resize_to, interpolation=cv2.INTER_AREA)
        out = self._sharpen(out)
        return out
