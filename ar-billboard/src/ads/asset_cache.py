import os, cv2
import numpy as np
from typing import List, Tuple

def _aspect_ratio(img):
    h, w = img.shape[:2]
    return w / max(h, 1)

def _closest_by_aspect(imgs, target_ar):
    if not imgs:
        return None
    return min(imgs, key=lambda im: abs(_aspect_ratio(im) - target_ar))

class AssetCache:
    def __init__(self, playlists_cfg_path: str):
        self.images: List[np.ndarray] = []
        self.paths: List[str] = []
        self._load_images(playlists_cfg_path)

    def _load_images(self, playlists_cfg_path):
        # Robust YAML read (no external deps here)
        import yaml
        with open(playlists_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for p in cfg.get("default_images", []):
            if os.path.exists(p):
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is not None:
                    self.images.append(img)
                    self.paths.append(p)
        if not self.images:
            print("[WARN] No default_images loaded. Add images to config/playlists.yaml.")

    def get_image_for_aspect(self, target_ar: float):
        """
        Returns a BGR image best matching the requested aspect ratio.
        """
        return _closest_by_aspect(self.images, target_ar)
