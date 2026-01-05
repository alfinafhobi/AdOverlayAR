import numpy as np

def _edge_len(p, q):
    dx, dy = p[0]-q[0], p[1]-q[1]
    return float(np.hypot(dx, dy))

def _quad_aspect_robust(quad):
    """Estimate aspect ratio (width/height) even under tilt."""
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    tl, tr, br, bl = q
    top    = _edge_len(tl, tr)
    bottom = _edge_len(bl, br)
    left   = _edge_len(tl, bl)
    right  = _edge_len(tr, br)
    width  = max(1e-6, 0.5*(top + bottom))
    height = max(1e-6, 0.5*(left + right))
    return width / height

class DecisionEngine:
    def __init__(self, asset_cache):
        self.cache = asset_cache

    def choose_image(self, quad):
        aspect = _quad_aspect_robust(quad)
        img = self.cache.get_image_for_aspect(aspect)
        return img, float(aspect)
