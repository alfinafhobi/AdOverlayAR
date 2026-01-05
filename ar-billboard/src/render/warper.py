# src/render/warper.py
import cv2, numpy as np

_render_prefs_ = {"canvas_scale": 1.2, "warp_interp": "cubic"}

def _order_points(pts4: np.ndarray) -> np.ndarray:
    q = np.asarray(pts4, np.float32).reshape(4, 2)
    s = q.sum(1); d = np.diff(q, axis=1).reshape(-1)
    tl = q[np.argmin(s)]; br = q[np.argmax(s)]
    tr = q[np.argmin(d)]; bl = q[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)

def _interp_flag(name: str) -> int:
    name = (name or "cubic").lower()
    return {
        "nearest": cv2.INTER_NEAREST,
        "linear":  cv2.INTER_LINEAR,
        "bilinear":cv2.INTER_LINEAR,
        "cubic":   cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }.get(name, cv2.INTER_CUBIC)

def warp_ad_to_quad(ad_bgr: np.ndarray,
                    quad_xy: np.ndarray,
                    canvas_shape: tuple,
                    target_aspect: float | None = None,
                    fit_mode: str = "cover",
                    feather: int = 3):
    if ad_bgr is None or ad_bgr.size == 0:
        return None, None

    Hc, Wc = int(canvas_shape[0]), int(canvas_shape[1])
    quad = _order_points(quad_xy)
    quad[:, 0] = np.clip(quad[:, 0], 0, Wc - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, Hc - 1)

    ha, wa = ad_bgr.shape[:2]
    if ha < 2 or wa < 2:
        return None, None

    if target_aspect is not None and target_aspect > 0:
        cur_aspect = wa / float(ha)
        if (fit_mode or "cover").lower() == "contain":
            if cur_aspect > target_aspect:
                new_h = int(round(wa / target_aspect))
                pad = max(0, (new_h - ha) // 2)
                ad_bgr = cv2.copyMakeBorder(ad_bgr, pad, pad, 0, 0, cv2.BORDER_REPLICATE)
            else:
                new_w = int(round(ha * target_aspect))
                pad = max(0, (new_w - wa) // 2)
                ad_bgr = cv2.copyMakeBorder(ad_bgr, 0, 0, pad, pad, cv2.BORDER_REPLICATE)
            ha, wa = ad_bgr.shape[:2]

    src = np.array([[0, 0],
                    [wa - 1, 0],
                    [wa - 1, ha - 1],
                    [0, ha - 1]], np.float32)

    H = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    if H is None or not np.isfinite(H).all():
        return None, None

    flags = _interp_flag(_render_prefs_.get("warp_interp", "cubic"))
    warped = cv2.warpPerspective(ad_bgr, H, (Wc, Hc), flags=flags, borderMode=cv2.BORDER_TRANSPARENT)

    mask = np.zeros((Hc, Wc), np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    if feather and feather > 0:
        k = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0, 0)

    return warped, mask
