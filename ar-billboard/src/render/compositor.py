import cv2
import numpy as np


def match_brightness(warped_ad, frame, mask, mix=1.0):
    if mix <= 0.0:
        return warped_ad
    m = (mask.astype(np.float32) / 255.0)[..., None]
    if m.sum() < 10:
        return warped_ad

    bg = frame.astype(np.float32) * m
    ad = warped_ad.astype(np.float32) * m

    def _luma(img):
        return 0.299*img[:,:,2] + 0.587*img[:,:,1] + 0.114*img[:,:,0]

    bg_mean = _luma(bg).sum() / (m[...,0].sum() + 1e-6)
    ad_mean = _luma(ad).sum() / (m[...,0].sum() + 1e-6)

    gain = 1.0 if ad_mean <= 1e-3 else float(bg_mean / ad_mean)
    # Clamp to avoid over-dark/over-bright fringes
    gain = max(0.75, min(1.25, gain))

    matched = np.clip(warped_ad.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.addWeighted(warped_ad, 1.0 - mix, matched, mix, 0.0)

def _unsharp(img, amount=0.15):
    if amount <= 0.0:
        return img
    blur = cv2.GaussianBlur(img, (0,0), 1.2)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)

def alpha_blend(base, overlay, mask, opacity=1.0, edge_boost=0.0):
    if edge_boost > 0:
        overlay = _unsharp(overlay, amount=float(edge_boost))
    m = (mask.astype(np.float32) / 255.0) * float(opacity)
    m3 = np.repeat(m[..., None], 3, axis=2)
    out = overlay.astype(np.float32) * m3 + base.astype(np.float32) * (1.0 - m3)
    return np.clip(out, 0, 255).astype(np.uint8)
