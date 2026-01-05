import cv2
import numpy as np

# ----------------- Basic geometric helpers -----------------
def order_points(pts4: np.ndarray) -> np.ndarray:
    q = np.asarray(pts4, np.float32).reshape(4, 2)
    s = q.sum(1); d = np.diff(q, axis=1).reshape(-1)
    tl = q[np.argmin(s)]; br = q[np.argmax(s)]
    tr = q[np.argmin(d)]; bl = q[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)

def area_of_quad(q): 
    return cv2.contourArea(np.asarray(q, np.float32))

def bbox_to_quad(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)

# ----------------- Scoring helpers for rectangle quality -----------------
def _angle_deg(a, b, c):
    ab = a - b; cb = c - b
    num = np.dot(ab, cb)
    den = np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6
    cosv = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cosv))

def _rectangularity_score(poly4):
    p = order_points(poly4)
    angs = [
        _angle_deg(p[3], p[0], p[1]),
        _angle_deg(p[0], p[1], p[2]),
        _angle_deg(p[1], p[2], p[3]),
        _angle_deg(p[2], p[3], p[0]),
    ]
    dev = np.mean([abs(a - 90.0) for a in angs])
    return 1.0 / (1.0 + dev)  # closer to 90° is better

def _aspect_ratio_score(poly4, expected_aspect=None):
    if expected_aspect is None or expected_aspect <= 0:
        return 1.0
    q = order_points(poly4)
    w = np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
    h = np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])
    if h <= 1e-3 or w <= 1e-3:
        return 0.0
    ar = (w / 2.0) / (h / 2.0)
    return 1.0 / (1.0 + abs(ar - expected_aspect))

# ----------------- Core: refine quad from ROI -----------------
def refine_quad_from_roi(frame_bgr, roi_xyxy, expected_aspect=None, cfg=None):
    """
    Refine a precise quadrilateral inside a YOLO bbox using edges + contours.
    Returns quad (4x2 float32) or None.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in roi_xyxy]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    # --- config defaults ---
    cfg = cfg or {}
    canny1 = int(cfg.get("canny1", 60))
    canny2 = int(cfg.get("canny2", 160))
    min_area_ratio = float(cfg.get("min_area_ratio", 0.02))   # of ROI area
    max_angle_dev = float(cfg.get("max_angle_dev", 25.0))     # degrees tolerance

    roi = frame_bgr[y1:y2+1, x1:x2+1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)

    edges = cv2.Canny(gray, canny1, canny2)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    roi_area = float(roi.shape[0] * roi.shape[1])
    best = None
    best_score = 0.0

    for c in cnts:
        if len(c) < 4:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        area = cv2.contourArea(approx)
        if area < min_area_ratio * roi_area:
            continue

        poly = approx.reshape(-1, 2).astype(np.float32)

        p = order_points(poly)
        angs = [
            _angle_deg(p[3], p[0], p[1]),
            _angle_deg(p[0], p[1], p[2]),
            _angle_deg(p[1], p[2], p[3]),
            _angle_deg(p[2], p[3], p[0]),
        ]
        if np.mean([abs(a - 90.0) for a in angs]) > max_angle_dev:
            continue

        rscore = _rectangularity_score(poly)
        ascore = _aspect_ratio_score(poly, expected_aspect)
        score = (0.7 * rscore) + (0.3 * ascore) + (area / (roi_area + 1e-6)) * 0.15

        if score > best_score:
            best_score = score
            best = poly

    if best is None:
        return None

    # map ROI coords back to full image coords
    best[:, 0] += x1
    best[:, 1] += y1
    return order_points(best)

# ----------------- Visualization helper -----------------
def draw_quad(frame, quad, color=(0, 255, 0), thickness=2):
    """
    Draws a connected quadrilateral on the frame for visualization.
    """
    if quad is None or len(quad) != 4:
        return
    q = np.int32(quad).reshape(-1, 2)
    cv2.polylines(frame, [q], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

