import cv2
import numpy as np

def _to_gray(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _order_quad(q):
    q = np.asarray(q, dtype=np.float32).reshape(4, 2)
    s = q.sum(1); d = np.diff(q, axis=1).reshape(-1)
    tl = q[np.argmin(s)]; br = q[np.argmax(s)]
    tr = q[np.argmin(d)]; bl = q[np.argmax(d)]
    return np.array([tl, tr, br, bl], np.float32)

def area_of_quad(q): 
    return cv2.contourArea(np.asarray(q, np.float32))

def _quad_mask(shape, quad, dilate=2):
    mask = np.zeros(shape[:2], np.uint8)
    q = np.asarray(quad, np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(mask, q, 255)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate, dilate))
        mask = cv2.dilate(mask, k)
    return mask

def _sample_grid(mask, step=14):
    h, w = mask.shape
    pts = []
    # sample a coarse grid inside the mask
    for y in range(step // 2, h, step):
        row = mask[y]
        for x in range(step // 2, w, step):
            if row[x]:
                pts.append([[float(x), float(y)]])
    return np.array(pts, np.float32) if pts else None

class QuadTracker:
    """
    Robust homography tracker:
      - seeds many points: GFTT + edges + grid inside quad
      - forward–backward LK consistency
      - periodic reseed
    """

    def __init__(self, cfg_or_kwargs=None, **kw):
        # ---- config normalize ----
        cfg = cfg_or_kwargs or {}
        if isinstance(cfg, dict):
            cfg.update(kw)

        # lk_win can be int/float or [w,h]
        lk_win = cfg.get("lk_win", 23)
        if isinstance(lk_win, (int, float)):
            self.lk_win = (int(lk_win), int(lk_win))
        elif isinstance(lk_win, (list, tuple)) and len(lk_win) == 2:
            self.lk_win = (int(lk_win[0]), int(lk_win[1]))
        else:
            self.lk_win = (23, 23)

        self.lk_max_level   = int(cfg.get("lk_max_level", 3))
        self.lk_iters       = int(cfg.get("lk_iters", 30))
        self.lk_eps         = float(cfg.get("lk_eps", 0.01))
        self.lk_criteria    = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               self.lk_iters, self.lk_eps)

        self.max_point_error = float(cfg.get("max_point_error", 4.0))
        self.min_ok_points   = int(cfg.get("min_ok_points", 8))
        self.ema_alpha       = float(cfg.get("ema_alpha", 0.4))
        self.n_features      = int(cfg.get("n_features", 120))
        self.reseed_thresh   = int(cfg.get("reseed_thresh", 40))
        self.reseed_every    = int(cfg.get("reseed_every", 10))
        self.fb_thresh       = float(cfg.get("fb_thresh", 0.9))

        self.prev_gray  = None
        self.prev_pts   = None
        self.prev_quad  = None
        self.smooth_quad= None
        self.frame_count= 0
        self.debug_pts  = None  # for visualization

    # ---------------- seeding ----------------
    def _seed_points(self, gray, quad):
        mask = _quad_mask(gray.shape, quad, dilate=2)

        # 1) GFTT corners
        pts1 = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.n_features, qualityLevel=0.01,
            minDistance=6, mask=mask, blockSize=7, useHarrisDetector=False
        )

        # 2) Edge points
        edges = cv2.Canny(gray, 50, 120)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        ys, xs = np.where(edges > 0)
        pts2 = None
        if len(xs) > 60:
            take = np.linspace(0, len(xs) - 1, num=min(120, len(xs)), dtype=int)
            pts2 = np.array([[[float(xs[i]), float(ys[i])]] for i in take], np.float32)

        # 3) Grid fallback
        pts3 = _sample_grid(mask, step=14)

        pts_list = [p for p in (pts1, pts2, pts3) if p is not None]
        if not pts_list:
            pts = quad.reshape(4, 1, 2).astype(np.float32)
        else:
            pts = np.vstack(pts_list)
            # include corners
            pts = np.vstack([quad.reshape(4, 1, 2).astype(np.float32), pts])

        # de-duplicate (rounded integer coords, then back to float32)
        if len(pts) > 0:
            pts = np.unique(pts.reshape(-1, 2).round().astype(np.int32), axis=0)
            pts = pts.astype(np.float32).reshape(-1, 1, 2)

        return pts

    # ---------------- API ----------------
    def init(self, frame_bgr, quad):
        self.prev_gray   = _to_gray(frame_bgr)
        self.prev_quad   = _order_quad(quad)
        self.smooth_quad = self.prev_quad.copy()
        self.prev_pts    = self._seed_points(self.prev_gray, self.prev_quad)
        self.frame_count = 0
        self.debug_pts   = self.prev_pts.copy() if self.prev_pts is not None else None

    def _apply_H(self, H, quad4):
        q = quad4.astype(np.float32)
        ones = np.ones((4, 1), np.float32)
        qh = np.hstack([q, ones])
        q2 = (H @ qh.T).T
        q2 = q2[:, :2] / (q2[:, 2:3] + 1e-6)
        return _order_quad(q2)

    def _lk_fb_filter(self, prev_gray, gray, pts):
        # Ensure type/shape
        pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)

        # forward
        nxt, st1, err1 = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, pts, None,
            winSize=self.lk_win, maxLevel=self.lk_max_level,
            criteria=self.lk_criteria
        )
        # backward
        back, st2, err2 = cv2.calcOpticalFlowPyrLK(
            gray, prev_gray, nxt, None,
            winSize=self.lk_win, maxLevel=self.lk_max_level,
            criteria=self.lk_criteria
        )

        if nxt is None or back is None or st1 is None or st2 is None:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        st1 = st1.reshape(-1)
        st2 = st2.reshape(-1)
        if err1 is None:
            err1 = np.zeros((len(pts), 1), np.float32)

        good_prev, good_next = [], []
        for i in range(len(pts)):
            if st1[i] == 1 and st2[i] == 1:
                fb = np.linalg.norm(pts[i, 0] - back[i, 0])
                e1 = float(err1[i][0]) if (err1 is not None and len(err1) > i) else 0.0
                if np.isfinite(nxt[i]).all() and fb <= self.fb_thresh and e1 <= self.max_point_error:
                    good_prev.append(pts[i, 0]); good_next.append(nxt[i, 0])

        if not good_prev:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        return (np.array(good_prev, np.float32).reshape(-1, 2),
                np.array(good_next, np.float32).reshape(-1, 2))

    def track(self, frame_bgr):
        if self.prev_gray is None or self.prev_pts is None or self.prev_quad is None:
            return None, False, None

        self.frame_count += 1
        gray = _to_gray(frame_bgr)

        prev, nxt = self._lk_fb_filter(self.prev_gray, gray, self.prev_pts)
        n = len(nxt)
        self.prev_gray = gray  # update pyramids

        if n < self.min_ok_points:
            # reseed for next frame; report fail now
            self.prev_pts = self._seed_points(self.prev_gray, self.prev_quad)
            self.debug_pts = self.prev_pts
            return None, False, None

        H, inliers = cv2.findHomography(prev, nxt, cv2.RANSAC, 4.0)
        if H is None or not np.isfinite(H).all():
            self.prev_pts = self._seed_points(self.prev_gray, self.prev_quad)
            self.debug_pts = self.prev_pts
            return None, False, None

        new_quad = self._apply_H(H, self.prev_quad)
        self.smooth_quad = (1.0 - self.ema_alpha) * self.smooth_quad + self.ema_alpha * new_quad
        self.prev_quad = self.smooth_quad.copy()

        # reseed periodically or when count dropped a lot
        if (n < self.reseed_thresh) or (self.frame_count % self.reseed_every == 0):
            self.prev_pts = self._seed_points(self.prev_gray, self.prev_quad)
        else:
            self.prev_pts = nxt.reshape(-1, 1, 2)

        self.debug_pts = self.prev_pts
        score = float(inliers.mean()) if inliers is not None else None
        return self.prev_quad, True, score
