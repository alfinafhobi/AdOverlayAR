# src/app.py
import os
import yaml
import cv2
import numpy as np

from src.ingest.camera import Camera, QualityEnhancer
from src.ui.overlay_hud import draw_hud
from src.utils.timing import FPSTracker
from src.utils.async_worker import AsyncWorker

from src.detect.billboard_detector import BillboardDetector
from src.geo.homography import refine_quad_from_roi, bbox_to_quad, draw_quad
from src.track.tracker import QuadTracker, area_of_quad

# Overlay pipeline
from src.ads.asset_cache import AssetCache
from src.ads.decision_engine import DecisionEngine
from src.render.warper import warp_ad_to_quad
from src.render.compositor import match_brightness, alpha_blend
import src.render.warper as warper  # for runtime warp prefs

# ------------------------------ Paths ------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG_RUNTIME = os.path.join(ROOT, "config", "runtime.yaml")
CFG_DETECT  = os.path.join(ROOT, "config", "detector.yaml")
CFG_PLAY    = os.path.join(ROOT, "config", "playlists.yaml")

# --------------------------- Config utils --------------------------
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_opencv_perf(cfg):
    perf = cfg.get("perf", {})
    if perf.get("opencv_optimized", True):
        try: cv2.setUseOptimized(True)
        except Exception: pass
    if "num_threads" in perf:
        try: cv2.setNumThreads(int(perf["num_threads"]))
        except Exception: pass

# ------------------------- Geometry helpers ------------------------
def quad_to_bbox(q):
    xs, ys = q[:, 0], q[:, 1]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def pad_box(box, pad, w, h):
    x1, y1, x2, y2 = box
    return max(0, x1 - pad), max(0, y1 - pad), min(w - 1, x2 + pad), min(h - 1, y2 + pad)

def need_redetect(prev_quad, new_quad, area_jump_ratio):
    if prev_quad is None or new_quad is None:
        return True
    a0, a1 = area_of_quad(prev_quad), area_of_quad(new_quad)
    if a0 <= 1 or a1 <= 1:
        return True
    jump = abs(a1 - a0) / max(a0, 1.0)
    return jump > area_jump_ratio

def quad_aspect(quad):
    x1, y1, x2, y2 = quad_to_bbox(quad)
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    return float(w) / float(h)

# ------------------------------- Main ------------------------------
def main():
    # Load configs
    runtime_cfg  = load_yaml(CFG_RUNTIME) or {}
    detector_cfg = load_yaml(CFG_DETECT)  or {}
    refine_cfg   = runtime_cfg.get("refine", {}) or {}

    # Render settings
    render_cfg     = runtime_cfg.get("render", {})
    opacity        = float(render_cfg.get("opacity", 1.0))
    fit_mode       = str(render_cfg.get("fit_mode", "cover"))   # "cover" or "contain"
    feather        = int(render_cfg.get("feather", 3))
    brightness_mix = float(render_cfg.get("brightness_mix", 0.35))
    edge_boost     = float(render_cfg.get("edge_boost", 0.12))

    # Warper prefs (quality vs speed)
    HI_PREFS = {
        "canvas_scale": float(render_cfg.get("canvas_scale", 1.3)),
        "warp_interp":  str(render_cfg.get("warp_interp", "cubic")),
    }
    FAST_PREFS = {"canvas_scale": 1.05, "warp_interp": "linear"}
    warper._render_prefs_ = HI_PREFS

    # Perf settings
    set_opencv_perf(runtime_cfg)
    perf_cfg = runtime_cfg.get("perf", {})
    trk_cfg  = runtime_cfg.get("tracking", {})
    detect_interval = trk_cfg.get("detect_interval", detector_cfg.get("detect_interval", 24))

    # Camera
    cam_cfg = runtime_cfg.get("camera", {})
    backend = cam_cfg.get("backend", "dshow").lower()
    backend = "dshow" if (backend == "dshow" and os.name == "nt") else "cv2"

    cam = Camera(
        index=cam_cfg.get("index", 0),
        backend=backend,
        width=cam_cfg.get("width", 960),
        height=cam_cfg.get("height", 540),
        autofocus=cam_cfg.get("autofocus", True),
        auto_exposure=cam_cfg.get("auto_exposure", True),
    )
    try:
        cam.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cam.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    enhancer = QualityEnhancer(runtime_cfg)
    fps = FPSTracker()

    # Components
    det_model = BillboardDetector(detector_cfg)
    tracker   = QuadTracker(trk_cfg)
    assets    = AssetCache(CFG_PLAY)
    decider   = DecisionEngine(assets)

    # Async detector worker
    def detect_job(job):
        frame_bgr, roi_xyxy, scale = job["frame"], job["roi"], job["scale"]
        if roi_xyxy is not None:
            dets = det_model.detect_roi(frame_bgr, roi_xyxy, scale=scale)
            if len(dets) == 0:
                dets = det_model.detect_full(frame_bgr, scale=scale)
        else:
            dets = det_model.detect_full(frame_bgr, scale=scale)
        return {"dets": dets}

    worker = None
    if perf_cfg.get("detector_async", True):
        worker = AsyncWorker(detect_job, max_queue=int(perf_cfg.get("detector_queue", 1)))
        worker.start()

    # State
    frame_idx = 0
    quad, prev_quad, last_det = None, None, None

    # Lock gating: overlay only when locked
    locked = False
    locked_quad = None
    locked_fail = 0
    pending_request = False

    print("[INFO] Keys: q quit | d denoise | s sharpen | r superres | b lock-from-detect | l unlock | f refine-now ")

    while True:
        fps.tick()
        frame = cam.read()
        frame = enhancer.process(frame)
        if frame is None:
            continue

        h, w = frame.shape[:2]
        scale   = float(perf_cfg.get("downscale_for_detect", 0.6))
        use_roi = bool(perf_cfg.get("use_roi_for_detect", False))
        roi_pad = int(perf_cfg.get("roi_pad", 140))

        # 1) Schedule detection ONLY when NOT locked
        if (frame_idx % detect_interval == 0) and worker and not pending_request and not locked:
            roi = pad_box(quad_to_bbox(quad), roi_pad, w, h) if (use_roi and quad is not None) else None
            worker.submit({"frame": frame, "roi": roi, "scale": scale})
            pending_request = True

        # 2) Consume detection (used to display yellow box or to lock)
        if worker:
            out = worker.latest_result()
            if out and "dets" in out:
                dets = out["dets"]
                pending_request = False
                last_det = dets[0] if len(dets) > 0 else None
                if last_det:
                    print("[DBG] YOLO dets=1 conf>0.15")
                else:
                    print("[DBG] YOLO dets=0 conf>0.15")

        # 3) Tracking (only when locked)
        if locked and locked_quad is not None:
            new_quad, ok, _ = tracker.track(frame)
            if ok and new_quad is not None:
                quad = new_quad
                locked_quad = new_quad.copy()
                locked_fail = 0
            else:
                locked_fail += 1
                quad = locked_quad.copy()
                # Opportunistic local re-detect to help when LK drifts
                if locked_fail % 3 == 0:
                    x1, y1, x2, y2 = quad_to_bbox(locked_quad)
                    roi = pad_box((x1, y1, x2, y2), 120, w, h)
                    dets = det_model.detect_roi(frame, roi, scale=0.6)
                    if dets:
                        dx1, dy1, dx2, dy2 = dets[0].box
                        exp_ar = float(max(1, dx2 - dx1)) / float(max(1, dy2 - dy1))
                        q = refine_quad_from_roi(frame, (dx1, dy1, dx2, dy2), exp_ar, refine_cfg)
                        if q is None:
                            q = bbox_to_quad(dx1, dy1, dx2, dy2)
                        quad = q
                        locked_quad = q.copy()
                        tracker.init(frame, quad)
                        locked_fail = 0

            # Periodic refine to keep size/alignment perfect while locked
            if quad is not None and (refine_cfg.get("enable", True)):
                every_n = int(refine_cfg.get("every_n", 10))
                pad     = int(refine_cfg.get("pad", 40))
                if every_n > 0 and (frame_idx % every_n == 0):
                    x1 = int(quad[:, 0].min()) - pad
                    y1 = int(quad[:, 1].min()) - pad
                    x2 = int(quad[:, 0].max()) + pad
                    y2 = int(quad[:, 1].max()) + pad
                    # clip to frame
                    x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
                    w_bb = max(1, x2 - x1); h_bb = max(1, y2 - y1)
                    exp_ar = float(w_bb) / float(h_bb)
                    rq = refine_quad_from_roi(frame, (x1, y1, x2, y2), exp_ar, refine_cfg)
                    if rq is not None:
                        quad = (0.6 * quad + 0.4 * rq).astype(np.float32)  # gentle snap
                        locked_quad = quad.copy()
                        tracker.init(frame, quad)

        # 4) Visualization:
        #    - If NOT locked: draw ONLY detection rectangle (no overlay).
        #    - If locked: draw overlay on the tracked/refined quad.
        if not locked:
            if last_det is not None:
                x1, y1, x2, y2 = last_det.box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(frame, "detect", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, "detect", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
        else:
            # Choose speed/quality profile
            use_fast = ((frame_idx % 3) != 0)
            warper._render_prefs_ = FAST_PREFS if use_fast else HI_PREFS
            local_mix = min(0.35, brightness_mix) if use_fast else brightness_mix
            local_feather = max(2, feather - 1) if use_fast else feather

            if quad is not None:
                target_ar = quad_aspect(quad)
                ad_img, _ = decider.choose_image(quad)  # keep engine unchanged
                if ad_img is not None:
                    try:
                        warped, mask = warp_ad_to_quad(
                            ad_img, quad, frame.shape,
                            target_aspect=target_ar,   # helps fit consistency
                            fit_mode=fit_mode,         # "cover" fills quad fully
                            feather=local_feather
                        )
                        if warped is None or mask is None:
                            raise RuntimeError("warp returned None")
                        adjusted = match_brightness(warped, frame, mask, mix=local_mix)
                        frame = alpha_blend(frame, adjusted, mask, opacity=opacity, edge_boost=edge_boost)
                    except Exception:
                        # fallback paste if warp fails
                        x1 = int(quad[:, 0].min()); y1 = int(quad[:, 1].min())
                        x2 = int(quad[:, 0].max()); y2 = int(quad[:, 1].max())
                        bx, by = max(1, x2 - x1), max(1, y2 - y1)
                        ad_resized = cv2.resize(ad_img, (bx, by), interpolation=cv2.INTER_AREA)
                        frame[y1:y2, x1:x2] = ad_resized
                draw_quad(frame, quad, (30, 255, 30), 2)

        # HUD + keys
        if runtime_cfg.get("hud", {}).get("show_fps", True):
            draw_hud(frame, fps=fps.fps, help_on=runtime_cfg.get("hud", {}).get("show_help", True))

        cv2.imshow("AR-Billboard (Lock to Overlay)", frame)
        key = cv2.waitKey(1) & 0xFF

        # ---- Key handling ----
        if key == ord("q"):
            break

        # Lock from current detection (start overlay)
        elif key == ord("b"):
            if last_det is not None:
                x1, y1, x2, y2 = last_det.box
                expected_aspect = float(max(1, x2 - x1)) / float(max(1, y2 - y1))
                q = refine_quad_from_roi(frame, (x1, y1, x2, y2), expected_aspect, refine_cfg)
                if q is None:
                    q = bbox_to_quad(x1, y1, x2, y2)
                    print("[LOCK] Refine failed; using bbox.")
                else:
                    print("[LOCK] Refine OK; using precise quad.")
                quad = q
                tracker.init(frame, quad)
                locked = True
                locked_quad = quad.copy()
                pending_request = False
                print("[LOCK] Locked; overlay enabled.")
            else:
                print("[LOCK] No detection available; keep the board in view.")

        # Unlock (stop overlay and resume detection)
        elif key == ord("l"):
            if locked:
                locked = False
                locked_quad = None
                quad = None
                pending_request = False
                print("[LOCK] Unlocked; overlay disabled; resuming detection.")

        # Optional manual refine (while unlocked) to preview a clean quad
        elif key == ord("f") and not locked and last_det is not None:
            x1, y1, x2, y2 = last_det.box
            expected_aspect = float(max(1, x2 - x1)) / float(max(1, y2 - y1))
            q = refine_quad_from_roi(frame, (x1, y1, x2, y2), expected_aspect, refine_cfg)
            if q is not None:
                quad = q
                tracker.init(frame, quad)
                print("[REFINE] Quad refined; press 'b' to lock.")

        frame_idx += 1

    if worker: worker.stop()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
