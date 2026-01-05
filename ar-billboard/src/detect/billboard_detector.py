# src/detect/billboard_detector.py
import os
from collections import namedtuple
from typing import List, Optional, Tuple

import cv2
import numpy as np

Det = namedtuple("Det", "box score cls class_name")  # (x1,y1,x2,y2), conf, class_id, readable name


def _largest_rect_bbox(bgr: np.ndarray) -> Optional[Det]:
    """
    Fallback: find the largest convex 4-point contour and return its bbox.
    Works well for screens/boards when YOLO misses in bad light.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    e = cv2.Canny(g, 60, 160)
    e = cv2.dilate(e, None, iterations=1)
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    h, w = g.shape
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, ww, hh = cv2.boundingRect(approx)
            if ww < 40 or hh < 40 or ww >= w or hh >= h:
                continue
            area = ww * hh
            if area > best_area:
                best = (x, y, x + ww, y + hh)
                best_area = area
    if best is None:
        return None
    return Det(best, 0.15, -1, "fallback-rect")


class BillboardDetector:
    """
    YOLOv8 (Ultralytics) Torch backend with a robust rectangle fallback.
    Config keys read from detector.yaml:
      - backend: "torch"
      - weights_path: "yolov8n.pt"
      - img_size: 768
      - conf_thres: 0.15
      - iou_nms: 0.45
      - classes: [62,63,67]  # tv, laptop, cell phone (COCO)
      - max_detections: 1
      - device: "auto" | "cuda" | "cpu"
      - half: false
      - agnostic_nms: true
    """

    _COCO_NAMES = [
        # 0..79 coco names
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
        "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
        "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
        "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
        "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
        "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake",
        "chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
        "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock",
        "vase","scissors","teddy bear","hair drier","toothbrush"
    ]

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.backend = self.cfg.get("backend", "torch")
        self.weights_path = self.cfg.get("weights_path", "yolov8n.pt")
        self.img_size = int(self.cfg.get("img_size", 768))
        self.conf_thres = float(self.cfg.get("conf_thres", 0.15))
        self.iou_nms = float(self.cfg.get("iou_nms", 0.45))
        self.max_det = int(self.cfg.get("max_detections", 1))
        self.device = str(self.cfg.get("device", "auto"))
        self.half = bool(self.cfg.get("half", False))
        self.agnostic_nms = bool(self.cfg.get("agnostic_nms", True))
        # class filter
        cls_list = self.cfg.get("classes", [])
        self.class_filter = [int(c) for c in cls_list] if isinstance(cls_list, list) else []

        # lazy-load YOLO
        self._yolo = None
        self._yolo_loaded = False

    # ------------------------- public API -------------------------
    def detect_full(self, frame_bgr: np.ndarray, scale: float = 1.0) -> List[Det]:
        dets = self._yolo_detect(frame_bgr, scale=scale, offset=(0, 0))
        print(f"[DBG] YOLO dets={len(dets)} conf>{self.conf_thres}")

        if len(dets) == 0:
            fb = _largest_rect_bbox(frame_bgr)
            if fb:
                dets = [fb]
        return self._limit_and_sort(dets)

    def detect_roi(self, frame_bgr: np.ndarray, roi_xyxy: Tuple[int, int, int, int], scale: float = 1.0) -> List[Det]:
        x1, y1, x2, y2 = [int(v) for v in roi_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_bgr.shape[1] - 1, x2), min(frame_bgr.shape[0] - 1, y2)
        crop = frame_bgr[y1:y2, x1:x2]
        dets = self._yolo_detect(crop, scale=scale, offset=(x1, y1))
        if len(dets) == 0:
            fb = _largest_rect_bbox(crop)
            if fb:
                bx1, by1, bx2, by2 = fb.box
                dets = [Det((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1), fb.score, fb.cls, fb.class_name)]
        return self._limit_and_sort(dets)

    # ------------------------ internal YOLO -----------------------
    def _ensure_yolo(self):
        if self._yolo_loaded:
            return
        if self.backend != "torch":
            raise RuntimeError("Only 'torch' YOLO backend is implemented in this file.")
        try:
            from ultralytics import YOLO  # requires ultralytics in requirements.txt
        except Exception as e:
            raise RuntimeError(
                "Ultralytics not available. Add 'ultralytics' to requirements.txt and install."
            ) from e

        if not os.path.exists(self.weights_path):
            # Ultralytics will auto-download official weights if you pass the filename
            pass
        self._yolo = YOLO(self.weights_path)
        self._yolo_loaded = True

    def _class_name(self, cid: int) -> str:
        if cid < 0 or cid >= len(self._COCO_NAMES):
            return f"id{cid}"
        return self._COCO_NAMES[cid]

    def _yolo_detect(
        self, bgr: np.ndarray, scale: float = 1.0, offset: Tuple[int, int] = (0, 0)
    ) -> List[Det]:
        self._ensure_yolo()

        # NOTE: ultralytics handles resizing/letterbox internally via imgsz
        imgsz = int(self.img_size * float(scale))
        # device/half handled by Ultralytics internally; half only meaningful on CUDA
        kwargs = dict(
            imgsz=imgsz,
            conf=self.conf_thres,
            iou=self.iou_nms,
            device=self.device,
            half=self.half if self.device in ("cuda", "auto") else False,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
            max_det=self.max_det if self.max_det > 0 else 300,
        )
        res = self._yolo.predict(bgr, **kwargs)
        dets: List[Det] = []

        if not res or len(res) == 0:
            return dets

        r0 = res[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return dets

        xyxy = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = r0.boxes.conf.cpu().numpy().astype(np.float32)
        cls  = r0.boxes.cls.cpu().numpy().astype(np.int32)

        # Filter classes if requested
        for i in range(xyxy.shape[0]):
            cid = int(cls[i])
            if self.class_filter and (cid not in self.class_filter):
                continue
            x1, y1, x2, y2 = xyxy[i]
            score = float(conf[i])
            ox, oy = offset
            box = (int(x1) + ox, int(y1) + oy, int(x2) + ox, int(y2) + oy)
            dets.append(Det(box, score, cid, self._class_name(cid)))

        return dets

    def _limit_and_sort(self, dets: List[Det]) -> List[Det]:
        if not dets:
            return dets
        # sort by score * area (favor bigger confident boards)
        def score_area(d: Det):
            x1, y1, x2, y2 = d.box
            return d.score * max(1, (x2 - x1) * (y2 - y1))
        dets = sorted(dets, key=score_area, reverse=True)
        if self.max_det > 0:
            dets = dets[: self.max_det]
        return dets
