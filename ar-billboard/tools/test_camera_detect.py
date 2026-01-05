import cv2, yaml, os
from src.detect.billboard_detector import BillboardDetector

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG_DET = os.path.join(ROOT, "config", "detector.yaml")
det_cfg = yaml.safe_load(open(CFG_DET, "r", encoding="utf-8"))
det = BillboardDetector(det_cfg)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
print("[INFO] Press q to quit")

while True:
    ok, frame = cap.read()
    if not ok: break
    dets = det.detect_full(frame, scale=0.75)
    if dets:
        x1,y1,x2,y2 = dets[0].box
        label = f"{dets[0].class_name}:{dets[0].score:.2f}"
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
        cv2.putText(frame,label,(x1,max(0,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(frame,label,(x1,max(0,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),1,cv2.LINE_AA)
    else:
        cv2.putText(frame,"NO DET",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(frame,"NO DET",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2,cv2.LINE_AA)
    cv2.imshow("cam-detect", frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release(); cv2.destroyAllWindows()
