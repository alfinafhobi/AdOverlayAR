# AR Billboard (Step 1)

## Run
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m src.app

Keys:

q quit

d toggle denoise

s toggle sharpen

r toggle super-resolution (needs models/sr/EDSR_x2.pb)


---

### 8) (Optional) Super-Resolution weights
If you want SR right now, place a compatible OpenCV SR model at:
models/sr/EDSR_x2.pb
(We can switch to ESPCN or FSRCNN later if you prefer speed.)

---

## ✅ What you should see now
- A live camera window using your **system webcam**.
- Noticeably **cleaner** output with denoise + subtle sharpen.
- Toggle quality features in real time.

---

If this runs fine, say “**Step 2**” and I’ll add:
- `src/detect/billboard_detector.py` (YOLO wrapper, config-driven)
- `src/geo/homography.py` (quad refine + H)
- Integrate into `app.py` to draw the detected **billboard quad** (still no overlay).
