import cv2

def draw_text(img, text, x, y, scale=0.6, color=(255,255,255)):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def draw_hud(frame, fps=None, help_on=True):
    h, w = frame.shape[:2]
    if fps is not None:
        draw_text(frame, f"FPS: {fps:.1f}", 12, 28, 0.7, (0,255,0))
    if help_on:
        draw_text(frame, "q: quit | s: toggle sharpen | d: toggle denoise | r: toggle superres",
                  12, h-12, 0.55, (255,255,0))
