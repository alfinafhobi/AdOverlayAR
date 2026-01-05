AR Billboard 

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



##  What you should see now
- A live camera window using your **system webcam**.
- Noticeably **cleaner** output with denoise + subtle sharpen.
- Toggle quality features in real time.



