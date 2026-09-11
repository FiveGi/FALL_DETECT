import os, time, statistics, glob
import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture('training/data/gmdcsa24_fall_raw/s1_Fall_01.mp4')
frames = []
while len(frames) < 20:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)

for name in ['yolo26s-pose.pt', 'yolo26n-pose.pt', 'yolo11n-pose.pt']:
    for imgsz in [640, 448, 320]:
        try:
            m = YOLO(os.path.join('models', name)) if os.path.exists(os.path.join('models', name)) else YOLO(name)
            m.predict(frames[0], verbose=False, conf=0.5, classes=[0], imgsz=imgsz)
            t = []
            for f in frames:
                s = time.perf_counter()
                m.predict(f, verbose=False, conf=0.5, classes=[0], imgsz=imgsz)
                t.append((time.perf_counter() - s) * 1000)
            mean = statistics.mean(t)
            print(f'{name:18} imgsz={imgsz:4}  {mean:7.1f} ms  -> {1000/mean:5.1f} fps')
        except Exception as e:
            print(f'{name:18} imgsz={imgsz:4}  FAILED: {str(e)[:80]}')
