"""Can ONNX export close the gap? The .pt path runs through PyTorch's CPU kernels; ONNX
Runtime's graph optimisation + fused kernels are usually meaningfully faster for the same
weights, and it is already a dependency here (the fall classifier runs on it)."""
import os, time, statistics
import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture('training/data/gmdcsa24_fall_raw/s1_Fall_01.mp4')
frames = []
while len(frames) < 20:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)

for name in ['yolo26s-pose.pt', 'yolo26n-pose.pt']:
    src = os.path.join('models', name) if os.path.exists(os.path.join('models', name)) else name
    for imgsz in [320, 448]:
        out = os.path.join('models', f'{os.path.basename(name)[:-3]}_{imgsz}.onnx')
        try:
            if not os.path.exists(out):
                p = YOLO(src).export(format='onnx', imgsz=imgsz, simplify=True, dynamic=False)
                import shutil; shutil.move(p, out)
            m = YOLO(out, task='pose')
            m.predict(frames[0], verbose=False, conf=0.5, classes=[0], imgsz=imgsz)
            t = []
            for f in frames:
                s = time.perf_counter()
                m.predict(f, verbose=False, conf=0.5, classes=[0], imgsz=imgsz)
                t.append((time.perf_counter() - s) * 1000)
            mean = statistics.mean(t)
            print(f'ONNX {name:18} imgsz={imgsz:4} {mean:7.1f} ms -> {1000/mean:5.1f} fps')
        except Exception as e:
            print(f'ONNX {name:18} imgsz={imgsz:4} FAILED: {str(e)[:100]}')
