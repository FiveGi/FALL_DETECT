import os
import cv2
import numpy as np
from ultralytics import YOLO

ROOT = os.path.join(os.path.dirname(__file__), "..")
yp_model = YOLO("yolo26s-pose.pt")

SKELETON = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]
COLORS = [(0,255,0), (255,128,0), (0,200,255), (255,0,255)]

def draw_all(frame, kxy_list):
    out = frame.copy()
    for i, kxy in enumerate(kxy_list):
        color = COLORS[i % len(COLORS)]
        for x, y in kxy:
            if x > 0 and y > 0:
                cv2.circle(out, (int(x), int(y)), 4, color, -1)
        for a, b in SKELETON:
            xa, ya = kxy[a]
            xb, yb = kxy[b]
            if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)
    return out

cap = cv2.VideoCapture(r"D:\project\PROJECT\Test\15.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
cap.set(cv2.CAP_PROP_POS_FRAMES, int(7.0 * fps))
ok, frame = cap.read()
cap.release()

r = yp_model.predict(frame, verbose=False, conf=0.5)[0]
n = len(r.keypoints.xy) if r.keypoints is not None else 0
print(f"yolo26s-pose detected {n} people in this frame")
kxy_list = [r.keypoints.xy[i].cpu().numpy() for i in range(n)]
out = draw_all(frame, kxy_list)
out_path = os.path.join(ROOT, "training", "data", "posecmp_clip15_ALLPEOPLE.jpg")
cv2.imwrite(out_path, out)
print("saved", out_path)
