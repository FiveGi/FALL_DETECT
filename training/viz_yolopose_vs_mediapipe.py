import os
import cv2
import numpy as np
import importlib.util
from ultralytics import YOLO

_mod_path = os.path.join(os.path.dirname(__file__), "..", "app", "detection", "v3_fall_detection.py")
_spec = importlib.util.spec_from_file_location("v3_fall_detection", _mod_path)
_v3mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3mod)
V3PoseFallDetector = _v3mod.V3PoseFallDetector

ROOT = os.path.join(os.path.dirname(__file__), "..")
mp_detector = V3PoseFallDetector(os.path.join(ROOT, "models"))
yp_model = YOLO("yolo26s-pose.pt")

SKELETON = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]

def draw(frame, kpts17, color):
    out = frame.copy()
    for x, y in kpts17[:, :2]:
        if x > 0 and y > 0:
            cv2.circle(out, (int(x * out.shape[1]), int(y * out.shape[0])), 4, color, -1)
    for a, b in SKELETON:
        xa, ya = kpts17[a, :2]
        xb, yb = kpts17[b, :2]
        if xa > 0 and ya > 0 and xb > 0 and yb > 0:
            cv2.line(out, (int(xa*out.shape[1]), int(ya*out.shape[0])),
                      (int(xb*out.shape[1]), int(yb*out.shape[0])), color, 2)
    return out

def extract_frame(path, t):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

TEST_DIR = os.environ.get("TEST_DIR", r"D:\project\PROJECT\Test")
CASES = [
    (os.path.join(ROOT, "training", "data", "gmdcsa24_fall_raw", "s2_Fall_09.mp4"), 5.0, "s2_Fall_09_t5"),
    (os.path.join(ROOT, "training", "data", "gmdcsa24_fall_raw", "s3_Fall_16.mp4"), 5.5, "s3_Fall_16_t5.5"),
    (os.path.join(TEST_DIR, "15.mp4"), 7.0, "clip15_t7"),
]

out_dir = os.path.join(ROOT, "training", "data")
for path, t, label in CASES:
    frame = extract_frame(path, t)
    if frame is None:
        print(f"{label}: frame read failed")
        continue
    h, w = frame.shape[:2]

    mp_kpts, mp_found = mp_detector.extract_keypoints(frame)
    mp_viz = draw(frame, mp_kpts, (0, 0, 255)) if mp_found else frame.copy()
    cv2.putText(mp_viz, f"MediaPipe found={mp_found}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    r = yp_model.predict(frame, verbose=False, conf=0.5)[0]
    yp_viz = frame.copy()
    yp_found = r.keypoints is not None and len(r.keypoints.xy) > 0
    if yp_found:
        kxy = r.keypoints.xyn[0].cpu().numpy()  # normalized (17,2)
        kpts17 = np.zeros((17, 3), dtype=np.float32)
        kpts17[:, :2] = kxy
        yp_viz = draw(frame, kpts17, (0, 255, 0))
    cv2.putText(yp_viz, f"yolo26s-pose found={yp_found}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    combined = np.hstack([mp_viz, yp_viz])
    out_path = os.path.join(out_dir, f"posecmp_{label}.jpg")
    cv2.imwrite(out_path, combined)
    print(f"{label}: mp_found={mp_found} yolo_found={yp_found} -> {out_path}")
