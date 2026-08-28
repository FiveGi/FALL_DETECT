import os
import importlib.util
import cv2

spec = importlib.util.spec_from_file_location(
    "v3_fall_detection", os.path.join("..", "app", "detection", "v3_fall_detection.py")
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

detector = v3.V3PoseFallDetector(model_dir=os.path.join("..", "models"))
state = v3.V3MultiPersonFallState()
cap = cv2.VideoCapture(r"D:\project\PROJECT\Test\15.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_idx = 0
any_alerts = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = frame_idx / fps
    results = v3.detect_v3_fall_multi(frame, state, detector, config=None)
    any_detected = any(r[1] for r in results)
    if any_detected:
        top = max(results, key=lambda r: r[2])
        any_alerts.append((round(t, 2), round(top[2], 2), len(results)))
    frame_idx += 1
cap.release()
print("clip15 (multi-person production path) alert frames (t, prob, n_tracked):")
for a in any_alerts[:10]:
    print(" ", a)
print("total alert frames:", len(any_alerts))
