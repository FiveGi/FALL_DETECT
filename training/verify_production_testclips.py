import os
import importlib.util
import cv2

spec = importlib.util.spec_from_file_location(
    "v3_fall_detection", os.path.join("..", "app", "detection", "v3_fall_detection.py")
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

detector = v3.V3PoseFallDetector(model_dir=os.path.join("..", "models"))
TEST_DIR = r"D:\project\PROJECT\Test"

for clip in [f"{i}.mp4" for i in range(1, 18)]:
    path = os.path.join(TEST_DIR, clip)
    if not os.path.exists(path):
        continue
    state = v3.V3FallDetectionState()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    alerts = []
    last_label = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        detected, prob, label, _ = v3.detect_v3_fall(frame, state, detector, config=None)
        if label != last_label:
            if label == "fall":
                alerts.append((round(t, 2), round(prob, 2)))
            last_label = label
        frame_idx += 1
    cap.release()
    marker = "ALERT" if alerts else "no alert"
    print(f"{clip}: {marker}  alerts={alerts}")
