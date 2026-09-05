"""Run the deployed production pipeline (both single- and multi-person entry points)
over every Test/ clip, save alert timestamps+probabilities to JSON for Gemini
verification (mirrors verify_current_model_alerts.py's methodology)."""
import os
import json
import importlib.util
import cv2

spec = importlib.util.spec_from_file_location(
    "v3_fall_detection", os.path.join("..", "app", "detection", "v3_fall_detection.py")
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

TEST_DIR = os.environ.get("TEST_DIR", r"D:\project\PROJECT\Test")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")

detector = v3.V3PoseFallDetector(model_dir=os.path.join("..", "models"))

single_alerts = {}
multi_alerts = {}

for i in range(1, 18):
    clip = f"{i}.mp4"
    path = os.path.join(TEST_DIR, clip)
    if not os.path.exists(path):
        continue

    # single-person path
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
                alerts.append((t, prob))
            last_label = label
        frame_idx += 1
    cap.release()
    single_alerts[str(i)] = alerts
    print(f"{clip} single: {len(alerts)} alerts")

    # multi-person path
    mstate = v3.V3MultiPersonFallState()
    cap = cv2.VideoCapture(path)
    frame_idx = 0
    malerts = []
    last_any = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        results = v3.detect_v3_fall_multi(frame, mstate, detector, config=None)
        any_detected = any(r[1] for r in results)
        if any_detected and not last_any:
            top = max(results, key=lambda r: r[2])
            malerts.append((t, top[2]))
        last_any = any_detected
        frame_idx += 1
    cap.release()
    multi_alerts[str(i)] = malerts
    print(f"{clip} multi: {len(malerts)} alerts")

json.dump(single_alerts, open(os.path.join(OUT_DIR, "prod_single_alerts.json"), "w"), indent=2)
json.dump(multi_alerts, open(os.path.join(OUT_DIR, "prod_multi_alerts.json"), "w"), indent=2)
print("\nSaved prod_single_alerts.json and prod_multi_alerts.json")
