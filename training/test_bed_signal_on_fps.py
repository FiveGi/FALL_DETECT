"""
Test whether YOLO bed/couch/chair detection at the exact alert frame/timestamp
would correctly flag "bed nearby" for each of the 6 known GMDCSA24-val false positives
of the currently deployed model (ss28_seed42_without.pt).

Alert list (from latest eval_v3_on_gmdcsa24_val.py run):
  s1_ADL_01: 2.0
  s2_ADL_03: 9.4
  s2_ADL_15: 6.6
  s4_ADL_07: 1.3
  s4_ADL_08: 1.3, 6.7
  s4_ADL_10: 4.7
"""
import os
import cv2
from ultralytics import YOLO

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_adl_raw_val")

ALERTS = {
    "s1_ADL_01": [2.0],
    "s2_ADL_03": [9.4],
    "s2_ADL_15": [6.6],
    "s4_ADL_07": [1.3],
    "s4_ADL_08": [1.3, 6.7],
    "s4_ADL_10": [4.7],
}

FURNITURE_CLASSES = {56: "chair", 57: "couch", 59: "bed"}

model = YOLO("yolov10x.pt")

for clip, timestamps in ALERTS.items():
    path = os.path.join(RAW_DIR, f"{clip}.mp4")
    if not os.path.exists(path):
        print(f"{clip}: MISSING FILE {path}")
        continue
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for ts in timestamps:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"{clip} @ {ts}s: FRAME READ FAILED")
            continue
        r = model.predict(frame, verbose=False, conf=0.15)[0]
        dets = []
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            dets.append((name, round(conf, 2)))
        furniture_hits = [d for d in dets if d[0] in ("bed", "couch", "chair")]
        print(f"{clip} @ {ts}s: furniture={furniture_hits}  all={dets}")
    cap.release()
