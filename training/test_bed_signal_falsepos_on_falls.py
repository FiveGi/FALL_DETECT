"""
Safety check: does YOLO bed/couch detection spuriously fire during genuine
GMDCSA24-val FALL clips (especially in the last 2s, where the person is lying
still on the floor after falling - the exact phase a bed-suppression signal
must NOT touch)? Samples every 0.5s across each clip.
"""
import os
import cv2
from ultralytics import YOLO

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_fall_raw")

VAL_FALL = ["s1_Fall_02", "s1_Fall_06", "s1_Fall_16", "s2_Fall_02", "s2_Fall_04",
            "s2_Fall_09", "s2_Fall_14", "s2_Fall_20", "s3_Fall_02", "s3_Fall_09",
            "s3_Fall_12", "s3_Fall_13", "s3_Fall_16", "s4_Fall_06", "s4_Fall_17"]

model = YOLO("yolov10x.pt")

any_hits = False
for clip in VAL_FALL:
    path = os.path.join(RAW_DIR, f"{clip}.mp4")
    if not os.path.exists(path):
        print(f"{clip}: MISSING")
        continue
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    hits = []
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, verbose=False, conf=0.15)[0]
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            if name in ("bed", "couch"):
                conf = float(box.conf[0])
                hits.append((round(t, 1), name, round(conf, 2)))
        t += 0.5
    cap.release()
    last2s_hits = [h for h in hits if h[0] >= duration - 2.0]
    flag = " <-- BED/COUCH NEAR END" if last2s_hits else ""
    print(f"{clip} (dur={duration:.1f}s): all_hits={hits}{flag}")
    if hits:
        any_hits = True

print("\n=== SUMMARY ===")
print("Any bed/couch detections across all 15 val fall clips:", any_hits)
