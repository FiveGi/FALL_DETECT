import os
import glob
import json
import torch
from ultralytics import YOLO

torch.set_num_threads(4)

FRAMES_DIR = os.path.join(os.path.dirname(__file__), "data", "gt_frames")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
gemini_counts = json.load(open(os.path.join(DATA_DIR, "gt_gemini_counts.json")))

frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
frame_names = [os.path.basename(f) for f in frame_paths]
valid = [(p, n) for p, n in zip(frame_paths, frame_names) if gemini_counts.get(n, -1) >= 0]

model = YOLO("yolo26l.pt")
print("Errors at conf=0.35:")
for path, name in valid:
    r = model.predict(path, verbose=False, conf=0.35, classes=[0])[0]
    count = len(r.boxes) if r.boxes is not None else 0
    gt = gemini_counts[name]
    if (gt == 1) != (count == 1):
        print(f"  {name}: gemini_gt={gt} yolo_pred={count}")
