"""Run all 6 YOLO variants (yolov10x, yolo26 n/s/m/l/x) on the sampled ground-truth
frames, recording each one's person count (conf=0.6, matching production's
V2PersonDetector threshold exactly)."""
import os
import glob
import json
import torch
from ultralytics import YOLO

torch.set_num_threads(4)

FRAMES_DIR = os.path.join(os.path.dirname(__file__), "data", "gt_frames")
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "gt_yolo_counts.json")
CONF = 0.6

MODELS = ["yolov10x.pt", "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]

frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
print(f"{len(frame_paths)} frames to process across {len(MODELS)} models")

results = {}  # frame_name -> {model: count}
for model_name in MODELS:
    model = YOLO(model_name)
    for fp in frame_paths:
        name = os.path.basename(fp)
        r = model.predict(fp, verbose=False, conf=CONF, classes=[0])[0]
        count = len(r.boxes) if r.boxes is not None else 0
        results.setdefault(name, {})[model_name] = count
    print(f"  {model_name}: done")

json.dump(results, open(OUT_PATH, "w"), indent=2)
print(f"Saved to {OUT_PATH}")
