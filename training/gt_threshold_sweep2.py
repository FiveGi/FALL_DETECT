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


def run(model, conf, imgsz=640):
    correct = 0
    fp_added = 0
    for path, name in valid:
        r = model.predict(path, verbose=False, conf=conf, classes=[0], imgsz=imgsz)[0]
        count = len(r.boxes) if r.boxes is not None else 0
        gt = gemini_counts[name]
        gt_is_one = gt == 1
        pred_is_one = count == 1
        if gt_is_one == pred_is_one:
            correct += 1
        if not gt_is_one and pred_is_one:
            fp_added += 1
    n = len(valid)
    return correct, n, fp_added


print("=== yolo26x.pt, threshold sweep, imgsz=640 ===")
model = YOLO("yolo26x.pt")
for conf in [0.5, 0.4, 0.35, 0.3, 0.25]:
    c, n, fp = run(model, conf)
    print(f"  conf={conf}: {c}/{n} ({c/n:.1%})  new_false_alone_alerts={fp}")

print("\n=== yolo26l.pt, imgsz=1280 (higher res), threshold sweep ===")
model = YOLO("yolo26l.pt")
for conf in [0.4, 0.35, 0.3, 0.25]:
    c, n, fp = run(model, conf, imgsz=1280)
    print(f"  conf={conf}: {c}/{n} ({c/n:.1%})  new_false_alone_alerts={fp}")
