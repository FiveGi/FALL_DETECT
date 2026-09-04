import os
import glob
import json
import torch
from ultralytics import YOLO

torch.set_num_threads(4)

FRAMES_DIR = os.path.join(os.path.dirname(__file__), "data", "gt_frames")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
gemini_counts = json.load(open(os.path.join(DATA_DIR, "gt_gemini_counts.json")))

MODELS = ["yolo26m.pt", "yolo26l.pt"]
THRESHOLDS = [0.6, 0.5, 0.4, 0.35, 0.3, 0.25]

frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
frame_names = [os.path.basename(f) for f in frame_paths]
valid = [(p, n) for p, n in zip(frame_paths, frame_names) if gemini_counts.get(n, -1) >= 0]

for model_name in MODELS:
    model = YOLO(model_name)
    print(f"\n=== {model_name} ===")
    for conf in THRESHOLDS:
        correct = 0
        false_pos_added = 0  # frames where gt != 1 but pred became 1 (new false alarm risk)
        for path, name in valid:
            r = model.predict(path, verbose=False, conf=conf, classes=[0])[0]
            count = len(r.boxes) if r.boxes is not None else 0
            gt = gemini_counts[name]
            gt_is_one = gt == 1
            pred_is_one = count == 1
            if gt_is_one == pred_is_one:
                correct += 1
            if not gt_is_one and pred_is_one:
                false_pos_added += 1
        n = len(valid)
        print(f"  conf={conf}: binary_acc={correct}/{n} ({correct/n:.1%})  new_false_alone_alerts={false_pos_added}")
