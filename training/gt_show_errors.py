import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
yolo_counts = json.load(open(os.path.join(DATA_DIR, "gt_yolo_counts.json")))
gemini_counts = json.load(open(os.path.join(DATA_DIR, "gt_gemini_counts.json")))

model = "yolo26l.pt"
frames = sorted(f for f in yolo_counts.keys() if gemini_counts.get(f, -1) >= 0)

print(f"Errors for {model} (binary 'exactly 1 person' mismatches):")
for f in frames:
    gt = gemini_counts[f]
    pred = yolo_counts[f][model]
    gt_is_one = gt == 1
    pred_is_one = pred == 1
    if gt_is_one != pred_is_one:
        print(f"  {f}: gemini_gt={gt} yolo_pred={pred}")
