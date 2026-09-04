"""What actually matters for alone-detection isn't exact headcount in a crowd --
it's correctly classifying 'exactly 1 person' (the real trigger condition in
camera_manager.py: `if person_count == 1 and not fall_detected`). Recompute
accuracy on that binary classification across ALL 77 frames including the
crowd scene, since 'not exactly 1' is the correct call there regardless of the
precise headcount."""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
yolo_counts = json.load(open(os.path.join(DATA_DIR, "gt_yolo_counts.json")))
gemini_counts = json.load(open(os.path.join(DATA_DIR, "gt_gemini_counts.json")))

MODELS = ["yolov10x.pt", "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
frames = sorted(f for f in yolo_counts.keys() if gemini_counts.get(f, -1) >= 0)

print(f"{len(frames)} frames total\n=== 'Is exactly 1 person present' binary accuracy ===")
for model in MODELS:
    correct = 0
    for f in frames:
        gt_is_one = gemini_counts[f] == 1
        pred_is_one = yolo_counts[f][model] == 1
        if gt_is_one == pred_is_one:
            correct += 1
    print(f"  {model:15s}: {correct}/{len(frames)} ({correct/len(frames):.1%})")
