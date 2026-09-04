import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
yolo_counts = json.load(open(os.path.join(DATA_DIR, "gt_yolo_counts.json")))
gemini_counts = json.load(open(os.path.join(DATA_DIR, "gt_gemini_counts.json")))

MODELS = ["yolov10x.pt", "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]

frames = sorted(yolo_counts.keys())
gt_valid_frames = [f for f in frames if gemini_counts.get(f, -1) >= 0]
print(f"{len(gt_valid_frames)}/{len(frames)} frames have valid Gemini ground truth\n")

# Separate the crowd-scene clip (16) since exact-count is nearly meaningless there --
# report it separately rather than letting it dominate the "typical scene" accuracy.
normal_frames = [f for f in gt_valid_frames if not f.startswith("16_")]
crowd_frames = [f for f in gt_valid_frames if f.startswith("16_")]

print(f"=== Normal scenes (0-3 people, {len(normal_frames)} frames) ===")
for model in MODELS:
    exact = 0
    within1 = 0
    abs_err = 0
    for f in normal_frames:
        gt = gemini_counts[f]
        pred = yolo_counts[f][model]
        if pred == gt:
            exact += 1
        if abs(pred - gt) <= 1:
            within1 += 1
        abs_err += abs(pred - gt)
    n = len(normal_frames)
    print(f"  {model:15s}: exact={exact}/{n} ({exact/n:.1%})  within-1={within1}/{n} ({within1/n:.1%})  "
          f"mean_abs_err={abs_err/n:.2f}")

print(f"\n=== Crowd scene (clip16, {len(crowd_frames)} frames, GT 12-18 people) ===")
for model in MODELS:
    diffs = [(yolo_counts[f][model], gemini_counts[f]) for f in crowd_frames]
    abs_err = sum(abs(p - g) for p, g in diffs) / len(diffs)
    avg_pred = sum(p for p, g in diffs) / len(diffs)
    avg_gt = sum(g for p, g in diffs) / len(diffs)
    print(f"  {model:15s}: avg_predicted={avg_pred:.1f}  avg_gemini_gt={avg_gt:.1f}  mean_abs_err={abs_err:.1f}")
