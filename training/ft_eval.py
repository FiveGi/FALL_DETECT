import os
import glob
import json
import sys
from ultralytics import YOLO

FRAMES_DIR = os.path.join(os.path.dirname(__file__), "data", "gt_frames")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
gemini_counts = json.load(open(os.path.join(DATA_DIR, "gt_gemini_counts.json")))

weights = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    DATA_DIR, "ft_runs", "yolo26m_person_finetune", "weights", "best.pt")

frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
frame_names = [os.path.basename(p) for p in frame_paths]
valid = [(p, n) for p, n in zip(frame_paths, frame_names) if gemini_counts.get(n, -1) >= 0]
realistic = [(p, n) for p, n in valid if n.split("_")[0] in ("14", "15", "16")]

model = YOLO(weights)
print(f"=== {weights} ===")
for conf in [0.6, 0.5, 0.4, 0.35, 0.3, 0.25]:
    correct = 0
    correct_realistic = 0
    fp_added = 0
    for path, name in valid:
        r = model.predict(path, verbose=False, conf=conf, classes=[0])[0]
        count = len(r.boxes) if r.boxes is not None else 0
        gt = gemini_counts[name]
        match = (gt == 1) == (count == 1)
        correct += match
        if not (gt == 1) and (count == 1):
            fp_added += 1
        if name.split("_")[0] in ("14", "15", "16"):
            correct_realistic += match
    n = len(valid)
    nr = len(realistic)
    print(f"  conf={conf}: overall={correct}/{n} ({correct/n:.1%})  "
          f"realistic_domain={correct_realistic}/{nr} ({correct_realistic/nr:.1%})  new_false_alone={fp_added}")
