"""Generate pseudo-labels for the fine-tuning frames using yolo26l (the best
candidate found in the ground-truth comparison), keeping only high-confidence
detections (conf>=0.5) as labels -- avoids teaching the fine-tuned model to
imitate the teacher's own low-confidence mistakes. Writes a standard
Ultralytics YOLO dataset (images/ + labels/ + data.yaml)."""
import os
import glob
import shutil
import random
import torch
from ultralytics import YOLO

torch.set_num_threads(4)

SRC_FRAMES = os.path.join(os.path.dirname(__file__), "data", "ft_frames")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "data", "ft_dataset")
PSEUDO_LABEL_CONF = 0.5
VAL_FRACTION = 0.15
SEED = 42

for split in ("train", "val"):
    os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

model = YOLO("yolo26l.pt")
frame_paths = sorted(glob.glob(os.path.join(SRC_FRAMES, "*.jpg")))
random.Random(SEED).shuffle(frame_paths)
n_val = max(1, int(len(frame_paths) * VAL_FRACTION))
val_set = set(frame_paths[:n_val])

n_with_person = 0
for fp in frame_paths:
    name = os.path.basename(fp)
    split = "val" if fp in val_set else "train"
    r = model.predict(fp, verbose=False, conf=PSEUDO_LABEL_CONF, classes=[0])[0]

    img_out = os.path.join(DATASET_DIR, "images", split, name)
    shutil.copy2(fp, img_out)

    label_out = os.path.join(DATASET_DIR, "labels", split, name.replace(".jpg", ".txt"))
    lines = []
    if r.boxes is not None and len(r.boxes) > 0:
        xywhn = r.boxes.xywhn.cpu().numpy()
        for x, y, w, h in xywhn:
            lines.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    if lines:
        n_with_person += 1
    with open(label_out, "w") as f:
        f.write("\n".join(lines))

print(f"{n_with_person}/{len(frame_paths)} frames got >=1 pseudo-label box")

yaml_content = f"""path: {DATASET_DIR}
train: images/train
val: images/val
names:
  0: person
"""
with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
    f.write(yaml_content)
print(f"Dataset written to {DATASET_DIR}")
