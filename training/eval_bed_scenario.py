"""Specifically check: can the trained model tell "fell out of bed" (f_mask_b_*)
apart from "got in/out of bed normally" (nf_mask_b_*)? This is the exact false-alarm
risk the user asked about -- a person lying in bed looks similar at rest to a person
who has already fallen and is lying on the floor, so the model needs to key off the
motion pattern, not just the end position.
"""
import os
import glob
import numpy as np
import torch

from dataset import load_all_videos, make_windows, FallWindowDataset, WINDOW_SIZE
from model import FallClassifier

CKPT_PATH = os.path.join(os.path.dirname(__file__), "data", "best_model.pt")
FV_DIR = os.path.join(os.path.dirname(__file__), "data", "poses_fallvision")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = FallClassifier().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: val_f1={ckpt['val_f1']:.3f}\n")

    videos = load_all_videos(FV_DIR)
    bed_fall = [v for v in videos if v["label"] == 1 and v["subject"].startswith("fv_b")]
    bed_nofall = [v for v in videos if v["label"] == 0 and v["subject"].startswith("fv_b")]
    print(f"Bed FALL videos: {len(bed_fall)}")
    print(f"Bed NO-FALL (normal bed activity) videos: {len(bed_nofall)}")

    for name, group, expected in [("FALL from bed", bed_fall, 1), ("NORMAL bed activity", bed_nofall, 0)]:
        samples = make_windows(group)
        ds = FallWindowDataset(samples)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
        correct, total = 0, 0
        all_probs = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                probs = torch.sigmoid(model(x)).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                correct += (preds == expected).sum()
                total += len(preds)
                all_probs.extend(probs.tolist())
        acc = correct / total if total else 0
        print(f"\n{name}: {total} windows, {correct} predicted as expected ({acc*100:.1f}%)")
        print(f"  mean predicted fall-probability: {np.mean(all_probs):.3f}")


if __name__ == "__main__":
    main()
