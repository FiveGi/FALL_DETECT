"""Targeted check: can the model tell a real bed-fall apart from someone just
getting into/out of bed normally? Breaks down accuracy specifically on the
FallVision bed-scenario clips (f_mask_b_* = real falls from bed, nf_mask_b_* =
normal bed activity, no fall).
"""
import os
import torch
from torch.utils.data import DataLoader

from dataset import load_all_videos, make_windows, FallWindowDataset
from model import FallClassifier

POSE_DIR_FV = os.path.join(os.path.dirname(__file__), "data", "poses_fallvision")
CKPT_PATH = os.path.join(os.path.dirname(__file__), "data", "best_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = FallClassifier().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: val_f1={ckpt['val_f1']:.3f}")

    videos = load_all_videos(POSE_DIR_FV)
    bed_videos = [v for v in videos if "mask_b_" in v["name"]]
    print(f"Bed-scenario videos: {len(bed_videos)}")

    fall_bed = [v for v in bed_videos if v["label"] == 1]
    nofall_bed = [v for v in bed_videos if v["label"] == 0]
    print(f"  f_mask_b (real bed falls): {len(fall_bed)}")
    print(f"  nf_mask_b (normal bed activity, no fall): {len(nofall_bed)}")

    for group_name, group in [("f_mask_b (should predict FALL)", fall_bed),
                               ("nf_mask_b (should predict NO FALL)", nofall_bed)]:
        samples = make_windows(group)
        loader = DataLoader(FallWindowDataset(samples), batch_size=64, shuffle=False)
        correct, total = 0, 0
        pred_fall_count = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                logits = model(x)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu()
                correct += (preds == y).sum().item()
                pred_fall_count += preds.sum().item()
                total += y.size(0)
        print(f"\n{group_name}")
        print(f"  windows: {total}, predicted FALL: {pred_fall_count} ({100*pred_fall_count/total:.1f}%)")
        print(f"  window-level accuracy vs this group's window labels: {100*correct/total:.1f}%")

    # Per-video verdict: does the clip's peak fall-probability window cross the fall threshold?
    print("\n=== Per-video verdict (does any window in the clip get flagged as FALL?) ===")
    for group_name, group in [("f_mask_b (real falls)", fall_bed), ("nf_mask_b (normal activity)", nofall_bed)]:
        flagged = 0
        for v in group:
            samples = make_windows([v])
            x = torch.stack([torch.from_numpy(s[0]) for s in samples]).to(DEVICE)
            with torch.no_grad():
                probs = torch.sigmoid(model(x))
            if (probs > 0.5).any().item():
                flagged += 1
        print(f"{group_name}: {flagged}/{len(group)} clips had at least one FALL-flagged window ({100*flagged/len(group):.1f}%)")


if __name__ == "__main__":
    main()
