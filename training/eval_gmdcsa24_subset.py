"""Compare baseline vs Gemini-relabeled checkpoints, evaluated ONLY on the GMDCSA24
validation videos (not the full pooled val set), to isolate whether the label fix
helped where it was actually applied -- the pooled metric is dominated by FallVision
(5845 of ~6100 pooled videos), so a small, real GMDCSA24-only improvement could be
invisible in the pooled number.

Both runs used the same seed=42 stratified split over identically-sized POSE_DIRS
(only the GMDCSA24 directory's *content* differs between runs, not video count), so
split_videos() assigns the same specific GMDCSA24 videos to val in both cases --
this is an apples-to-apples subset comparison.
"""
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import load_all_videos, make_windows, split_videos, FallWindowDataset
from model import FallClassifier

POSE_DIRS_FULL = {
    "baseline": [
        os.path.join(os.path.dirname(__file__), "data", "poses"),
        os.path.join(os.path.dirname(__file__), "data", "poses_fallvision"),
        os.path.join(os.path.dirname(__file__), "data", "poses_caucafall"),
        os.path.join(os.path.dirname(__file__), "data", "poses_ofitw"),
    ],
    "gemini": [
        os.path.join(os.path.dirname(__file__), "data", "poses_gmdcsa24_v2"),
        os.path.join(os.path.dirname(__file__), "data", "poses_fallvision"),
        os.path.join(os.path.dirname(__file__), "data", "poses_caucafall"),
        os.path.join(os.path.dirname(__file__), "data", "poses_ofitw"),
    ],
}
CKPTS = {
    "baseline": os.path.join(os.path.dirname(__file__), "data", "best_model_baseline.pt"),
    "gemini": os.path.join(os.path.dirname(__file__), "data", "best_model_gemini_relabeled.pt"),
}


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
    if sum(all_labels) == 0:
        return None
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, prec, rec, f1, len(all_labels), sum(all_labels)


def main():
    for tag in ("baseline", "gemini"):
        videos = load_all_videos(POSE_DIRS_FULL[tag])
        _, val_videos = split_videos(videos, val_ratio=0.2, seed=42)
        gmdcsa24_val = [v for v in val_videos if v["name"].startswith("s") and ("_Fall_" in v["name"] or "_ADL_" in v["name"])]
        print(f"[{tag}] GMDCSA24 val videos: {len(gmdcsa24_val)} (fall={sum(v['label'] for v in gmdcsa24_val)})")

        samples = make_windows(gmdcsa24_val)
        ds = FallWindowDataset(samples)
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        ckpt = torch.load(CKPTS[tag], map_location="cpu", weights_only=False)
        model = FallClassifier()
        model.load_state_dict(ckpt["model_state"])

        result = evaluate(model, loader)
        if result is None:
            print(f"[{tag}] no positive windows in subset, skipping")
            continue
        acc, prec, rec, f1, n, npos = result
        print(f"[{tag}] GMDCSA24-only: n_windows={n} n_fall_windows={npos} acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")
        print()


if __name__ == "__main__":
    main()
