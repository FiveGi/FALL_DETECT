"""4-way cross-comparison on the GMDCSA24 validation subset: each of the two
checkpoints (trained on old-heuristic vs Gemini-onset labels) evaluated against
each of the two label versions as ground truth, on the identical 43 val videos.
This separates "did the model change" from "did the yardstick change."
"""
import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import load_all_videos, make_windows, split_videos, FallWindowDataset
from model import FallClassifier

HERE = os.path.dirname(__file__)
OTHER_DIRS = [
    os.path.join(HERE, "data", "poses_fallvision"),
    os.path.join(HERE, "data", "poses_caucafall"),
    os.path.join(HERE, "data", "poses_ofitw"),
]
GMDCSA24_DIRS = {
    "old_heuristic_labels": os.path.join(HERE, "data", "poses"),
    "gemini_labels": os.path.join(HERE, "data", "poses_gmdcsa24_v2"),
}
CKPTS = {
    "baseline_model": os.path.join(HERE, "data", "best_model_baseline.pt"),
    "gemini_model": os.path.join(HERE, "data", "best_model_gemini_relabeled.pt"),
}


def get_gmdcsa24_val_windows(label_source):
    videos = load_all_videos([GMDCSA24_DIRS[label_source]] + OTHER_DIRS)
    _, val_videos = split_videos(videos, val_ratio=0.2, seed=42)
    subset = [v for v in val_videos if v["name"].startswith("s") and "_Fall_" in v["name"] or (v["name"].startswith("s") and "_ADL_" in v["name"])]
    return make_windows(subset)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, prec, rec, f1


def main():
    windows_by_labels = {name: get_gmdcsa24_val_windows(name) for name in GMDCSA24_DIRS}
    models = {}
    for name, path in CKPTS.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = FallClassifier()
        m.load_state_dict(ckpt["model_state"])
        models[name] = m

    print(f"{'model':<16}{'labels':<20}{'acc':>7}{'prec':>7}{'rec':>7}{'f1':>7}")
    for model_name, model in models.items():
        for label_name, samples in windows_by_labels.items():
            loader = DataLoader(FallWindowDataset(samples), batch_size=32, shuffle=False)
            acc, prec, rec, f1 = evaluate(model, loader)
            print(f"{model_name:<16}{label_name:<20}{acc:>7.3f}{prec:>7.3f}{rec:>7.3f}{f1:>7.3f}")


if __name__ == "__main__":
    main()
