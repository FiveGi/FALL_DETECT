import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import load_all_videos, make_windows, split_videos, FallWindowDataset
from model import FallClassifier

# GMDCSA24_DIR_NAME lets relabel_gmdcsa24_gemini.py's downstream comparison point this
# at "poses" (original motion-peak heuristic) or "poses_gmdcsa24_v2" (Gemini-verified
# fall onset) without editing this file, so both runs use identical code.
GMDCSA24_DIR_NAME = os.environ.get("GMDCSA24_DIR_NAME", "poses")
# CAUCAFALL_DIR_NAME/OFITW_DIR_NAME: same pattern as GMDCSA24_DIR_NAME, added so the
# YOLO-pose-vs-MediaPipe pose-extractor comparison can point all three video-derived
# datasets at their yolopose_extractor.py-produced siblings (poses_yolopose,
# poses_caucafall_yolopose, poses_ofitw_yolopose_matched) without editing this file.
# poses_fallvision is untouched either way -- it's pre-extracted external keypoints,
# not derived from either pose backend, so it stays a shared constant across both runs.
CAUCAFALL_DIR_NAME = os.environ.get("CAUCAFALL_DIR_NAME", "poses_caucafall")
OFITW_DIR_NAME = os.environ.get("OFITW_DIR_NAME", "poses_ofitw")
POSE_DIRS = [
    os.path.join(os.path.dirname(__file__), "data", GMDCSA24_DIR_NAME),    # GMDCSA24
    os.path.join(os.path.dirname(__file__), "data", "poses_fallvision"),   # FallVision (COCO-17, heuristic labels)
    os.path.join(os.path.dirname(__file__), "data", CAUCAFALL_DIR_NAME),   # CAUCAFall (MediaPipe-33, REAL per-frame labels)
    os.path.join(os.path.dirname(__file__), "data", OFITW_DIR_NAME),       # OmniFall OF-ItW / OOPS (MediaPipe-33, REAL segment labels, real-world not staged)
]
# USE_REALTEST_V1: 67 Gemini+Claude-verified segments from Test/4.mp4-11.mp4 (SS21/SS22)
# -- 45 real falls, 22 explicit hard negatives (bed-lying, dancing, standing-near-objects
# that the deployed model currently misfires on). Off by default so SS20/SS21's
# baseline numbers stay reproducible; set to 1 for the SS22 before/after comparison.
if os.environ.get("USE_REALTEST_V1") == "1":
    POSE_DIRS.append(os.path.join(os.path.dirname(__file__), "data", "poses_realtest_v1"))
# USE_OMNIFALL_ADL: 117 real OmniFall segments labeled lying/lie_down/sitting/sit_down/
# kneeling/squatting (SS28) -- unlike SS22, these directly target the SPECIFIC pattern
# (bed/floor-lying, confirmed across SS17/SS20/SS21/SS23/SS27) the deployed model keeps
# misfiring on, sourced from ~100 different OOPS subjects/rooms instead of GMDCSA24's 4.
if os.environ.get("USE_OMNIFALL_ADL") == "1":
    POSE_DIRS.append(os.path.join(os.path.dirname(__file__), "data", "poses_omnifall_adl"))
CKPT_PATH = os.environ.get(
    "CKPT_PATH", os.path.join(os.path.dirname(__file__), "data", "best_model.pt")
)
EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(os.environ.get("TRAIN_SEED", 42))  # fixes torch's init/shuffle RNG so before/after label-quality comparisons aren't noise


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, prec, rec, f1


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    videos = load_all_videos(POSE_DIRS)
    print(f"Loaded {len(videos)} videos")
    labels = [v["label"] for v in videos]
    print(f"  Fall: {sum(labels)}, ADL: {len(labels) - sum(labels)}")

    train_videos, val_videos = split_videos(videos, val_ratio=0.2)
    print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")

    train_samples = make_windows(train_videos)
    val_samples = make_windows(val_videos)
    print(f"Train windows: {len(train_samples)}, Val windows: {len(val_samples)}")

    # AUGMENT (SS35): horizontal-flip + synthetic per-joint occlusion, applied only to the
    # training split (never val, which must stay a clean measure of real generalization).
    # Off by default so every prior run in this file stays exactly reproducible.
    AUGMENT = os.environ.get("AUGMENT", "0") == "1"
    train_ds = FallWindowDataset(train_samples, augment=AUGMENT)
    val_ds = FallWindowDataset(val_samples, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Missing a real fall is far worse than a false alarm, so weight the loss to punish
    # false negatives harder than the raw class ratio would -- pushes the model away from
    # the "always predict no-fall" local optimum it settled into last run.
    # POS_WEIGHT_MULT (SS29): SS28 found adding real targeted lying/sitting hard negatives
    # pushed recall to 100% but didn't fix a single one of the ADL false positives it was
    # meant to fix -- testing whether this 1.5x recall bias is why, by trying it at 1.0x
    # (no artificial bias beyond the natural class ratio) alongside the same SS28 data.
    POS_WEIGHT_MULT = float(os.environ.get("POS_WEIGHT_MULT", "1.5"))
    train_labels = np.array([lbl for _, lbl, _ in train_samples])
    n_pos, n_neg = train_labels.sum(), len(train_labels) - train_labels.sum()
    pos_weight = torch.tensor([(n_neg / max(n_pos, 1)) * POS_WEIGHT_MULT], device=DEVICE)
    print(f"Train windows: {n_pos} fall, {n_neg} no-fall -> pos_weight={pos_weight.item():.2f} (mult={POS_WEIGHT_MULT})")

    # HIDDEN_SIZE (SS31): model.py's docstring notes a much bigger architecture (ST-GCN,
    # ~17x more params) underperformed this one on the old data (0.582 vs 0.615 F1) --
    # testing a modest capacity increase (not 17x) on top of SS29's winning data/pos_weight
    # config specifically, since that comparison predates this session's data additions.
    HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", "128"))
    model = FallClassifier(hidden=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_ds)

        acc, prec, rec, f1 = evaluate(model, val_loader)
        scheduler.step(f1)
        print(f"Epoch {epoch:3d} | loss {train_loss:.4f} | val_acc {acc:.3f} val_prec {prec:.3f} val_rec {rec:.3f} val_f1 {f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                "model_state": model.state_dict(),
                "val_f1": f1,
                "val_acc": acc,
                "epoch": epoch,
            }, CKPT_PATH)
            print(f"  -> saved new best (f1={f1:.3f})")

    print(f"\nBest val F1: {best_f1:.3f}")
    print(f"Checkpoint saved to: {CKPT_PATH}")


if __name__ == "__main__":
    main()
