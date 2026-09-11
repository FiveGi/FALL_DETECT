"""Same sweep as tune_threshold.py, but pointed at the checkpoint actually deployed in
models/fall_classifier_v3.onnx (yolopose_aug_seed42.pt, confirmed by MD5 match) and the
YOLO-pose-derived pose dirs it was trained on -- tune_threshold.py's own hardcoded
CKPT_PATH/POSE_DIRS point at an older, undeployed checkpoint on MediaPipe-derived data,
which would give numbers with no bearing on the real system. Kept separate rather than
editing tune_threshold.py so neither script's "what does this evaluate" is ambiguous."""
import os
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from dataset import load_all_videos, make_windows, split_videos
from model import FallClassifier

BASE = os.path.dirname(__file__)
POSE_DIRS = [
    os.path.join(BASE, "data", "poses_yolopose"),
    os.path.join(BASE, "data", "poses_fallvision"),
    os.path.join(BASE, "data", "poses_caucafall_yolopose"),
    os.path.join(BASE, "data", "poses_ofitw_yolopose_matched"),
]
CKPT_PATH = os.path.join(BASE, "data", "yolopose_aug_seed42.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLDS = np.arange(0.15, 0.86, 0.05)


def smooth_flags(probs, threshold, min_consecutive=2):
    raw = probs > threshold
    if min_consecutive <= 1:
        return raw
    smoothed = np.zeros_like(raw)
    run = 0
    for i, flag in enumerate(raw):
        run = run + 1 if flag else 0
        if run >= min_consecutive:
            smoothed[i - min_consecutive + 1: i + 1] = True
    return smoothed


def get_video_probs(model, videos):
    out = []
    for v in videos:
        samples = make_windows([v])
        x = torch.stack([torch.from_numpy(s[0].astype(np.float32)) for s in samples]).to(DEVICE)
        y = np.array([s[1] for s in samples])
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).cpu().numpy()
        out.append((v, probs, y))
    return out


def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = FallClassifier().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded DEPLOYED checkpoint: {CKPT_PATH}")
    print(f"val_f1={ckpt['val_f1']:.3f} (threshold=0.5, no smoothing baseline)\n")

    videos = load_all_videos(POSE_DIRS)
    print(f"Loaded {len(videos)} videos total, label counts: fall={sum(v['label'] for v in videos)}, adl={sum(1 for v in videos if v['label']==0)}")
    _, val_videos = split_videos(videos, val_ratio=0.2)
    print(f"Val videos: {len(val_videos)}\n")

    per_video = get_video_probs(model, val_videos)
    all_probs = np.concatenate([p for _, p, _ in per_video])
    all_labels = np.concatenate([y for _, _, y in per_video])

    print("=== Window-level (no smoothing) ===")
    print(f"{'thr':>5} {'prec':>6} {'rec':>6} {'f1':>6}")
    for t in THRESHOLDS:
        preds = (all_probs > t).astype(int)
        p = precision_score(all_labels, preds, zero_division=0)
        r = recall_score(all_labels, preds, zero_division=0)
        f = f1_score(all_labels, preds, zero_division=0)
        print(f"{t:5.2f} {p:6.3f} {r:6.3f} {f:6.3f}")

    print("\n=== Video-level: does the clip ever get flagged? (sensitivity / false-alarm rate) ===")
    print("with smoothing (>=2 consecutive windows required):")
    print(f"{'thr':>5} {'sens(fall)':>11} {'falsealarm(adl)':>16}")
    smoothed_results = []
    for t in THRESHOLDS:
        fall_videos = [(v, p) for v, p, _ in per_video if v["label"] == 1]
        adl_videos = [(v, p) for v, p, _ in per_video if v["label"] == 0]
        sens = np.mean([smooth_flags(p, t).any() for _, p in fall_videos]) if fall_videos else 0
        fa = np.mean([smooth_flags(p, t).any() for _, p in adl_videos]) if adl_videos else 0
        smoothed_results.append((t, sens, fa))
        print(f"{t:5.2f} {sens*100:10.1f}% {fa*100:15.1f}%")

    candidates = [c for c in smoothed_results if c[2] <= 0.30]
    rec = max(candidates, key=lambda c: c[1]) if candidates else max(smoothed_results, key=lambda c: c[1] - c[2])
    print(f"\nRecommended operating point: threshold={rec[0]:.2f}, smoothing=2-consecutive")
    print(f"  -> sensitivity {rec[1]*100:.1f}%, false-alarm rate {rec[2]*100:.1f}% (video-level, val set)")
    print(f"\n(Current production THRESHOLD in app/detection/v3_fall_detection.py is 0.5, smoothing already 2-of-3 windows)")


if __name__ == "__main__":
    main()
