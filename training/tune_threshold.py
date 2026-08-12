"""Pick a real-world operating point for the trained model without retraining.

Two knobs, both free (just re-scoring the same validation predictions):
  1. Decision threshold -- missing a real fall is worse than a false alarm, so the
     right threshold for this task is usually LOWER than the default 0.5.
  2. Temporal smoothing -- only raise an alert if >=2 consecutive windows (windows
     overlap every STRIDE=10 frames, so "consecutive" means ~1/3 second apart) agree
     it's a fall. This kills single-window blips without needing more/better data.

Reports window-level PR at each threshold, then the more meaningful video-level
number: "does this clip ever get flagged" (sensitivity on real falls, false-alarm
rate on ADL clips), with and without smoothing, so we can pick a threshold+smoothing
combo and see its effect on the bed-scenario videos specifically.
"""
import os
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from dataset import load_all_videos, make_windows, split_videos, FallWindowDataset
from model import FallClassifier

POSE_DIRS = [
    os.path.join(os.path.dirname(__file__), "data", "poses"),
    os.path.join(os.path.dirname(__file__), "data", "poses_fallvision"),
    os.path.join(os.path.dirname(__file__), "data", "poses_caucafall"),
]
CKPT_PATH = os.path.join(os.path.dirname(__file__), "data", "best_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLDS = np.arange(0.15, 0.86, 0.05)


def smooth_flags(probs, threshold, min_consecutive=2):
    """probs: (T,) window probabilities in temporal order. Returns bool array,
    True only where `min_consecutive` consecutive windows all exceed threshold."""
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
    """Returns list of (video, probs (T_windows,), window_labels (T_windows,)) in temporal order."""
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
    print(f"Loaded checkpoint: val_f1={ckpt['val_f1']:.3f} (threshold=0.5, no smoothing baseline)\n")

    videos = load_all_videos(POSE_DIRS)
    _, val_videos = split_videos(videos, val_ratio=0.2)
    print(f"Val videos: {len(val_videos)}\n")

    per_video = get_video_probs(model, val_videos)

    all_probs = np.concatenate([p for _, p, _ in per_video])
    all_labels = np.concatenate([y for _, _, y in per_video])

    print("=== Window-level (no smoothing) ===")
    print(f"{'thr':>5} {'prec':>6} {'rec':>6} {'f1':>6}")
    best = None
    for t in THRESHOLDS:
        preds = (all_probs > t).astype(int)
        p = precision_score(all_labels, preds, zero_division=0)
        r = recall_score(all_labels, preds, zero_division=0)
        f = f1_score(all_labels, preds, zero_division=0)
        print(f"{t:5.2f} {p:6.3f} {r:6.3f} {f:6.3f}")
        if best is None or (r >= 0.85 and p > best[1]):
            best = (t, p, r, f)

    print("\n=== Video-level: does the clip ever get flagged? (sensitivity / false-alarm rate) ===")
    print("no smoothing (any single window over threshold triggers):")
    print(f"{'thr':>5} {'sens(fall)':>11} {'falsealarm(adl)':>16}")
    for t in THRESHOLDS:
        fall_videos = [(v, p) for v, p, _ in per_video if v["label"] == 1]
        adl_videos = [(v, p) for v, p, _ in per_video if v["label"] == 0]
        sens = np.mean([np.any(p > t) for _, p in fall_videos]) if fall_videos else 0
        fa = np.mean([np.any(p > t) for _, p in adl_videos]) if adl_videos else 0
        print(f"{t:5.2f} {sens*100:10.1f}% {fa*100:15.1f}%")

    print("\nwith smoothing (>=2 consecutive windows required):")
    print(f"{'thr':>5} {'sens(fall)':>11} {'falsealarm(adl)':>16}")
    smoothed_results = []
    for t in THRESHOLDS:
        fall_videos = [(v, p) for v, p, _ in per_video if v["label"] == 1]
        adl_videos = [(v, p) for v, p, _ in per_video if v["label"] == 0]
        sens = np.mean([smooth_flags(p, t).any() for _, p in fall_videos]) if fall_videos else 0
        fa = np.mean([smooth_flags(p, t).any() for _, p in adl_videos]) if adl_videos else 0
        smoothed_results.append((t, sens, fa))
        print(f"{t:5.2f} {sens*100:10.1f}% {fa*100:15.1f}%")

    # Recommend: highest sensitivity among thresholds where false-alarm rate <= 0.30,
    # falling back to the lowest false-alarm point that still keeps sensitivity >= 0.85.
    candidates = [c for c in smoothed_results if c[2] <= 0.30]
    if candidates:
        rec = max(candidates, key=lambda c: c[1])
    else:
        rec = max(smoothed_results, key=lambda c: c[1] - c[2])
    print(f"\nRecommended operating point: threshold={rec[0]:.2f}, smoothing=2-consecutive")
    print(f"  -> sensitivity {rec[1]*100:.1f}%, false-alarm rate {rec[2]*100:.1f}% (video-level, val set)")

    # Bed-scenario sweep -- the hardest case (lying in bed resembles lying after a fall),
    # and the one the team specifically flagged as a safety concern.
    fv_dir = os.path.join(os.path.dirname(__file__), "data", "poses_fallvision")
    fv_videos = load_all_videos(fv_dir)
    bed_videos = [v for v in fv_videos if "mask_b_" in v["name"]]
    if bed_videos:
        bed_probs = get_video_probs(model, bed_videos)
        fall_bed = [(v, p) for v, p, _ in bed_probs if v["label"] == 1]
        nofall_bed = [(v, p) for v, p, _ in bed_probs if v["label"] == 0]
        print(f"\n=== Bed scenario sweep (smoothing=2-consecutive): {len(fall_bed)} real falls, {len(nofall_bed)} normal bed clips ===")
        print(f"{'thr':>5} {'sens(bed fall)':>15} {'falsealarm(bed adl)':>20}")
        bed_results = []
        for t in THRESHOLDS:
            sens_bed = np.mean([smooth_flags(p, t).any() for _, p in fall_bed]) if fall_bed else 0
            fa_bed = np.mean([smooth_flags(p, t).any() for _, p in nofall_bed]) if nofall_bed else 0
            bed_results.append((t, sens_bed, fa_bed))
            print(f"{t:5.2f} {sens_bed*100:14.1f}% {fa_bed*100:19.1f}%")

        bed_candidates = [c for c in bed_results if c[2] <= 0.30]
        bed_rec = max(bed_candidates, key=lambda c: c[1]) if bed_candidates else max(bed_results, key=lambda c: c[1] - c[2])
        print(f"\nBed-scenario-optimized operating point: threshold={bed_rec[0]:.2f}, smoothing=2-consecutive")
        print(f"  -> bed-fall sensitivity {bed_rec[1]*100:.1f}%, bed false-alarm rate {bed_rec[2]*100:.1f}%")


if __name__ == "__main__":
    main()
