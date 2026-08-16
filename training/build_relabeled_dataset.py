"""Build data/poses_gmdcsa24_v2/: the 81 ADL clips copied unchanged (their
frame_labels are already exactly correct -- no fall ever happens in them, not a
guess) plus the 79 Fall clips with frame_labels rebuilt from Gemini's fall-onset
timestamp (data/gemini_relabel_results.json) instead of the old motion-energy-peak
heuristic. Any clip Gemini failed to label keeps the old heuristic-derived boundary
as a fallback so the run still produces a complete dataset.
"""
import os
import json
import glob
import shutil

import cv2
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), "data", "poses")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "poses_gmdcsa24_v2")
RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_fall_raw")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "data", "gemini_relabel_results.json")


def to_coco17(raw_seq):
    MEDIAPIPE33_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    if raw_seq.shape[1] == 17:
        return raw_seq
    return raw_seq[:, MEDIAPIPE33_TO_COCO17, :]


def compute_motion_energy(raw_seq, smooth=5):
    xy = raw_seq[:, :, :2]
    diffs = np.linalg.norm(np.diff(xy, axis=0), axis=2).mean(axis=1)
    energy = np.concatenate([[0.0], diffs])
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        energy = np.convolve(energy, kernel, mode="same")
    return energy


def fallback_onset_frame(keypoints):
    """Same heuristic dataset.py's make_windows() uses for whole-clip-only labels:
    the frame of peak motion energy, so clips Gemini couldn't label degrade to the
    old behavior instead of being silently wrong."""
    raw = to_coco17(keypoints.astype(np.float32))
    motion = compute_motion_energy(raw)
    return int(np.argmax(motion))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    all_npz = sorted(glob.glob(os.path.join(SRC_DIR, "*.npz")))
    adl = [p for p in all_npz if "_Fall_" not in os.path.basename(p)]
    fall = [p for p in all_npz if "_Fall_" in os.path.basename(p)]
    print(f"ADL clips (copied unchanged): {len(adl)}")
    print(f"Fall clips (relabeling): {len(fall)}")

    for p in adl:
        shutil.copy2(p, os.path.join(OUT_DIR, os.path.basename(p)))

    used_gemini, used_fallback = 0, 0
    for p in fall:
        name = os.path.splitext(os.path.basename(p))[0]
        data = np.load(p, allow_pickle=True)
        keypoints = data["keypoints"]
        T = keypoints.shape[0]

        result = results.get(name)
        video_path = os.path.join(RAW_DIR, name + ".mp4")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        if result and "error" not in result:
            onset_frame = int(round(result["fall_onset_s"] * fps))
            onset_frame = max(0, min(onset_frame, T - 1))
            used_gemini += 1
        else:
            onset_frame = fallback_onset_frame(keypoints)
            used_fallback += 1
            print(f"  fallback (no valid Gemini result) for {name}")

        frame_labels = np.zeros(T, dtype=np.int64)
        frame_labels[onset_frame:] = 1

        np.savez_compressed(
            os.path.join(OUT_DIR, os.path.basename(p)),
            keypoints=keypoints,
            label=data["label"],
            subject=data["subject"],
            frame_labels=frame_labels,
        )

    print(f"\nDone. {used_gemini} clips used Gemini onset, {used_fallback} used heuristic fallback.")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
