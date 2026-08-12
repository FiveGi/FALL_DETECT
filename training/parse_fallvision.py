"""Parse the FallVision dataset's pre-extracted keypoint CSVs (Harvard Dataverse,
doi:10.7910/DVN/75QPKK) into the same per-video .npz format produced by
extract_poses.py, so both datasets can be trained on together.

FallVision ships one CSV per video clip with columns: Frame, Keypoint, X, Y, Confidence
using the standard COCO-17 body keypoints (no face mesh / hands like MediaPipe's 33).
Label comes from the folder/file prefix: f_ = fall (1), nf_ = no-fall / ADL (0).
"""
import os
import glob
import re
import numpy as np
import pandas as pd

from dataset import COCO17_ORDER

SRC_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..",
)
SRC_ROOT = os.environ.get(
    "FALLVISION_ROOT",
    r"C:\Users\USER\AppData\Local\Temp\claude\d--project-PROJECT\3c35afae-a94f-4f2a-888b-a12d22a7b72c"
    r"\scratchpad\fallvision_keypoints\extracted",
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "poses_fallvision")

KPT_INDEX = {name: i for i, name in enumerate(COCO17_ORDER)}


def parse_csv(path):
    df = pd.read_csv(path)
    n_frames = int(df["Frame"].max())
    seq = np.zeros((n_frames, 17, 3), dtype=np.float32)
    for _, row in df.iterrows():
        kpt = str(row["Keypoint"]).strip()
        idx = KPT_INDEX.get(kpt)
        if idx is None:
            continue
        f = int(row["Frame"]) - 1
        seq[f, idx] = [row["X"], row["Y"], row["Confidence"]]
    return seq


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(SRC_ROOT, "**", "*.csv"), recursive=True))
    print(f"Found {len(csv_files)} CSV files")

    total, skipped = 0, 0
    for i, path in enumerate(csv_files, 1):
        # archive folder names look like f_mask_b_1_keypoints_csv or nf_mask_c_2_keypoints_csv
        archive_name = os.path.basename(os.path.dirname(path))
        is_fall = 1 if archive_name.startswith("f_") and not archive_name.startswith("nf_") else 0

        m = re.match(r"^(f|nf)_mask_([a-z])_(\d+)", archive_name)
        scene = m.group(2) if m else "x"  # b=bed, c=chair, s=standing
        part = m.group(3) if m else "0"

        clip_id = os.path.splitext(os.path.basename(path))[0]
        out_name = f"fv_{archive_name}_{clip_id}.npz"
        out_path = os.path.join(OUT_DIR, out_name)
        if os.path.exists(out_path):
            total += 1
            continue

        try:
            seq = parse_csv(path)
        except Exception as e:
            print(f"  SKIP (parse error) {path}: {e}")
            skipped += 1
            continue

        if seq.shape[0] < 10:
            skipped += 1
            continue

        np.savez_compressed(out_path, keypoints=seq, label=is_fall, scene=scene, subject=f"fv_{scene}_{part}")
        total += 1
        if i % 500 == 0:
            print(f"  [{i}/{len(csv_files)}] processed...")

    print(f"\nDone. Parsed {total}, skipped {skipped}.")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
