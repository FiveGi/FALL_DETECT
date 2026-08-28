"""Same as extract_ofitw_poses.py but using YOLO-pose instead of MediaPipe --
see extract_poses_yolopose.py for why. Output to poses_ofitw_yolopose/."""
import os
import re
import glob
import cv2
import numpy as np
from datasets import load_dataset, concatenate_datasets

from yolopose_extractor import YoloPoseExtractor, NUM_KEYPOINTS

VIDEO_ROOT = os.environ.get(
    "OOPS_VIDEO_ROOT",
    os.path.join(os.path.dirname(__file__), "data", "oops_download", "oops_dataset"),
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "poses_ofitw_yolopose")
MODEL_NAME = os.environ.get("YOLOPOSE_MODEL", "yolo26s-pose.pt")
FALL_LABEL_IDS = {1, 2}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def build_video_index(root):
    index = {}
    for ext in ("mp4", "avi", "mov", "webm", "mkv"):
        for fp in glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True):
            stem = os.path.splitext(os.path.basename(fp))[0]
            index[_norm(stem)] = fp
    return index


def extract_segment_keypoints(video_path, start_s, end_s, extractor):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start_s * fps)
    end_frame = max(start_frame + 1, int(end_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    sequence = []
    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        kpts, found = extractor.extract_keypoints(frame)
        sequence.append(kpts)
        frame_idx += 1
    cap.release()
    if not sequence:
        return None
    return np.stack(sequence, axis=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = load_dataset("simplexsigil2/omnifall", "of-itw")
    all_rows = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
    print(f"Total OF-ItW segments: {len(all_rows)}")

    index = build_video_index(VIDEO_ROOT)
    print(f"Found {len(index)} video files under {VIDEO_ROOT}")
    if not index:
        print("No video files found -- has the OOPS archive been extracted to this path yet?")
        return

    extractor = YoloPoseExtractor(MODEL_NAME)
    total, skipped, missing = 0, 0, 0
    for i, row in enumerate(all_rows):
        stem = os.path.basename(row["path"])
        video_path = index.get(_norm(stem))
        if video_path is None:
            missing += 1
            continue

        safe_start = f"{row['start']:.3f}".replace(".", "-")
        safe_end = f"{row['end']:.3f}".replace(".", "-")
        out_name = f"ofitw_{stem}_{safe_start}_{safe_end}.npz"
        out_path = os.path.join(OUT_DIR, out_name)
        if os.path.exists(out_path):
            total += 1
            continue

        seq = extract_segment_keypoints(video_path, row["start"], row["end"], extractor)
        if seq is None or len(seq) < 5:
            skipped += 1
            continue

        label = 1 if row["label"] in FALL_LABEL_IDS else 0
        frame_labels = np.full(len(seq), label, dtype=np.int64)
        np.savez_compressed(
            out_path, keypoints=seq, label=label, subject=-1, frame_labels=frame_labels
        )
        total += 1
        if total % 200 == 0:
            print(f"  [{total}/{len(all_rows)}] processed (skipped {skipped}, missing video {missing})")

    print(f"\nDone. Extracted {total}, skipped (too short) {skipped}, missing video file {missing}.")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
