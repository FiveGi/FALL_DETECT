"""Extract MediaPipe poses from OmniFall's OF-In-the-Wild (OOPS) segments.

Unlike the other extractors, one OOPS source video contains several labeled
segments back to back (e.g. walk -> fall -> fallen, from OmniFall's 16-class
taxonomy), so each row of the OF-ItW label table becomes its own .npz, sliced
from the source video by [start, end) time in seconds -- not one .npz per video.

These are genuine real-world accidents (sourced from the OOPS "unintentional
action" dataset), not staged lab recordings like GMDCSA24/FallVision/CAUCAFall,
and the segment boundaries are real human annotations, not a motion-peak guess.
Because each segment is already trimmed to a single homogeneous activity, every
window inside it truly is that one label -- so we store a constant frame_labels
array, which makes dataset.py's make_windows() treat it as exact ground truth
(the same code path used for CAUCAFall) instead of applying the motion-peak
heuristic meant for whole, untrimmed clips.

Binary label: 1 if the segment's class is "fall" or "fallen" (taxonomy ids 1, 2),
0 otherwise (walk, sitting, lying, standing, kneeling, squatting, crawl, jump, other).
"""
import os
import re
import glob
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from datasets import load_dataset, concatenate_datasets

VIDEO_ROOT = os.environ.get(
    "OOPS_VIDEO_ROOT",
    os.path.join(os.path.dirname(__file__), "data", "oops_extracted"),
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "poses_ofitw")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models_cache", "pose_landmarker_lite.task")
NUM_LANDMARKS = 33
FALL_LABEL_IDS = {1, 2}  # "fall", "fallen" in OmniFall's 16-class taxonomy


def make_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _norm(s):
    """Alphanumeric-only, lowercased key. The OF-ItW label 'path' values have
    spaces/punctuation stripped relative to the real OOPS filenames (e.g. label
    'BestFailsofWeek2July2016_FailArmy9' vs file 'Best Fails of Week 2 July 2016
    _ FailArmy9.mp4'), so exact-name matching fails -- normalize both sides."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def build_video_index(root):
    """Map a normalized label 'path' basename to an actual file on disk, regardless
    of the extracted archive's exact layout or filename punctuation."""
    index = {}
    for ext in ("mp4", "avi", "mov", "webm", "mkv"):
        for fp in glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True):
            stem = os.path.splitext(os.path.basename(fp))[0]
            index[_norm(stem)] = fp
    return index


def extract_segment_keypoints(video_path, start_s, end_s, landmarker):
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            kpts = np.array([[p.x, p.y, p.visibility] for p in lm], dtype=np.float32)
        else:
            kpts = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
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

    landmarker = make_landmarker()
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

        seq = extract_segment_keypoints(video_path, row["start"], row["end"], landmarker)
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

    landmarker.close()
    print(f"\nDone. Extracted {total}, skipped (too short) {skipped}, missing video file {missing}.")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
