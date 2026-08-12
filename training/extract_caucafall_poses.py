"""Extract MediaPipe pose keypoints from the CAUCAFall dataset (Mendeley,
doi:10.17632/7w7fccy7ky), pairing each frame with its REAL ground-truth label
from the dataset's own per-frame YOLO annotation files (ClassID x y w h,
class 0 = no-fall, class 1 = fall) -- not the motion-peak heuristic used for
the other two datasets. This gives one video with exact per-frame fall/no-fall
labels instead of a single whole-clip label.

CAUCAFall ships one .png + one .txt per frame (same base filename) alongside
the source .avi, so we read the images directly rather than re-decoding the
video -- that keeps frame order and the label files in exact 1:1 correspondence.
"""
import os
import glob
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

DATASET_ROOT = os.environ.get(
    "CAUCAFALL_ROOT",
    r"d:\project\PROJECT\Dataset CAUCAFall\CAUCAFall",
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "poses_caucafall")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models_cache", "pose_landmarker_lite.task")
NUM_LANDMARKS = 33

FALL_ACTIVITIES = {"Fall backwards", "Fall forward", "Fall left", "Fall right", "Fall sitting"}


def make_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def read_frame_label(txt_path):
    """Returns 1 (fall) or 0 (no-fall) from the first column of the YOLO label file."""
    try:
        with open(txt_path) as f:
            line = f.readline().strip()
        if not line:
            return 0
        return int(line.split()[0])
    except (FileNotFoundError, ValueError, IndexError):
        return 0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    landmarker = make_landmarker()

    subjects = sorted(
        (d for d in os.listdir(DATASET_ROOT) if d.startswith("Subject.")),
        key=lambda s: int(s.split(".")[1]),
    )
    print(f"Found {len(subjects)} subjects")

    total, skipped = 0, 0
    for subject in subjects:
        subject_dir = os.path.join(DATASET_ROOT, subject)
        activities = sorted(os.listdir(subject_dir))
        for activity in activities:
            act_dir = os.path.join(subject_dir, activity)
            if not os.path.isdir(act_dir):
                continue

            out_name = f"cauca_{subject}_{activity.replace(' ', '_')}.npz"
            out_path = os.path.join(OUT_DIR, out_name)
            if os.path.exists(out_path):
                total += 1
                continue

            png_files = sorted(glob.glob(os.path.join(act_dir, "*.png")))
            if len(png_files) < 10:
                skipped += 1
                continue

            sequence, frame_labels = [], []
            for png_path in png_files:
                frame = cv2.imread(png_path)
                if frame is None:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect(mp_image)
                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    kpts = np.array([[p.x, p.y, p.visibility] for p in lm], dtype=np.float32)
                else:
                    kpts = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
                sequence.append(kpts)

                txt_path = os.path.splitext(png_path)[0] + ".txt"
                frame_labels.append(read_frame_label(txt_path))

            seq = np.stack(sequence, axis=0)
            labels = np.array(frame_labels, dtype=np.int64)
            video_label = int(labels.max())  # matches whole-clip label convention used elsewhere

            np.savez_compressed(
                out_path,
                keypoints=seq,
                frame_labels=labels,
                label=video_label,
                subject=f"cauca_{subject}",
                activity=activity,
            )
            total += 1
            print(f"  [{total}] {subject}/{activity} -> {seq.shape[0]} frames, "
                  f"{labels.sum()} fall-frames, video_label={video_label}")

    print(f"\nDone. Extracted {total}, skipped {skipped}.")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
