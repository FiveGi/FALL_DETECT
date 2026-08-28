"""Same as extract_caucafall_poses.py but using YOLO-pose instead of MediaPipe --
see extract_poses_yolopose.py for why. Output to poses_caucafall_yolopose/."""
import os
import glob
import cv2
import numpy as np

from yolopose_extractor import YoloPoseExtractor, NUM_KEYPOINTS

DATASET_ROOT = os.environ.get(
    "CAUCAFALL_ROOT",
    r"d:\project\PROJECT\Dataset CAUCAFall\CAUCAFall",
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "poses_caucafall_yolopose")
MODEL_NAME = os.environ.get("YOLOPOSE_MODEL", "yolo26s-pose.pt")


def read_frame_label(txt_path):
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
    extractor = YoloPoseExtractor(MODEL_NAME)

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
                kpts, found = extractor.extract_keypoints(frame)
                sequence.append(kpts)

                txt_path = os.path.splitext(png_path)[0] + ".txt"
                frame_labels.append(read_frame_label(txt_path))

            seq = np.stack(sequence, axis=0)
            labels = np.array(frame_labels, dtype=np.int64)
            video_label = int(labels.max())

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
