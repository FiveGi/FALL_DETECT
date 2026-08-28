"""
Same as extract_poses.py (GMDCSA24 keypoint extraction) but using YOLO-pose
(yolopose_extractor.py) instead of MediaPipe -- feasibility test for whether a
different pose backend fixes MediaPipe's documented weak detection rate on
prone/fallen bodies (see SKILL.md; v3_fall_detection.py's MIN_PERSON_FRACTION
comment). Output goes to a separate poses_yolopose/ dir so the original
MediaPipe-derived poses/ stays untouched for the before/after comparison.
"""
import os
import cv2
import numpy as np

from yolopose_extractor import YoloPoseExtractor, NUM_KEYPOINTS

DATASET_ROOT = os.environ.get(
    "DATASET_ROOT",
    os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_redownload"),
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "poses_yolopose")
MODEL_NAME = os.environ.get("YOLOPOSE_MODEL", "yolo26s-pose.pt")


def extract_video_keypoints(video_path, extractor):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kpts, found = extractor.extract_keypoints(frame)
        sequence.append(kpts)
    cap.release()
    if not sequence:
        return None
    return np.stack(sequence, axis=0)  # (T, 17, 3)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    extractor = YoloPoseExtractor(MODEL_NAME)

    subjects = sorted(
        d for d in os.listdir(DATASET_ROOT) if d.lower().startswith("subject")
    )
    print(f"Found {len(subjects)} subjects: {subjects}")

    total = 0
    skipped = 0
    for subject in subjects:
        subject_id = subject.replace("Subject ", "").strip()
        for cls_name, label in (("ADL", 0), ("Fall", 1)):
            cls_dir = os.path.join(DATASET_ROOT, subject, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith((".mp4", ".avi", ".mov")))
            for fname in files:
                video_path = os.path.join(cls_dir, fname)
                out_name = f"s{subject_id}_{cls_name}_{os.path.splitext(fname)[0]}.npz"
                out_path = os.path.join(OUT_DIR, out_name)
                if os.path.exists(out_path):
                    total += 1
                    continue
                seq = extract_video_keypoints(video_path, extractor)
                if seq is None or len(seq) < 10:
                    print(f"  SKIP (too short/no frames): {video_path}")
                    skipped += 1
                    continue
                np.savez_compressed(out_path, keypoints=seq, label=label, subject=int(subject_id))
                total += 1
                print(f"  [{total}] {subject}/{cls_name}/{fname} -> {seq.shape[0]} frames")

    print(f"\nDone. Extracted {total} videos, skipped {skipped}.")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
