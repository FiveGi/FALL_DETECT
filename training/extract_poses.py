"""Extract MediaPipe pose keypoints from the GMDCSA24 fall detection dataset.

Walks Subject */ADL and Subject */Fall folders, runs MediaPipe Pose on every
frame of every video, and saves one .npz per video containing the per-frame
keypoint sequence (x, y, visibility for each of the 33 landmarks) plus the
label (1 = Fall, 0 = ADL) and subject id (used later for a leakage-free split).
"""
import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

DATASET_ROOT = os.environ.get(
    "DATASET_ROOT",
    r"C:\Users\USER\AppData\Local\Temp\claude\d--project-PROJECT\3c35afae-a94f-4f2a-888b-a12d22a7b72c"
    r"\scratchpad\gmdcsa24\ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-d3edb5d",
)
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "poses")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models_cache", "pose_landmarker_lite.task")
NUM_LANDMARKS = 33


def make_landmarker():
    # IMAGE mode: each frame is treated independently (no timestamp bookkeeping
    # needed), so a single landmarker instance can be reused across all videos.
    # We do our own temporal windowing/normalization downstream anyway, so we
    # don't need mediapipe's VIDEO-mode temporal smoothing.
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_video_keypoints(video_path, landmarker):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    while True:
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
    cap.release()
    if not sequence:
        return None
    return np.stack(sequence, axis=0)  # (T, 33, 3)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    subjects = sorted(
        d for d in os.listdir(DATASET_ROOT) if d.lower().startswith("subject")
    )
    print(f"Found {len(subjects)} subjects: {subjects}")

    total = 0
    skipped = 0
    landmarker = make_landmarker()
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
                seq = extract_video_keypoints(video_path, landmarker)
                if seq is None or len(seq) < 10:
                    print(f"  SKIP (too short/no frames): {video_path}")
                    skipped += 1
                    continue
                np.savez_compressed(out_path, keypoints=seq, label=label, subject=int(subject_id))
                total += 1
                print(f"  [{total}] {subject}/{cls_name}/{fname} -> {seq.shape[0]} frames")
    landmarker.close()

    print(f"\nDone. Extracted {total} videos, skipped {skipped}.")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
