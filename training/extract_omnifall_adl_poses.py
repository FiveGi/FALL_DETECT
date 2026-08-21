"""Extract MediaPipe poses for real OmniFall segments labeled lying/lie_down/
sitting/sit_down/kneeling/squatting -- all genuine ADL activities (label=0),
picked because SS27/SS20/SS21/SS23 all independently found the fall classifier
misfiring on exactly these poses (bed-lying most of all). Unlike SS22's mistake
(hard negatives from an unrelated failure surface), these are the SAME pattern
the model keeps failing on, sourced from ~100 different OOPS subjects/rooms
instead of GMDCSA24's 4 -- real diversity, not just more of what's already there.

Same segment-slicing approach as extract_ofitw_poses.py: each row already has
real start/end boundaries from human annotation, so frame_labels is a constant
0 array (this is exact ground truth, not a guess).
"""
import os
import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "poses_omnifall_adl")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models_cache", "pose_landmarker_lite.task")
PICKED_PATH = os.path.join(os.path.dirname(__file__), "data", "omnifall_adl_picked.json")
NUM_LANDMARKS = 33


def make_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_segment_keypoints(video_path, start_s, end_s, landmarker):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = max(0, int(start_s * fps))
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
    picked = json.load(open(PICKED_PATH))
    print(f"{len(picked)} segments to extract")

    landmarker = make_landmarker()
    total, skipped = 0, 0
    for i, seg in enumerate(picked):
        out_name = f"omnifall_adl_{i:04d}_label{seg['label_id']}.npz"
        out_path = os.path.join(OUT_DIR, out_name)
        if os.path.exists(out_path):
            total += 1
            continue

        seq = extract_segment_keypoints(seg["path"], seg["start"], seg["end"], landmarker)
        if seq is None or len(seq) < 5:
            skipped += 1
            continue

        frame_labels = np.zeros(len(seq), dtype=np.int64)  # all ADL, not fall
        np.savez_compressed(
            out_path, keypoints=seq, label=0, subject=-1, frame_labels=frame_labels
        )
        total += 1
        if total % 20 == 0:
            print(f"  [{total}/{len(picked)}] extracted (skipped {skipped})")

    landmarker.close()
    print(f"\nDone. Extracted {total}, skipped (too short) {skipped}.")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
