"""
Refined hypothesis #2: can a recent keypoint-velocity "impact spike" disambiguate
the cases where person-bed bbox overlap is high?

  - ADL false-positive clips (person calmly lay/sat on bed): expect LOW max velocity
    in the seconds leading up to the alert frame.
  - Genuine Fall clips that end up overlapping the bed: expect a HIGH velocity spike
    somewhere in the clip before the person goes still.

Uses the exact same _normalize_and_velocity() pipeline as production (torso-relative,
smoothed) so results are directly comparable to what the deployed model sees.
"""
import os
import sys
import cv2
import numpy as np

import importlib.util
_mod_path = os.path.join(os.path.dirname(__file__), "..", "app", "detection", "v3_fall_detection.py")
_spec = importlib.util.spec_from_file_location("v3_fall_detection", _mod_path)
_v3mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3mod)
V3PoseFallDetector = _v3mod.V3PoseFallDetector
_normalize_and_velocity = _v3mod._normalize_and_velocity

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(ROOT, "models")

detector = V3PoseFallDetector(MODEL_DIR)


def extract_raw_sequence(path, start_t, end_t):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_f = max(0, int(start_t * fps))
    end_f = int(end_t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    seq = []
    for _ in range(end_f - start_f):
        ok, frame = cap.read()
        if not ok:
            break
        kpts, found = detector.extract_keypoints(frame)
        seq.append(kpts)
    cap.release()
    return np.array(seq, dtype=np.float32) if seq else None


def max_velocity_mag(raw_seq):
    """raw_seq: (T,17,3) -> per-frame mean velocity magnitude across keypoints, then max over T."""
    if raw_seq is None or len(raw_seq) < 2:
        return 0.0
    feat = _normalize_and_velocity(raw_seq)  # (T,17,5): x,y,vis,vx,vy
    vel = feat[:, :, 3:5]
    mag_per_frame = np.linalg.norm(vel, axis=-1).mean(axis=-1)  # (T,)
    return float(mag_per_frame.max())


print("=== A) ADL false-positive clips: velocity in the 3s BEFORE the alert ===")
ADL_DIR = os.path.join(ROOT, "training", "data", "gmdcsa24_adl_raw_val")
ALERTS = {
    "s1_ADL_01": [2.0], "s2_ADL_03": [9.4], "s2_ADL_15": [6.6],
    "s4_ADL_08": [1.3, 6.7], "s4_ADL_10": [4.7],
}
for clip, timestamps in ALERTS.items():
    path = os.path.join(ADL_DIR, f"{clip}.mp4")
    for ts in timestamps:
        seq = extract_raw_sequence(path, max(0.0, ts - 3.0), ts)
        v = max_velocity_mag(seq)
        print(f"  {clip} @ {ts}s: max_vel(prev 3s)={v:.4f}")

print("\n=== B) Genuine Fall clips: max velocity across WHOLE clip (should show the fall impact) ===")
FALL_DIR = os.path.join(ROOT, "training", "data", "gmdcsa24_fall_raw")
VAL_FALL = ["s1_Fall_02", "s1_Fall_06", "s1_Fall_16", "s2_Fall_02", "s2_Fall_04",
            "s2_Fall_09", "s2_Fall_14", "s2_Fall_20", "s3_Fall_02", "s3_Fall_09",
            "s3_Fall_12", "s3_Fall_13", "s3_Fall_16", "s4_Fall_06", "s4_Fall_17"]
BED_OVERLAP_SUPPRESSED = {"s2_Fall_04", "s2_Fall_09", "s2_Fall_20", "s3_Fall_02",
                           "s3_Fall_09", "s3_Fall_12", "s3_Fall_13", "s3_Fall_16",
                           "s4_Fall_06", "s4_Fall_17"}
for clip in VAL_FALL:
    path = os.path.join(FALL_DIR, f"{clip}.mp4")
    cap = cv2.VideoCapture(path)
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0))
    cap.release()
    seq = extract_raw_sequence(path, 0.0, dur)
    v = max_velocity_mag(seq)
    tag = " [bed-overlap-suppressed candidate]" if clip in BED_OVERLAP_SUPPRESSED else ""
    print(f"  {clip}: max_vel(whole clip)={v:.4f}{tag}")
