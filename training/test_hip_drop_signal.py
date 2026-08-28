"""
Gemini-suggested refinement: instead of instantaneous per-frame velocity (noisy,
jitter-sensitive - already tested and failed), measure the fastest NET hip-height
drop over a 0.5s sub-window. Hypothesis: a genuine fall is a sudden discontinuous
collapse (large net drop in a short window), while lying/sitting down on a bed is a
smooth, gradual multi-second descent (small net drop per 0.5s window even though
total displacement may be similar).

Uses RAW (unnormalized, frame-height-relative) hip-center y so the "how far did they
actually descend, in a hurry" signal isn't swallowed by torso-relative normalization.
"""
import os
import cv2
import numpy as np
import importlib.util

_mod_path = os.path.join(os.path.dirname(__file__), "..", "app", "detection", "v3_fall_detection.py")
_spec = importlib.util.spec_from_file_location("v3_fall_detection", _mod_path)
_v3mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3mod)
V3PoseFallDetector = _v3mod.V3PoseFallDetector
LEFT_HIP = _v3mod.LEFT_HIP
RIGHT_HIP = _v3mod.RIGHT_HIP

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(ROOT, "models")
detector = V3PoseFallDetector(MODEL_DIR)


def extract_hip_y_sequence(path, start_t, end_t):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_f = max(0, int(start_t * fps))
    end_f = int(end_t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    ys, founds = [], []
    for _ in range(end_f - start_f):
        ok, frame = cap.read()
        if not ok:
            break
        kpts, found = detector.extract_keypoints(frame)
        hip_y = (kpts[LEFT_HIP, 1] + kpts[RIGHT_HIP, 1]) / 2.0  # normalized [0,1], y grows downward
        ys.append(hip_y)
        founds.append(found)
    cap.release()
    return np.array(ys, dtype=np.float32), fps


def max_net_drop(ys, fps, window_s=0.5):
    """Max (ys[t] - ys[t-w]) over any window_s-second span -- positive means moved DOWN
    (fall-like); this is net displacement, not summed per-frame jitter."""
    w = max(1, int(window_s * fps))
    if len(ys) <= w:
        return 0.0
    diffs = ys[w:] - ys[:-w]
    return float(diffs.max())


print("=== A) ADL false-positive clips: max net hip-drop / 0.5s in the 4s before alert ===")
ADL_DIR = os.path.join(ROOT, "training", "data", "gmdcsa24_adl_raw_val")
ALERTS = {
    "s1_ADL_01": [2.0], "s2_ADL_03": [9.4], "s2_ADL_15": [6.6],
    "s4_ADL_08": [1.3, 6.7], "s4_ADL_10": [4.7],
}
for clip, timestamps in ALERTS.items():
    path = os.path.join(ADL_DIR, f"{clip}.mp4")
    for ts in timestamps:
        ys, fps = extract_hip_y_sequence(path, max(0.0, ts - 4.0), ts)
        d = max_net_drop(ys, fps)
        print(f"  {clip} @ {ts}s: max_net_drop/0.5s={d:.4f}")

print("\n=== B) Genuine near-bed Fall clips (previously confounded by bbox-overlap): whole clip ===")
FALL_DIR = os.path.join(ROOT, "training", "data", "gmdcsa24_fall_raw")
BED_OVERLAP_SUPPRESSED = ["s2_Fall_04", "s2_Fall_09", "s2_Fall_20", "s3_Fall_02",
                          "s3_Fall_09", "s3_Fall_12", "s3_Fall_13", "s3_Fall_16",
                          "s4_Fall_06", "s4_Fall_17"]
for clip in BED_OVERLAP_SUPPRESSED:
    path = os.path.join(FALL_DIR, f"{clip}.mp4")
    cap = cv2.VideoCapture(path)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    ys, fps = extract_hip_y_sequence(path, 0.0, dur)
    d = max_net_drop(ys, fps)
    print(f"  {clip}: max_net_drop/0.5s={d:.4f}")

print("\n=== C) Non-confounded genuine Fall clips (sanity check, no bed overlap) ===")
OTHER_FALL = ["s1_Fall_02", "s1_Fall_06", "s1_Fall_16", "s2_Fall_02", "s2_Fall_14"]
for clip in OTHER_FALL:
    path = os.path.join(FALL_DIR, f"{clip}.mp4")
    cap = cv2.VideoCapture(path)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    ys, fps = extract_hip_y_sequence(path, 0.0, dur)
    d = max_net_drop(ys, fps)
    print(f"  {clip}: max_net_drop/0.5s={d:.4f}")
