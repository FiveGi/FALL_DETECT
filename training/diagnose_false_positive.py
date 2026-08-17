"""Dump every frame's raw diagnostic data (not just label transitions) around a
specific timestamp, to see what the classifier's input window actually looked like
right before a false-positive alert fired.

Usage: python diagnose_false_positive.py <clip.mp4> <alert_time_s>
"""
import os
import sys
import importlib.util
import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location(
    "v3_fall_detection", os.path.join(ROOT, "app", "detection", "v3_fall_detection.py")
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
V3PoseFallDetector = v3.V3PoseFallDetector
V3FallDetectionState = v3.V3FallDetectionState
detect_v3_fall = v3.detect_v3_fall
WINDOW_SIZE = v3.WINDOW_SIZE
MODEL_DIR = os.path.join(ROOT, "models")


def main():
    clip_path = sys.argv[1]
    target_t = float(sys.argv[2])
    window_before_s = 3.0

    detector = V3PoseFallDetector(model_dir=MODEL_DIR)
    state = V3FallDetectionState()

    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frame_idx = 0
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        kpts, person_found = detector.extract_keypoints(frame)
        detected, probability, label, _ = detect_v3_fall(frame, state, detector, config=None)

        if target_t - window_before_s <= t <= target_t + 0.5:
            # torso tilt: angle of the shoulder-center -> hip-center vector from vertical
            lsh, rsh, lhip, rhip = kpts[5], kpts[6], kpts[11], kpts[12]
            shoulder_c = (lsh[:2] + rsh[:2]) / 2
            hip_c = (lhip[:2] + rhip[:2]) / 2
            dx, dy = (hip_c - shoulder_c)
            tilt_deg = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))  # 0=upright, 90=horizontal
            avg_vis = kpts[:, 2].mean()
            rows.append((t, frame_idx, person_found, probability, label, tilt_deg, avg_vis))

        frame_idx += 1
        if t > target_t + 0.5:
            break

    cap.release()

    print(f"{'t':>6} {'frame':>6} {'person':>7} {'prob':>6} {'label':>12} {'tilt_deg':>9} {'avg_vis':>8}")
    for t, fi, pf, p, lbl, tilt, vis in rows:
        marker = " <-- ALERT" if abs(t - target_t) < 1.0/fps else ""
        print(f"{t:6.2f} {fi:6d} {str(pf):>7} {p:6.3f} {lbl:>12} {tilt:9.1f} {vis:8.3f}{marker}")


if __name__ == "__main__":
    main()
