"""Run the real production v3 fall-detection pipeline (same code path as
camera_manager.py's process_v2_fall_detection) over a test clip and produce, in
Test_Results/<clip_name>/ (one subfolder per clip, so results from different
clips never overwrite each other and are easy to click through):
  - annotated.mp4 -- skeleton + probability + label overlay, full clip
  - summary.txt -- plain-text list of every alert (readable without opening the video)
  - alert_frames/ -- one still image per alert moment, for an instant visual check

Usage: python test_v3_on_clip.py <path/to/clip.mp4>
"""
import os
import sys
import importlib.util
import cv2
import numpy as np

# Load v3_fall_detection.py directly by path -- importing it via the `app` package
# triggers app/__init__.py's Flask/SQLAlchemy imports, which aren't installed in this
# local training env (only inside the Docker container).
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
THRESHOLD = v3.THRESHOLD

MODEL_DIR = os.path.join(ROOT, "models")
RESULTS_ROOT = r"D:\project\PROJECT\Test_Results"

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # shoulders/arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]


def draw_overlay(frame, kpts17, person_found, probability, label, detected):
    h, w = frame.shape[:2]
    if person_found:
        pts = kpts17[:, :2] * [w, h]
        for a, b in SKELETON_EDGES:
            pa, pb = pts[a], pts[b]
            cv2.line(frame, tuple(pa.astype(int)), tuple(pb.astype(int)), (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 200, 255), -1)

    color = (0, 0, 255) if detected else (0, 200, 0)
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, f"{label}  p={probability:.2f}  person={person_found}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if detected:
        cv2.rectangle(frame, (2, 2), (w - 2, h - 2), (0, 0, 255), 6)
    return frame


def main():
    clip_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\project\PROJECT\Test\1d3c6dac-58bb-4ace-b22f-cbbf0c98d0cc.mp4"
    clip_name = os.path.splitext(os.path.basename(clip_path))[0]
    out_dir = os.path.join(RESULTS_ROOT, clip_name)
    frames_dir = os.path.join(out_dir, "alert_frames")
    os.makedirs(frames_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "annotated.mp4")
    summary_path = os.path.join(out_dir, "summary.txt")

    print(f"Loading V3 pose fall detector from {MODEL_DIR} ...")
    detector = V3PoseFallDetector(model_dir=MODEL_DIR)
    state = V3FallDetectionState()

    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Clip: {clip_path}")
    print(f"  {w}x{h} @ {fps:.1f}fps, {n_frames} frames, {n_frames/fps:.1f}s")
    print(f"  window_size={WINDOW_SIZE} frames (~{WINDOW_SIZE/fps:.1f}s), threshold={THRESHOLD}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    last_label = None
    events = []
    max_prob_seen = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        kpts, person_found = detector.extract_keypoints(frame)
        detected, probability, label, _ = detect_v3_fall(frame, state, detector, config=None)
        max_prob_seen = max(max_prob_seen, probability)

        annotated = draw_overlay(frame.copy(), kpts, person_found, probability, label, detected)
        writer.write(annotated)

        if label != last_label:
            print(f"  t={t:5.1f}s frame={frame_idx:4d}  -> {label:12s} p={probability:.3f} person_found={person_found}")
            events.append((t, label, probability))
            if label == "fall":
                cv2.imwrite(os.path.join(frames_dir, f"t={t:06.1f}s_p={probability:.2f}.jpg"), annotated)
            last_label = label

        frame_idx += 1

    cap.release()
    writer.release()

    fall_events = [e for e in events if e[1] == "fall"]
    lines = []
    lines.append(f"Clip: {clip_path}")
    lines.append(f"{w}x{h} @ {fps:.1f}fps, {n_frames} frames, {n_frames/fps:.1f}s")
    lines.append(f"Max fall-probability seen anywhere in clip: {max_prob_seen:.3f} (alert threshold={THRESHOLD})")
    lines.append(f"Fall alerts raised: {len(fall_events)}")
    for t, label, p in fall_events:
        lines.append(f"  ALERT at t={t:.1f}s, confidence={p:.3f}  (see alert_frames/t={t:06.1f}s_p={p:.2f}.jpg)")
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(f"\nResults folder: {out_dir}")
    print(f"  annotated.mp4   - full video with skeleton/label overlay")
    print(f"  summary.txt     - alert list (quick read, no video player needed)")
    print(f"  alert_frames/   - one still image per alert (instant visual check)")


if __name__ == "__main__":
    main()
