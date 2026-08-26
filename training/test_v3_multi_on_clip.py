"""Same idea as test_v3_on_clip.py but exercises the new multi-person path
(detect_v3_fall_multi / V3MultiPersonFallState / PersonTracker) instead of the
single-person one, so multi-person tracking can be checked against real footage
before trusting it in camera_manager.py.

Usage: python test_v3_multi_on_clip.py <path/to/clip.mp4>
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
V3MultiPersonFallState = v3.V3MultiPersonFallState
detect_v3_fall_multi = v3.detect_v3_fall_multi
WINDOW_SIZE = v3.WINDOW_SIZE
THRESHOLD = v3.THRESHOLD

MODEL_DIR = os.environ.get("TEST_MODEL_DIR", os.path.join(ROOT, "models"))
RESULTS_ROOT = os.environ.get("TEST_RESULTS_ROOT", r"D:\project\PROJECT\Test_Results_Multi")

TRACK_COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255), (255, 255, 0)]

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_person(frame, kpts17, color, track_id, probability, label, detected):
    h, w = frame.shape[:2]
    pts = kpts17[:, :2] * [w, h]
    for a, b in SKELETON_EDGES:
        pa, pb = pts[a], pts[b]
        if (pa == 0).all() or (pb == 0).all():
            continue
        cv2.line(frame, tuple(pa.astype(int)), tuple(pb.astype(int)), color, 2)
    head = pts[0].astype(int)
    tag_color = (0, 0, 255) if detected else color
    cv2.putText(frame, f"#{track_id} {label} p={probability:.2f}", (head[0] - 20, head[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_color, 2)
    if detected:
        cv2.circle(frame, tuple(head), 8, (0, 0, 255), -1)


def main():
    clip_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\project\PROJECT\Test\1.mp4"
    clip_name = os.path.splitext(os.path.basename(clip_path))[0]
    out_dir = os.path.join(RESULTS_ROOT, clip_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "annotated_multi.mp4")
    summary_path = os.path.join(out_dir, "summary_multi.txt")

    print(f"Loading V3 pose fall detector (multi-person, NUM_POSES={v3.NUM_POSES}) from {MODEL_DIR} ...")
    detector = V3PoseFallDetector(model_dir=MODEL_DIR)
    state = V3MultiPersonFallState()

    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Clip: {clip_path}\n  {w}x{h} @ {fps:.1f}fps, {n_frames} frames, {n_frames/fps:.1f}s")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    max_people_seen = 0
    all_track_ids_ever = set()
    alerts = []  # (t, track_id, probability)
    last_labels = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        results = detect_v3_fall_multi(frame, state, detector, config=None)
        # results includes tracks held through a brief dropout (up to MAX_MISSED_FRAMES,
        # by design -- see SS24), not just people actually visible this exact frame.
        # Drawing all of them every frame draws stale, frozen-in-place skeletons on top
        # of current ones -- that's the "messy lines" -- so only draw ones truly seen now.
        seen_now = {tid for tid, t_ in state.tracker.tracks.items() if t_["missed"] == 0}
        currently_visible = [r for r in results if r[0] in seen_now]
        max_people_seen = max(max_people_seen, len(currently_visible))
        annotated = frame.copy()
        for track_id, detected, probability, label, centroid in currently_visible:
            all_track_ids_ever.add(track_id)
            color = TRACK_COLORS[track_id % len(TRACK_COLORS)]
            kpts = state.person_states[track_id].raw_buffer[-1] if state.person_states[track_id].raw_buffer else np.zeros((17, 3))
            draw_person(annotated, kpts, color, track_id, probability, label, detected)

        for track_id, detected, probability, label, centroid in results:
            if last_labels.get(track_id) != label:
                if label == "fall":
                    print(f"  t={t:5.1f}s track#{track_id} -> FALL p={probability:.3f}")
                    alerts.append((t, track_id, probability))
                last_labels[track_id] = label

        cv2.putText(annotated, f"people visible: {len(currently_visible)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    summary = (
        f"Clip: {clip_path}\n"
        f"Max people tracked simultaneously: {max_people_seen}\n"
        f"Total distinct track IDs ever assigned: {len(all_track_ids_ever)}\n"
        f"Fall alerts: {len(alerts)}\n" +
        "\n".join(f"  t={t:.1f}s track#{tid} p={p:.3f}" for t, tid, p in alerts)
    )
    print("\n" + summary)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\nAnnotated video: {out_path}")


if __name__ == "__main__":
    main()
