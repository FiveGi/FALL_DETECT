"""Run the real production v3 pipeline over the 10 clips picked by
select_oops_test_clips (data/oops_test_selection.json) and score every alert
against OmniFall's real human-annotated fall segment boundaries -- an
objective ground truth, not manual eyeballing like the earlier Test/ clips.

An alert counts as a true positive if it falls within [seg_start - TOLERANCE,
seg_end + TOLERANCE] of any labeled fall/fallen segment in that video;
otherwise it's a false positive. A labeled segment with no alert overlapping
it (with tolerance) is a missed detection.
"""
import os
import sys
import json
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
MODEL_DIR = os.path.join(ROOT, "models")

TOLERANCE_S = 1.5  # generous, since we care about "did it catch the fall", not exact framing


def run_clip(detector, path):
    state = V3FallDetectionState()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    alerts = []
    last_label = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        detected, probability, label, _ = detect_v3_fall(frame, state, detector, config=None)
        if label != last_label:
            if label == "fall":
                alerts.append((t, probability))
            last_label = label
        frame_idx += 1
    cap.release()
    return alerts


def score(alerts, fall_segments):
    matched_segments = set()
    tp, fp = 0, 0
    for t, p in alerts:
        hit = False
        for i, (s, e) in enumerate(fall_segments):
            if s - TOLERANCE_S <= t <= e + TOLERANCE_S:
                hit = True
                matched_segments.add(i)
        if hit:
            tp += 1
        else:
            fp += 1
    missed = [seg for i, seg in enumerate(fall_segments) if i not in matched_segments]
    return tp, fp, missed


def main():
    with open(os.path.join(os.path.dirname(__file__), "data", "oops_test_selection.json")) as f:
        clips = json.load(f)

    detector = V3PoseFallDetector(model_dir=MODEL_DIR)

    total_tp, total_fp, total_segs, total_missed = 0, 0, 0, 0
    for c in clips:
        clip_path = os.path.join(os.path.dirname(__file__), c["path"])
        alerts = run_clip(detector, clip_path)
        segs = c["fall_segments"]
        tp, fp, missed = score(alerts, segs)
        total_tp += tp
        total_fp += fp
        total_segs += len(segs)
        total_missed += len(missed)
        print(f"{c['stem'][:50]:<50} dur={c['duration_s']:5.1f}s  "
              f"real_fall_segments={len(segs)}  alerts={len(alerts)}  TP={tp}  FP={fp}  missed={len(missed)}")
        if alerts:
            print(f"    alerts: {[(round(t,1), round(p,2)) for t,p in alerts]}")
        print(f"    real fall segments (s): {[(round(s,1), round(e,1)) for s,e in segs]}")
        if missed:
            print(f"    MISSED segments: {[(round(s,1), round(e,1)) for s,e in missed]}")

    print()
    print(f"TOTALS across {len(clips)} clips ({sum(c['duration_s'] for c in clips):.0f}s of real footage):")
    print(f"  Real fall segments: {total_segs}")
    print(f"  Alerts fired: {total_tp + total_fp}  (TP={total_tp}, FP={total_fp})")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else float("nan")
    seg_recall = (total_segs - total_missed) / total_segs if total_segs else float("nan")
    print(f"  Precision (of alerts, how many were real): {precision:.1%}")
    print(f"  Segment recall (of real falls, how many got at least one alert): {seg_recall:.1%}")
    print(f"  Fully missed fall segments: {total_missed}")


if __name__ == "__main__":
    main()
