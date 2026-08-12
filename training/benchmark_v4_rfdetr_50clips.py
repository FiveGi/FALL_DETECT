"""Benchmarks the v4 (RF-DETR) fall detector against the same 50-clip
CAUCAFall slice used throughout the v3->v4 comparison and tuning work
(see SKILL.md). Runs the literal production detect_v4_fall() function,
not a re-implementation, so results reflect what actually ships.

Usage:
    python training/benchmark_v4_rfdetr_50clips.py

Adjust CAUCAFALL_ROOT below if the dataset lives somewhere else on your
machine. Expects the standard CAUCAFall layout:
    <CAUCAFALL_ROOT>/Subject.N/<Activity>/*.png
"""
import glob
import importlib.util
import os
import time

import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAUCAFALL_ROOT = os.environ.get(
    "CAUCAFALL_ROOT", "d:/project/PROJECT/Dataset CAUCAFall/CAUCAFall"
)

spec = importlib.util.spec_from_file_location(
    "v4_fall_detection_rfdetr",
    os.path.join(REPO_ROOT, "app", "detection", "v4_fall_detection_rfdetr.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

detector = mod.V4RFDETRFallDetector(model_dir=os.path.join(REPO_ROOT, "models"))

FALL_ACTIVITIES = ["Fall backwards", "Fall forward", "Fall left", "Fall right", "Fall sitting"]
ADL_ACTIVITIES = ["Hop", "Kneel", "Pick up object", "Sit down", "Walk"]
# This specific 5-subject slice is what every sensitivity/specificity number
# in SKILL.md refers to -- 5 subjects x 10 activities (5 fall + 5 ADL) = 50 clips.
SUBJECTS = ["Subject.2", "Subject.4", "Subject.6", "Subject.8", "Subject.10"]


def run_clip(subject, activity):
    frames = sorted(glob.glob(f"{CAUCAFALL_ROOT}/{subject}/{activity}/*.png"))
    if not frames:
        return None
    state = mod.V4FallDetectionState()
    any_fall = False
    max_prob = 0.0
    for fp in frames:
        frame = cv2.imread(fp)
        detected, prob, label, _ = mod.detect_v4_fall(frame, state, detector, config={})
        if detected:
            any_fall = True
        max_prob = max(max_prob, prob)
    return any_fall, max_prob, len(frames)


results = []
t0 = time.time()
n = 0
for subject in SUBJECTS:
    for activity in FALL_ACTIVITIES + ADL_ACTIVITIES:
        expect = activity in FALL_ACTIVITIES
        r = run_clip(subject, activity)
        n += 1
        if r is None:
            print(f"[{n:2d}/50] {subject}/{activity}: NOT FOUND under {CAUCAFALL_ROOT}")
            continue
        flagged, max_prob, nframes = r
        status = "OK" if flagged == expect else "MISS" if expect else "FALSE-ALARM"
        print(f"[{n:2d}/50] {subject}/{activity} ({nframes}f): expect_fall={expect} flagged={flagged} max_prob={max_prob:.3f} -> {status}", flush=True)
        results.append((subject, activity, expect, flagged, max_prob, status))

elapsed = time.time() - t0
print(f"\nTotal time: {elapsed:.1f}s for {len(results)} clips")

fall_cases = [r for r in results if r[2]]
adl_cases = [r for r in results if not r[2]]
tp = sum(1 for r in fall_cases if r[3])
fn = sum(1 for r in fall_cases if not r[3])
fp = sum(1 for r in adl_cases if r[3])
tn = sum(1 for r in adl_cases if not r[3])

print(f"\nFall clips: {len(fall_cases)}  -> caught (TP)={tp}  missed (FN)={fn}  sensitivity={100*tp/max(len(fall_cases),1):.1f}%")
print(f"ADL clips:  {len(adl_cases)}  -> correct (TN)={tn}  false-alarm (FP)={fp}  specificity={100*tn/max(len(adl_cases),1):.1f}%")

if fn:
    print("\nMISSED falls:")
    for r in fall_cases:
        if not r[3]:
            print(" ", r[0], r[1], f"max_prob={r[4]:.3f}")
if fp:
    print("\nFALSE ALARMS on ADL:")
    for r in adl_cases:
        if r[3]:
            print(" ", r[0], r[1], f"max_prob={r[4]:.3f}")
