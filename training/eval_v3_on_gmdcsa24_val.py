"""Run the actual live production pipeline (detect_v3_fall, with the SS18 smoothing/
hold-state fix) over GMDCSA24's held-out validation clips -- the 31 videos (15 Fall,
16 ADL) split_videos(seed=42) set aside from training. Unlike SS17's offline windowed
F1 (classifier accuracy in isolation) or SS18/SS19's manual/OOPS spot checks, this is
the full live pipeline end-to-end on real, calm, indoor, elderly-fall-relevant footage
-- the domain this project actually targets, not FailArmy's sports/stunt content
(SS19 found only 1/30 FailArmy candidates were actually domain-relevant after
screening, so this reuses existing indoor data instead of searching further).

Fall clips: pass if at least one alert fires anywhere in the clip.
ADL clips: pass if zero alerts fire anywhere in the clip (any alert = false positive).
"""
import os
import glob
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
MODEL_DIR = os.environ.get("TEST_MODEL_DIR", os.path.join(ROOT, "models"))

VAL_FALL = ["s1_Fall_02", "s1_Fall_06", "s1_Fall_16", "s2_Fall_02", "s2_Fall_04",
            "s2_Fall_09", "s2_Fall_14", "s2_Fall_20", "s3_Fall_02", "s3_Fall_09",
            "s3_Fall_12", "s3_Fall_13", "s3_Fall_16", "s4_Fall_06", "s4_Fall_17"]
VAL_ADL = ["s1_ADL_01", "s1_ADL_05", "s1_ADL_11", "s1_ADL_13", "s2_ADL_03", "s2_ADL_07",
           "s2_ADL_13", "s2_ADL_15", "s2_ADL_16", "s2_ADL_18", "s2_ADL_20", "s3_ADL_07",
           "s3_ADL_11", "s4_ADL_07", "s4_ADL_08", "s4_ADL_10"]

FALL_DIR = os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_fall_raw")
ADL_DIR = os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_adl_raw_val")


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


def main():
    detector = V3PoseFallDetector(model_dir=MODEL_DIR)

    print("=== Fall clips (want >=1 alert each) ===")
    fall_caught = 0
    for name in VAL_FALL:
        path = os.path.join(FALL_DIR, name + ".mp4")
        if not os.path.exists(path):
            print(f"  {name}: MISSING FILE")
            continue
        alerts = run_clip(detector, path)
        caught = len(alerts) > 0
        fall_caught += caught
        marker = "OK" if caught else "MISSED"
        print(f"  {name}: {marker}  alerts={[(round(t,1), round(p,2)) for t,p in alerts]}")

    print("\n=== ADL clips (want 0 alerts each) ===")
    adl_clean = 0
    for name in VAL_ADL:
        path = os.path.join(ADL_DIR, name + ".mp4")
        if not os.path.exists(path):
            print(f"  {name}: MISSING FILE")
            continue
        alerts = run_clip(detector, path)
        clean = len(alerts) == 0
        adl_clean += clean
        marker = "OK" if clean else "FALSE ALARM"
        print(f"  {name}: {marker}  alerts={[(round(t,1), round(p,2)) for t,p in alerts]}")

    print(f"\nFall clips caught: {fall_caught}/{len(VAL_FALL)} ({fall_caught/len(VAL_FALL):.1%})")
    print(f"ADL clips clean (no false alarm): {adl_clean}/{len(VAL_ADL)} ({adl_clean/len(VAL_ADL):.1%})")


if __name__ == "__main__":
    main()
