"""Same clips/ground-truth as eval_v3_on_gmdcsa24_val.py (SS20), same scoring, but
through the multi-person path (detect_v3_fall_multi, NUM_POSES=4) instead of the
single-person one -- every one of these 31 clips has exactly one real person in it,
so this directly tests SS25's hypothesis (multi-person mode costs extra false
positives even on solo-person footage because it always searches up to 4 pose
slots) against ground-truth-scored data instead of hand-picked spot checks.

Fall clips: pass if at least one alert fires anywhere in the clip (any tracked person).
ADL clips: pass if zero alerts fire anywhere in the clip (any tracked person, any alert
= false positive -- matches camera_manager.py's "any tracked person falling" logic).
"""
import os
import importlib.util
import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location(
    "v3_fall_detection", os.path.join(ROOT, "app", "detection", "v3_fall_detection.py")
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
V3PoseFallDetector = v3.V3PoseFallDetector
V3MultiPersonFallState = v3.V3MultiPersonFallState
detect_v3_fall_multi = v3.detect_v3_fall_multi
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
    state = V3MultiPersonFallState()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    alerts = []  # (t, track_id, probability)
    last_labels = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        results = detect_v3_fall_multi(frame, state, detector, config=None)
        for track_id, detected, probability, label, _ in results:
            if last_labels.get(track_id) != label:
                if label == "fall":
                    alerts.append((t, track_id, probability))
                last_labels[track_id] = label
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
        print(f"  {name}: {marker}  alerts={[(round(t,1), tid, round(p,2)) for t,tid,p in alerts]}")

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
        print(f"  {name}: {marker}  alerts={[(round(t,1), tid, round(p,2)) for t,tid,p in alerts]}")

    print(f"\nFall clips caught: {fall_caught}/{len(VAL_FALL)} ({fall_caught/len(VAL_FALL):.1%})")
    print(f"ADL clips clean (no false alarm): {adl_clean}/{len(VAL_ADL)} ({adl_clean/len(VAL_ADL):.1%})")


if __name__ == "__main__":
    main()
