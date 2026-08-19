"""Same methodology as eval_v3_on_gmdcsa24_val.py (SS20), but on 50 clips (25 Fall,
25 ADL) picked from GMDCSA24's *training* split instead of the held-out 31. These
clips WERE in the pool the deployed model trained on -- this isn't a generalization
test, it's a different question: does the bed-lying false-alarm pattern (SS20/SS21)
show up even on clips the model has seen, or only on unseen ones? If it fails here
too, that's a stronger signal the pattern itself is hard, not just under-generalized.
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
V3FallDetectionState = v3.V3FallDetectionState
detect_v3_fall = v3.detect_v3_fall
MODEL_DIR = os.environ.get("TEST_MODEL_DIR", os.path.join(ROOT, "models"))

TRAIN_FALL = ['s4_Fall_13', 's3_Fall_04', 's2_Fall_21', 's1_Fall_07', 's4_Fall_08',
              's3_Fall_20', 's2_Fall_17', 's2_Fall_08', 's3_Fall_14', 's1_Fall_05',
              's2_Fall_10', 's1_Fall_10', 's2_Fall_19', 's2_Fall_11', 's4_Fall_11',
              's4_Fall_02', 's1_Fall_09', 's3_Fall_05', 's1_Fall_01', 's2_Fall_07',
              's4_Fall_09', 's1_Fall_03', 's4_Fall_01', 's1_Fall_13', 's4_Fall_10']
TRAIN_ADL = ['s4_ADL_16', 's4_ADL_15', 's2_ADL_08', 's3_ADL_12', 's3_ADL_22',
             's3_ADL_16', 's3_ADL_20', 's1_ADL_16', 's4_ADL_17', 's2_ADL_05',
             's3_ADL_10', 's2_ADL_09', 's2_ADL_04', 's1_ADL_09', 's4_ADL_18',
             's4_ADL_04', 's4_ADL_03', 's3_ADL_09', 's3_ADL_13', 's2_ADL_11',
             's3_ADL_04', 's4_ADL_01', 's4_ADL_11', 's3_ADL_05', 's3_ADL_02']

FALL_DIR = os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_fall_raw")
ADL_DIR = os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_adl_raw_train50")


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
    for name in TRAIN_FALL:
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
    for name in TRAIN_ADL:
        path = os.path.join(ADL_DIR, name + ".mp4")
        if not os.path.exists(path):
            print(f"  {name}: MISSING FILE")
            continue
        alerts = run_clip(detector, path)
        clean = len(alerts) == 0
        adl_clean += clean
        marker = "OK" if clean else "FALSE ALARM"
        print(f"  {name}: {marker}  alerts={[(round(t,1), round(p,2)) for t,p in alerts]}")

    print(f"\nFall clips caught: {fall_caught}/{len(TRAIN_FALL)} ({fall_caught/len(TRAIN_FALL):.1%})")
    print(f"ADL clips clean (no false alarm): {adl_clean}/{len(TRAIN_ADL)} ({adl_clean/len(TRAIN_ADL):.1%})")


if __name__ == "__main__":
    main()
