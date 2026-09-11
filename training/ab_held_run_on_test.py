"""A/B the MAX_HELD_RUN cap on the Test/ compilation clips.

check_held_run.py showed the cap barely engages on GMDCSA24 (which is why
eval_v3_on_gmdcsa24_val.py scored identically before/after) but fires constantly on
Test/, where dropouts run 121-359 consecutive frames -- the exact condition SS24 blamed
for high-confidence alerts fired with person_found=False. This counts alerts per clip
with the cap effectively disabled (old behaviour) vs enabled, so the change is measured
where the mechanism actually occurs instead of only where it doesn't.

No ground truth here, so this reports alert COUNTS, not correctness: clips 10/11/6/2 are
mostly-untracked (person visible in a minority of frames), where extra alerts are far
more likely to be the synthetic frozen-window artefact than real detections; 13-17 track
cleanly and act as the control that should not move."""
import os
import importlib.util
import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location(
    "v3_fall_detection", os.path.join(ROOT, "app", "detection", "v3_fall_detection.py")
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

MODEL_DIR = os.path.join(ROOT, "models")
TEST_DIR = os.path.join(ROOT, "Test")
DIRTY = ["10.mp4", "11.mp4", "6.mp4"]        # heaviest dropout (470/577, 427/686, 354/479 missed)
CLEAN = ["13.mp4", "14.mp4", "15.mp4"]       # control, tracks cleanly -- should not move


def run_clip(detector, path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    state = v3.V3FallDetectionState()
    alerts = []
    was = False
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detected, prob, _, _ = v3.detect_v3_fall(frame, state, detector, {})
        if detected and not was:
            alerts.append((i / fps, prob))
        was = detected
        i += 1
    cap.release()
    return alerts


def main():
    detector = v3.V3PoseFallDetector(MODEL_DIR)
    print(f"{'clip':<10} {'alerts OFF':>11} {'alerts ON':>10} {'delta':>7}")
    print("-" * 42)
    totals = {"off": 0, "on": 0}
    for group, names in (("heavy-dropout", DIRTY), ("control", CLEAN)):
        print(f"[{group}]")
        for name in names:
            path = os.path.join(TEST_DIR, name)
            if not os.path.exists(path):
                print(f"  {name:<8} MISSING")
                continue
            v3.MAX_HELD_RUN = 10 ** 9          # effectively the old always-hold behaviour
            off = run_clip(detector, path)
            v3.MAX_HELD_RUN = 5                # the new cap
            on = run_clip(detector, path)
            totals["off"] += len(off)
            totals["on"] += len(on)
            print(f"  {name:<8} {len(off):>11} {len(on):>10} {len(on)-len(off):>+7}")
        print()
    print(f"{'TOTAL':<10} {totals['off']:>11} {totals['on']:>10} {totals['on']-totals['off']:>+7}")


if __name__ == "__main__":
    main()
