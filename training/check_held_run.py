"""How often does a tracking dropout actually run longer than MAX_HELD_RUN?

The SS24 false-alarm mechanism (window packed with repeated copies of one pose) needs
LONG runs of consecutive misses to form. eval_v3_on_gmdcsa24_val.py showed the new cap
changes nothing on GMDCSA24, which could mean either "safe" or "never fired" -- this
measures which, per clip, so the distinction isn't left to assumption."""
import os
import glob
import importlib.util
from collections import Counter
import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location(
    "v3_fall_detection", os.path.join(ROOT, "app", "detection", "v3_fall_detection.py")
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

MODEL_DIR = os.path.join(ROOT, "models")
MAX_HELD_RUN = v3.MAX_HELD_RUN


def scan(detector, path):
    cap = cv2.VideoCapture(path)
    run = 0
    runs = []
    frames = 0
    misses = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        _, found = detector.extract_keypoints(frame)
        if found:
            if run:
                runs.append(run)
            run = 0
        else:
            misses += 1
            run += 1
    if run:
        runs.append(run)
    cap.release()
    return frames, misses, runs


def main():
    detector = v3.V3PoseFallDetector(MODEL_DIR)
    dirs = [
        os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_fall_raw"),
        os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_adl_raw_val"),
        os.path.join(ROOT, "Test"),
    ]
    print(f"MAX_HELD_RUN = {MAX_HELD_RUN}\n")
    grand = Counter()
    for d in dirs:
        vids = sorted(glob.glob(os.path.join(d, "*.mp4")))[:16]
        if not vids:
            continue
        print(f"=== {os.path.basename(d)} ({len(vids)} clips) ===")
        for v in vids:
            frames, misses, runs = scan(detector, v)
            over = [r for r in runs if r > MAX_HELD_RUN]
            grand["clips"] += 1
            grand["over_clips"] += 1 if over else 0
            grand["over_runs"] += len(over)
            longest = max(runs) if runs else 0
            flag = f"  <-- {len(over)} run(s) over cap" if over else ""
            print(f"  {os.path.basename(v):<22} frames={frames:4d} miss={misses:4d} "
                  f"longest_run={longest:3d}{flag}")
        print()
    print(f"Clips with at least one over-cap dropout: {grand['over_clips']}/{grand['clips']}")
    print(f"Total over-cap dropout runs: {grand['over_runs']}")


if __name__ == "__main__":
    main()
