"""Does the deployed detector still work at the frame rate production actually achieves?

Every accuracy number in SKILL.md was measured offline, reading a video file frame by frame
-- the detector sees all ~25-30 fps. Live cameras do not work that way: the loop in
camera_manager.process_v2_fall_detection has no stride, and `CAP_PROP_BUFFERSIZE=1` makes
each cap.read() hand back the NEWEST frame, so whatever the pipeline cannot keep up with is
silently dropped. Measured with training/bench_deployed_v3.py, the deployed pipeline runs at
~2.4 fps inside the production container limits, which means roughly every 12th frame reaches
the detector.

That matters because the classifier reads a 30-frame window. At 25 fps that window is ~1.2s
-- about the duration of a fall, which is what it was trained on. At 2.4 fps the same 30
frames span ~12 seconds, so the "fall" it is asked to recognise is stretched an order of
magnitude and the velocity features are computed across gaps 12x wider than in training.

This runs the real detect_v3_fall over the same GMDCSA24 held-out clips as
eval_v3_on_gmdcsa24_val.py, but feeding only every Nth frame, so the offline number and the
production-rate number are directly comparable.

Usage:
    python training/eval_v3_frame_drop.py --strides 1 6 12
"""
import argparse
import importlib.util
import os

import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location(
    "v3_fall_detection", os.path.join(ROOT, "app", "detection", "v3_fall_detection.py")
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

MODEL_DIR = os.environ.get("TEST_MODEL_DIR", os.path.join(ROOT, "models"))

VAL_FALL = ["s1_Fall_02", "s1_Fall_06", "s1_Fall_16", "s2_Fall_02", "s2_Fall_04",
            "s2_Fall_09", "s2_Fall_14", "s2_Fall_20", "s3_Fall_02", "s3_Fall_09",
            "s3_Fall_12", "s3_Fall_13", "s3_Fall_16", "s4_Fall_06", "s4_Fall_17"]
VAL_ADL = ["s1_ADL_01", "s1_ADL_05", "s1_ADL_11", "s1_ADL_13", "s2_ADL_03", "s2_ADL_07",
           "s2_ADL_13", "s2_ADL_15", "s2_ADL_16", "s2_ADL_18", "s2_ADL_20", "s3_ADL_07",
           "s3_ADL_11", "s4_ADL_07", "s4_ADL_08", "s4_ADL_10"]

# The train50 lists, copied from eval_v3_on_gmdcsa24_train50.py, so a configuration tuned
# on VAL can be re-checked on clips it was never selected against -- the same held-out
# discipline the rest of this project uses.
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
ADL_DIR = os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_adl_raw_val")
ADL_DIR_TRAIN50 = os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_adl_raw_train50")


def run_clip(detector, path, stride, threshold=None, multi=False):
    """multi=True runs detect_v3_fall_multi -- the entry point camera_manager actually calls.
    It tracks every person in frame separately, so its alerting behaviour is not the same as
    the single-person path most eval scripts use, and a frame-rate result measured on one
    does not automatically hold for the other."""
    state = v3.V3MultiPersonFallState() if multi else v3.V3FallDetectionState()
    cap = cv2.VideoCapture(path)
    frame_idx = 0
    alerts = 0
    last_label = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Drop frames the way a live camera does -- the detector simply never sees them,
        # rather than seeing them late.
        if frame_idx % stride == 0:
            if multi:
                results = v3.detect_v3_fall_multi(frame, state, detector, config=None,
                                                  threshold=threshold)
                # Same rule camera_manager uses to decide "this frame is a fall": any
                # tracked person detected as falling.
                label = "fall" if any(r[1] for r in results) else "no_fall"
            else:
                _, _, label, _ = v3.detect_v3_fall(frame, state, detector, config=None,
                                                   threshold=threshold)
            if label != last_label:
                if label == "fall":
                    alerts += 1
                last_label = label
        frame_idx += 1
    cap.release()
    return alerts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strides', type=int, nargs='+', default=[1, 6, 12])
    parser.add_argument('--thresholds', type=float, nargs='+', default=[None])
    parser.add_argument('--dataset', choices=['val', 'train50'], default='val')
    parser.add_argument('--multi', action='store_true',
                        help='use detect_v3_fall_multi, the production entry point')
    args = parser.parse_args()

    falls = VAL_FALL if args.dataset == 'val' else TRAIN_FALL
    adls = VAL_ADL if args.dataset == 'val' else TRAIN_ADL
    adl_dir = ADL_DIR if args.dataset == 'val' else ADL_DIR_TRAIN50

    detector = v3.V3PoseFallDetector(model_dir=MODEL_DIR)

    print(f"{'stride':>7} {'approx fps':>11} {'falls caught':>13} {'ADL clean':>11}")
    for stride in args.strides:
        for thr in args.thresholds:
            caught = 0
            for name in falls:
                path = os.path.join(FALL_DIR, name + ".mp4")
                if os.path.exists(path) and run_clip(detector, path, stride, thr, args.multi) > 0:
                    caught += 1
            clean = 0
            for name in adls:
                path = os.path.join(adl_dir, name + ".mp4")
                if os.path.exists(path) and run_clip(detector, path, stride, thr, args.multi) == 0:
                    clean += 1
            label = "default" if thr is None else f"{thr:.2f}"
            print(f"{stride:>7} {25.0 / stride:>10.1f}  thr={label:>7}  {caught:>6}/{len(falls)}      {clean:>4}/{len(adls)}", flush=True)


if __name__ == '__main__':
    main()
