"""Check that the deployed model, the runtime settings and the training defaults still agree.

Four numbers on this system are really one decision, and they live in four places:

  1. the ONNX file's input shape          models/fall_classifier_v3.onnx
  2. the runtime window                   app/detection/v3_fall_detection.WINDOW_SIZE
  3. the camera loop's sampling rate      V3_TARGET_FPS in docker-compose.gpu.yml
  4. the training window and stride       training/dataset.WINDOW_SIZE / TEMPORAL_STRIDE

Change one alone and the system still starts, still passes the API and browser smoke tests, and
quietly behaves like a detector nobody measured. That has already happened twice here: the
training defaults spent three generations producing MediaPipe-era 30-frame models that the
runtime could not load, and the alerting threshold outlived the window it was chosen for.

The relationship the numbers have to satisfy:

    runtime window == ONNX input frames == training window
    the live window covers a duration that has actually been measured

The second rule used to read `training stride * target fps == 30`, so that a window covered the
same real time in training and in deployment. That sounded obviously right and was measured to
be wrong (SS61): feeding the deployed 15-frame model more frames than the 15 fps it was trained
for catches more falls, not fewer -- URFD 41/60 at 15 fps against 45/60 at 20 fps, with the
held-out false-alarm rate unchanged, confirmed on the reserved half of URFD. A model trained
for the shorter window instead (12 frames, 0.80s at 15 fps) was tried and is worse on both
axes, 34/60, so it is not the window's duration that matters.

What is still true is that an untested duration is an untested detector, so this checks the
live window against the range that has been measured end to end rather than against a formula.

Runs without Docker, a database or a GPU.

Usage:
    python tools/check_config_coherence.py
"""
import importlib.util
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FPS = 30.0   # every training dataset here is 30fps footage

# The live window durations that have actually been run end to end on URFD, GMDCSA24 and the
# Test/ clips: 0.65s (15 frames at 23 fps) to 1.00s (15 frames at 15 fps). Deployed is 0.75s.
# Outside this band nothing has been measured, and an unmeasured detector is the thing this
# script exists to catch. Widen it by measuring, not by editing it.
MEASURED_SECONDS = (0.65, 1.00)

# What one 1080p camera sustains on this machine, measured uncapped (V3_TARGET_FPS=0) by
# reading the camera loop's own periodic log. Pinning at or above this makes the achieved rate
# depend on machine load again, which is what pinning it exists to prevent.
SUSTAINED_FPS = 23.4

failures = []


def check(label, ok, detail=''):
    print(f'{"PASS" if ok else "FAIL"}  {label}{"  " + detail if detail else ""}')
    if not ok:
        failures.append(label)


def load_module(name, path, env=None):
    if env:
        os.environ.update(env)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def onnx_input_frames(path):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    shape = sess.get_inputs()[0].shape
    return int(shape[1]), int(shape[2])


def compose_target_fps():
    text = open(os.path.join(ROOT, 'docker-compose.gpu.yml'), encoding='utf-8').read()
    m = re.search(r'V3_TARGET_FPS=([0-9.]+)', text)
    return float(m.group(1)) if m else None


def main():
    v3 = load_module('v3', os.path.join(ROOT, 'app', 'detection', 'v3_fall_detection.py'))
    sys.path.insert(0, os.path.join(ROOT, 'training'))
    ds = load_module('ds', os.path.join(ROOT, 'training', 'dataset.py'))
    md = load_module('md', os.path.join(ROOT, 'training', 'model.py'))

    frames, features = onnx_input_frames(os.path.join(ROOT, 'models', 'fall_classifier_v3.onnx'))
    target_fps = compose_target_fps()

    print(f'deployed ONNX      {frames} frames x {features} features')
    print(f'runtime            WINDOW_SIZE {v3.WINDOW_SIZE}, threshold {v3.THRESHOLD}, '
          f'smoothing {v3.SMOOTH_NEED} of {v3.SMOOTH_OF}')
    print(f'training           WINDOW_SIZE {ds.WINDOW_SIZE}, TEMPORAL_STRIDE {ds.TEMPORAL_STRIDE}, '
          f'{md.FEATURES_PER_FRAME} features')
    print(f'camera loop        V3_TARGET_FPS {target_fps}\n')

    check('runtime window matches the deployed ONNX',
          v3.WINDOW_SIZE == frames, f'{v3.WINDOW_SIZE} vs {frames}')
    check('training window matches the deployed ONNX',
          ds.WINDOW_SIZE == frames, f'{ds.WINDOW_SIZE} vs {frames}')
    check('feature count matches the deployed ONNX',
          md.FEATURES_PER_FRAME == features, f'{md.FEATURES_PER_FRAME} vs {features}')
    check('camera rate is pinned, not left to the hardware', bool(target_fps),
          '' if target_fps else 'V3_TARGET_FPS missing from docker-compose.gpu.yml')
    if target_fps:
        train_seconds = ds.WINDOW_SIZE * ds.TEMPORAL_STRIDE / SOURCE_FPS
        live_seconds = v3.WINDOW_SIZE / target_fps
        check('the live window covers a duration that has been measured',
              MEASURED_SECONDS[0] - 1e-6 <= live_seconds <= MEASURED_SECONDS[1] + 1e-6,
              f'{live_seconds:.2f}s live ({target_fps:.0f} fps), measured range '
              f'{MEASURED_SECONDS[0]:.2f}-{MEASURED_SECONDS[1]:.2f}s')
        check('the camera rate is one this machine sustains',
              target_fps <= SUSTAINED_FPS,
              f'pinned at {target_fps:.0f} fps, measured ceiling {SUSTAINED_FPS:.1f} fps '
              f'for one 1080p camera')
        # Reported, not asserted: training and live no longer have to match, but the ratio is
        # the first thing to look at if accuracy moves for no other reason.
        print(f'      a window spans {train_seconds:.2f}s in training and {live_seconds:.2f}s '
              f'live ({live_seconds / train_seconds:.2f}x)')

    print()
    if failures:
        print(f'{len(failures)} check(s) failed -- the deployed detector is not the one that '
              f'was measured')
        return 1
    print('all checks passed -- model, runtime, camera rate and training defaults agree')
    return 0


if __name__ == '__main__':
    sys.exit(main())
