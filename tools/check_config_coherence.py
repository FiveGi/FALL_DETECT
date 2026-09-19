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
    training stride * target fps == the source frame rate the datasets were recorded at (30)

so that a window covers the same amount of real time in training and in deployment.

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
        # A window must cover the same real time in training and in deployment, otherwise the
        # motion the classifier sees live is stretched or compressed against what it learned.
        expected_stride = SOURCE_FPS / target_fps
        check('training stride matches the pinned camera rate',
              abs(ds.TEMPORAL_STRIDE - expected_stride) < 1e-6,
              f'stride {ds.TEMPORAL_STRIDE} implies {SOURCE_FPS / ds.TEMPORAL_STRIDE:.1f} fps, '
              f'pinned at {target_fps:.1f}')
        train_seconds = ds.WINDOW_SIZE * ds.TEMPORAL_STRIDE / SOURCE_FPS
        live_seconds = v3.WINDOW_SIZE / target_fps
        check('a window covers the same real time in training and live',
              abs(train_seconds - live_seconds) < 1e-6,
              f'{train_seconds:.2f}s training vs {live_seconds:.2f}s live')

    print()
    if failures:
        print(f'{len(failures)} check(s) failed -- the deployed detector is not the one that '
              f'was measured')
        return 1
    print('all checks passed -- model, runtime, camera rate and training defaults agree')
    return 0


if __name__ == '__main__':
    sys.exit(main())
