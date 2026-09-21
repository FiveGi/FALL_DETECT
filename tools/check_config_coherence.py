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

# Every (input size, window, frame rate) this project has actually run end to end, with what
# it scored, so the check below can say "this exact configuration was measured" instead of
# "this configuration satisfies a formula". Formulas were how an assumption that turned out to
# be false (training and live windows must cover the same real time) got enforced for weeks.
#
# Add a row by measuring it -- scratchpad/rule_sweep_perclip.py over the 220 lab clips -- not
# by editing the table to make a check pass.
MEASURED = {
    # GPU profile: RTX 4070 Ti Super, one 1080p camera, uncapped ceiling 23.4 fps.
    (960, 15, 15.0): 'GPU: URFD 41/60 falls, 34/40 clean; held-out clean 41/56',
    (960, 15, 18.0): 'GPU: URFD 45/60 falls, 32/40 clean',
    (960, 15, 20.0): 'GPU: URFD 45/60 falls, 33/40 clean; held-out clean 41/56 (deployed)',
    (960, 15, 23.0): 'GPU: URFD 48/60 falls, 33/40 clean',
    # CPU profile: four cores, no GPU, the production server's shape. The whole ladder was
    # measured because on CPU the input size buys frame rate and frame rate buys recall.
    (960, 15, 4.0): 'CPU 4 cores: URFD 5/60 falls, 48/79 GMDCSA falls -- the old settings',
    (640, 15, 7.0): 'CPU 4 cores: URFD 25/60 falls, 66/79 GMDCSA falls',
    (480, 15, 6.0): 'CPU 4 cores: URFD 13/60 falls, 68/79 GMDCSA falls',
    (480, 15, 12.0): 'CPU: URFD 38/60 falls, 71/79 GMDCSA falls (not reachable on four cores)',
    (384, 15, 6.0): 'CPU 4 cores: URFD 14/60 falls, 69/79 GMDCSA falls',
    (384, 15, 17.0): 'CPU: URFD 39/60 falls, 69/79 GMDCSA falls (not reachable on four cores)',
    (320, 15, 8.0): 'CPU 3.5 cores: URFD 32/60 falls, 34/40 clean, 64/79 GMDCSA falls (deployed)',
    (320, 15, 9.0): 'CPU 4 cores: URFD 34/60 falls, 32/40 clean, 67/79 GMDCSA falls',
    (256, 15, 10.0): 'CPU 4 cores: URFD 25/60 falls, 62/79 GMDCSA falls',
    (192, 15, 14.0): 'CPU 4 cores: URFD 32/60 falls, 59/79 GMDCSA falls',
}

# What one 1080p camera sustains, measured uncapped (V3_TARGET_FPS=0) by reading the camera
# loop's own periodic log. Pinning at or above this makes the achieved rate depend on machine
# load again, which is what pinning it exists to prevent. The CPU figure is from a container
# limited to 3.5 cores on a fast desktop chip; the production server is a QEMU VM with slower
# cores and will sit below it.
SUSTAINED_FPS = {'gpu': 23.4, 'cpu': 8.1}

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


def compose_profiles():
    """-> {'cpu': (imgsz, fps), 'gpu': (imgsz, fps)} from the two compose files.

    docker-compose.yml IS the CPU deployment; docker-compose.gpu.yml overlays the GPU one on
    top of it, so a value it does not set is inherited. Both are checked, because the server
    this ships to has no GPU and the CPU profile used to be nobody's job.
    """
    def settings(path):
        text = open(os.path.join(ROOT, path), encoding='utf-8').read()
        fps = re.search(r'V3_TARGET_FPS=([0-9.]+)', text)
        imgsz = re.search(r'V3_IMGSZ=([0-9]+)', text)
        return (int(imgsz.group(1)) if imgsz else None,
                float(fps.group(1)) if fps else None)

    cpu = settings('docker-compose.yml')
    gpu = settings('docker-compose.gpu.yml')
    gpu = (gpu[0] if gpu[0] is not None else cpu[0],
           gpu[1] if gpu[1] is not None else cpu[1])
    return {'cpu': cpu, 'gpu': gpu}


def main():
    v3 = load_module('v3', os.path.join(ROOT, 'app', 'detection', 'v3_fall_detection.py'))
    sys.path.insert(0, os.path.join(ROOT, 'training'))
    ds = load_module('ds', os.path.join(ROOT, 'training', 'dataset.py'))
    md = load_module('md', os.path.join(ROOT, 'training', 'model.py'))

    frames, features = onnx_input_frames(os.path.join(ROOT, 'models', 'fall_classifier_v3.onnx'))
    profiles = compose_profiles()

    print(f'deployed ONNX      {frames} frames x {features} features')
    print(f'runtime            WINDOW_SIZE {v3.WINDOW_SIZE}, threshold {v3.THRESHOLD}, '
          f'smoothing {v3.SMOOTH_NEED} of {v3.SMOOTH_OF}')
    print(f'training           WINDOW_SIZE {ds.WINDOW_SIZE}, TEMPORAL_STRIDE {ds.TEMPORAL_STRIDE}, '
          f'{md.FEATURES_PER_FRAME} features')
    for _name in ('cpu', 'gpu'):
        _imgsz, _fps = profiles[_name]
        print(f'{_name} profile        imgsz {_imgsz}, V3_TARGET_FPS {_fps}')
    print()

    check('runtime window matches the deployed ONNX',
          v3.WINDOW_SIZE == frames, f'{v3.WINDOW_SIZE} vs {frames}')
    check('training window matches the deployed ONNX',
          ds.WINDOW_SIZE == frames, f'{ds.WINDOW_SIZE} vs {frames}')
    check('feature count matches the deployed ONNX',
          md.FEATURES_PER_FRAME == features, f'{md.FEATURES_PER_FRAME} vs {features}')
    train_seconds = ds.WINDOW_SIZE * ds.TEMPORAL_STRIDE / SOURCE_FPS
    for name in ('cpu', 'gpu'):
        imgsz, fps = profiles[name]
        check(f'{name}: camera rate is pinned, not left to the hardware', bool(fps),
              '' if fps else f'V3_TARGET_FPS missing from the {name} compose file')
        if not fps:
            continue
        measured = MEASURED.get((imgsz, v3.WINDOW_SIZE, fps))
        check(f'{name}: this exact configuration has been measured', measured is not None,
              measured or f'imgsz {imgsz}, window {v3.WINDOW_SIZE}, {fps:.0f} fps is not in '
              f'MEASURED -- run the sweep before deploying it')
        check(f'{name}: the rate is one the machine sustains', fps <= SUSTAINED_FPS[name],
              f'pinned at {fps:.0f} fps, measured ceiling {SUSTAINED_FPS[name]:.1f} fps '
              f'for one camera')
        # Reported, not asserted: training and live no longer have to cover the same real
        # time, but the ratio is the first thing to look at if accuracy moves for no other
        # reason. On the CPU profile it is deliberately far from 1.
        live_seconds = v3.WINDOW_SIZE / fps
        print(f'      {name}: a window spans {train_seconds:.2f}s in training and {live_seconds:.2f}s '
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
