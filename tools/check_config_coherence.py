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
    # (input size, window, frame rate, partial-window minimum, preprocessing, threshold)
    #   -> what it scored.
    # The preprocessing belongs in the key for the same reason the input size does: it changes
    # the frames the pose model sees, so a row measured without it does not describe a
    # deployment running with it. Rows predating it are recorded as ('off',).
    #
    # The THRESHOLD belongs here for a blunter reason: without it this check passed any
    # threshold at all. A deployment could run 0.99 -- alerting essentially never -- and be
    # told its configuration had been measured, because the lookup key never asked. That is
    # the failure mode this table exists to prevent, reproduced inside the table itself.
    # The three CPU rows below differ ONLY by threshold, which is precisely why omitting it
    # collapsed them into one row that described whichever had been measured last.
    # GPU profile: RTX 4070 Ti Super, one 1080p camera, uncapped ceiling 23.4 fps.
    (960, 15, 15.0, 0, ('off',), 0.65, 0, 0): 'GPU: URFD 41/60 falls, 34/40 clean; held-out clean 41/56',
    (960, 15, 18.0, 0, ('off',), 0.65, 0, 0): 'GPU: URFD 45/60 falls, 32/40 clean',
    (960, 15, 20.0, 0, ('off',), 0.65, 0, 0): 'GPU: URFD 45/60 falls, 33/40 clean; held-out clean 41/56',
    (960, 15, 23.0, 0, ('off',), 0.65, 0, 0): 'GPU: URFD 48/60 falls, 33/40 clean',
    (960, 15, 20.0, 4, ('off',), 0.65, 0, 0): 'GPU: URFD 56/60 falls, 33/40 clean; held-out clean 40/56',
    # CPU profile: four cores, no GPU, the production server's shape.
    (960, 15, 4.0, 0, ('off',), 0.65, 0, 0): 'CPU 4 cores: URFD 5/60 falls, 48/79 GMDCSA falls -- the old settings',
    (640, 15, 7.0, 0, ('off',), 0.65, 0, 0): 'CPU 4 cores: URFD 25/60 falls, 66/79 GMDCSA falls',
    (480, 15, 6.0, 0, ('off',), 0.65, 0, 0): 'CPU 4 cores: URFD 13/60 falls, 68/79 GMDCSA falls',
    (480, 15, 12.0, 0, ('off',), 0.65, 0, 0): 'CPU: URFD 38/60 falls, 71/79 GMDCSA falls (not reachable on four cores)',
    (384, 15, 6.0, 0, ('off',), 0.65, 0, 0): 'CPU 4 cores: URFD 14/60 falls, 69/79 GMDCSA falls',
    (384, 15, 17.0, 0, ('off',), 0.65, 0, 0): 'CPU: URFD 39/60 falls, 69/79 GMDCSA falls (not reachable on four cores)',
    (320, 15, 6.0, 0, ('off',), 0.65, 0, 0): 'CPU 4 cores: URFD 13/60 falls -- below the cliff, do not deploy',
    (320, 15, 8.0, 0, ('off',), 0.65, 0, 0): 'CPU 3.5 cores: URFD 32/60 falls, 34/40 clean, 64/79 GMDCSA falls',
    (320, 15, 9.0, 0, ('off',), 0.65, 0, 0): 'CPU 4 cores: URFD 34/60 falls, 32/40 clean, 67/79 GMDCSA falls',
    (256, 15, 10.0, 0, ('off',), 0.65, 0, 0): 'CPU 4 cores: URFD 25/60 falls, 62/79 GMDCSA falls',
    (192, 15, 14.0, 0, ('off',), 0.65, 0, 0): 'CPU 4 cores: URFD 32/60 falls, 59/79 GMDCSA falls',
    (320, 15, 8.0, 2, ('off',), 0.65, 0, 0): 'CPU 3.5 cores: URFD 41/60 falls, held-out clean 40/56',
    (320, 15, 8.0, 3, ('off',), 0.65, 0, 0): 'CPU 3.5 cores: URFD 44/60 falls, held-out clean 40/56',
    (320, 15, 8.0, 4, ('off',), 0.65, 0, 0): 'CPU 3.5 cores: URFD 45/60 falls, held-out clean 41/56',
    # Deployed. Frame preprocessing costs no falls on either profile and removes false alarms:
    # three between the two profiles, on the darkest clips in the corpus.
    # Deployed. The darkness gate does not appear in this key because gates of 32, 50 and 70
    # measured bit-identical on every surface at full light, on both profiles -- so one row
    # describes all of them here. Where they differ is dim footage, which this table does not
    # cover at all (see docs/next_steps.md item 14).
    (320, 15, 8.0, 4, ('auto',), 0.65, 0, 0):
        'CPU 3.5 cores: URFD 45/60 falls, held-out clean 43/56 (deployed, preprocessing on)',
    # Person-crop at 256 (V3_ROI_IMGSZ). The crop setting is part of the key for the reason the
    # threshold is: without it, a cropped deployment matched the full-frame row above and was
    # told it had been measured. Cache replay deff31100752 vs 58bd55e55f1d, same classifier.
    (320, 15, 8.0, 4, ('auto',), 0.65, 256, 8):
        'CPU: URFD 46/60 falls, held-out clean 42/56; owner clips 34/75 vs 28/75 (crop 256)',
    # The two threshold variants app/services/detector_profiles.py offers an operator. Replayed
    # from the cached pose stream of the row above -- same clips, same pose pass, classifier
    # re-run at each threshold -- so the three rows are a controlled comparison rather than
    # three separate measurements. Re-measured when threshold entered this key, which is how
    # the 'catch more' row was found to have been carrying 38/56 when it scores 40/56.
    (320, 15, 8.0, 4, ('auto',), 0.70, 0, 0):
        'CPU 3.5 cores: URFD 44/60 falls, held-out clean 44/56 (fewer false alarms)',
    (320, 15, 8.0, 4, ('auto',), 0.50, 0, 0):
        'CPU 3.5 cores: URFD 49/60 falls, held-out clean 40/56 (catch more)',
    (960, 15, 20.0, 4, ('auto',), 0.65, 0, 0):
        'GPU: URFD 56/60 falls, 33/40 clean; held-out clean 41/56 (deployed, preprocessing on)',
    # GPU threshold variants, so an operator picking a threshold is not told the machine is
    # unmeasured on whichever profile they are not running. CACHE REPLAYS, not fresh pose
    # passes: the same cached keypoints with the classifier re-run, which is a controlled
    # comparison between thresholds but is not identical to a sweep. Replaying the row above
    # this way returns 34/40 where the sweep recorded 33/40 -- one clip, the known disagreement
    # between a cached stream and a fresh pose pass, and the reason these rows say which they
    # are instead of being quietly merged with the sweep rows.
    (960, 15, 20.0, 4, ('auto',), 0.70, 0, 0):
        'GPU: URFD 54/60 falls, held-out clean 41/56 (cache replay, fewer false alarms)',
    (960, 15, 20.0, 4, ('auto',), 0.50, 0, 0):
        'GPU: URFD 58/60 falls, held-out clean 37/56 (cache replay, catch more)',
    (320, 15, 8.0, 6, ('off',), 0.65, 0, 0): 'CPU 3.5 cores: URFD 38/60 falls, held-out clean 42/56',
    (320, 15, 8.0, 8, ('off',), 0.65, 0, 0): 'CPU 3.5 cores: URFD 31/60 falls, held-out clean 42/56',
}

def measured_key(imgsz, window, fps, partial_min, preprocess, threshold, roi_imgsz=0,
                 roi_full_every=0):
    """The ONE way to build a MEASURED lookup key. The check below and the web page's
    /api/detector both call this; the page once built its own five-field key and, after the
    threshold and the crop joined the table's key, silently reported every configuration --
    the deployed one included -- as unmeasured (Codex REVIEW-2).

    The crop cadence is in the key because a crop measured with a full-frame pass every 8 frames
    says nothing about cadence 1, which never crops at all."""
    roi = int(roi_imgsz or 0)
    # The cadence the RUNTIME uses, not the number written down: v3_fall_detection does
    # `% max(ROI_FULL_EVERY, 1)`, so 0 means a full pass every frame. Normalising here means an
    # explicit 0 can never borrow the every-8 row's measurement (Codex REVIEW-3).
    every = max(int(roi_full_every), 1) if (roi and roi_full_every is not None) else (8 if roi else 0)
    return (int(imgsz) if imgsz is not None else None, window,
            float(fps) if fps is not None else None, partial_min,
            tuple(preprocess or ()) or ('off',),
            round(float(threshold), 4) if threshold is not None else None,
            roi, every)


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
    """-> {'cpu': settings, 'gpu': settings}: the worker's EFFECTIVE environment per profile.

    docker-compose.yml IS the CPU deployment; docker-compose.gpu.yml overlays the GPU one, and
    Compose merges an overlay's `environment:` over the base's -- a value the overlay does not
    set is inherited. This used to regex each file separately for two keys, and then the
    measured-config lookup took the person-crop setting from THIS script's own environment, so
    a shell variable could change the verdict and the GPU profile silently inherited the CPU
    crop (Codex review). Now the worker service is read as YAML and merged the way Compose does.
    """
    import yaml

    def read_env_file(path):
        out = {}
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            return out
        for line in open(full, encoding='utf-8'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, _, v = line.partition('=')
                out[k.strip()] = v.strip().strip('"').strip("'")
        return out

    def worker(path):
        doc = yaml.safe_load(open(os.path.join(ROOT, path), encoding='utf-8')) or {}
        svc = (doc.get('services') or {}).get('celery_worker') or {}
        env = svc.get('environment') or {}
        if isinstance(env, list):
            env = dict(item.split('=', 1) if '=' in item else (item, '') for item in env)
        files = svc.get('env_file') or []
        return ([files] if isinstance(files, str) else list(files),
                {str(k): str(v) for k, v in env.items()})

    # Compose precedence, lowest first: env_file entries, then `environment:`; an overlay's
    # entries replace the base's. env_file values DO reach the container when `environment:`
    # does not set the key, so ignoring them let a crop set in .env run while this reported the
    # full-frame row as measured (Codex REVIEW-3).
    base_files, base_env = worker('docker-compose.yml')
    gpu_files, gpu_env = worker('docker-compose.gpu.yml')
    cpu = {}
    for f in base_files:
        cpu.update(read_env_file(f))
    cpu.update(base_env)
    # Compose CONCATENATES an overlay's env_file list onto the base's (Codex REVIEW-3b,
    # confirmed against `docker compose config`); replacing it let a base .env crop vanish from
    # the GPU profile's view.
    gpu = {}
    for f in base_files + [f for f in gpu_files if f not in base_files]:
        gpu.update(read_env_file(f))
    gpu.update(base_env)
    gpu.update(gpu_env)

    def settings(env):
        def num(key, cast):
            v = env.get(key)
            return cast(v) if v not in (None, '') else None
        return {'imgsz': num('V3_IMGSZ', int), 'fps': num('V3_TARGET_FPS', float),
                'roi': num('V3_ROI_IMGSZ', int) or 0,
                'roi_every': num('V3_ROI_FULL_EVERY', int)}   # None = unset; measured_key decides

    return {'cpu': settings(cpu), 'gpu': settings(gpu)}


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
        _p = profiles[_name]
        print(f'{_name} profile        imgsz {_p["imgsz"]}, V3_TARGET_FPS {_p["fps"]}, '
              f'crop {_p["roi"]} every {_p["roi_every"] if _p["roi"] else "-"}')
    print()

    check('runtime window matches the deployed ONNX',
          v3.WINDOW_SIZE == frames, f'{v3.WINDOW_SIZE} vs {frames}')
    check('training window matches the deployed ONNX',
          ds.WINDOW_SIZE == frames, f'{ds.WINDOW_SIZE} vs {frames}')
    check('training feature count matches the deployed ONNX',
          md.FEATURES_PER_FRAME == features, f'{md.FEATURES_PER_FRAME} vs {features}')
    # Three places have to agree on how wide a frame's feature vector is, and two of them are
    # set by environment flags -- USE_FRAME_POSITION for training, V3_FRAME_POSITION for the
    # runtime. Setting one and forgetting the other produces a model that loads and scores
    # nonsense, so the third place, the deployed file itself, is the arbiter for both.
    check('runtime feature count matches the deployed ONNX',
          v3.FEATURES_PER_FRAME == features, f'{v3.FEATURES_PER_FRAME} vs {features}')
    train_seconds = ds.WINDOW_SIZE * ds.TEMPORAL_STRIDE / SOURCE_FPS
    for name in ('cpu', 'gpu'):
        imgsz, fps = profiles[name]['imgsz'], profiles[name]['fps']
        roi, roi_every = profiles[name]['roi'], profiles[name]['roi_every']
        check(f'{name}: camera rate is pinned, not left to the hardware', bool(fps),
              '' if fps else f'V3_TARGET_FPS missing from the {name} compose file')
        if not fps:
            continue
        preprocess = tuple(v3.PREPROCESS) or ('off',)
        measured = MEASURED.get(measured_key(imgsz, v3.WINDOW_SIZE, fps, v3.PARTIAL_MIN,
                                             preprocess, v3.THRESHOLD, roi, roi_every))
        check(f'{name}: this exact configuration has been measured', measured is not None,
              measured or f'imgsz {imgsz}, window {v3.WINDOW_SIZE}, {fps:.0f} fps, partial '
              f'from {v3.PARTIAL_MIN}, preprocess {"+".join(preprocess)}, threshold '
              f'{v3.THRESHOLD:g}, crop {roi} every {roi_every} is not in MEASURED -- run the sweep before deploying it')
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
