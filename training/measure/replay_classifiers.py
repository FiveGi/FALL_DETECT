# -*- coding: utf-8 -*-
"""Replay a cached pose stream through the detector, so only the classifier differs.

Reads what `cache_pose_streams.py` wrote and runs the real pipeline over it: the same hip
tracker, the same per-person windows, the same partial-window padding, the same N-of-M
smoothing and the same "did this clip ever alert" rule as `rule_sweep_perclip.py`. The one
substitution is `extract_all_keypoints`, which returns the cached keypoints for the frame
instead of running the pose model.

Output is the same JSON shape rule_sweep_perclip.py writes, so every downstream analysis --
the URFD half-split, the workbook generator -- reads it unchanged.

It refuses to run against a cache built at a different input size, frame rate or pose model,
because those change the keypoints and a replay cannot recover them.

Usage:
    TEST_MODEL_DIR=... CACHE_DIR=... TARGET_FPS=8 V3_IMGSZ=320 OUT=... python replay_classifiers.py
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
import cache_pose_streams as cache  # noqa: E402

MODEL_DIR = os.environ.get('TEST_MODEL_DIR', os.path.join(ROOT, 'models'))
# Read in main(), not at import: load_stream() is useful to other measurement scripts and a
# module that cannot be imported without an environment variable set is a trap for them.


def load_stream(npz_path):
    """-> list of (n_people, (n, 17, 3)) per frame, in frame order."""
    with np.load(npz_path) as data:
        counts, kpts = data['counts'], data['kpts']
    frames, at = [], 0
    for n in counts:
        frames.append(kpts[at:at + n])
        at += int(n)
    return frames


def alerted(det, frames):
    """The identical alerting rule rule_sweep_perclip.py applies: did the clip ever alert."""
    st = v3.V3MultiPersonFallState()
    hits = 0
    last = None
    for people in frames:
        det._replay = [(kp, (kp[v3.LEFT_HIP, :2] + kp[v3.RIGHT_HIP, :2]) / 2.0) for kp in people]
        hit = any(r[1] for r in v3.detect_v3_fall_multi(None, st, det, config=None))
        label = 'fall' if hit else 'no_fall'
        if label != last:
            hits += 1 if hit else 0
            last = label
    return hits > 0


def main():
    out = os.environ['OUT']
    key = cache.cache_key()
    cache_dir = cache.cache_dir_for(key)
    key_file = os.path.join(cache_dir, 'key.json')
    if not os.path.exists(key_file):
        raise SystemExit(
            'no cached pose stream for this configuration:\n  %s\nrun cache_pose_streams.py '
            'with the same V3_IMGSZ / TARGET_FPS / V3_POSE_MODEL first.' % json.dumps(key))

    det = v3.V3PoseFallDetector(model_dir=MODEL_DIR)
    # The pose model is loaded and then never called -- the constructor needs it, and paying
    # a couple of seconds is worth keeping the detector object exactly as production builds it.
    det._replay = []
    det.extract_all_keypoints = lambda frame: det._replay

    meta = {'model': MODEL_DIR, 'window': v3.WINDOW_SIZE, 'need': v3.SMOOTH_NEED,
            'of': v3.SMOOTH_OF, 'threshold': v3.THRESHOLD, 'fps': key['fps'],
            'partial_min': v3.PARTIAL_MIN, 'replay_of': key}

    rows = []
    if os.path.exists(out):
        prev = json.load(open(out))
        if prev.get('meta') == meta:
            rows = prev['rows']
            print('resuming: %d clips already measured' % len(rows), flush=True)
        else:
            raise SystemExit('%s holds a different configuration' % out)
    done = {(r[0], r[1]) for r in rows}

    for group, paths, _rgb_half in cache.clip_groups():
        todo = 0
        for path in paths:
            name = os.path.basename(path)
            if (group, name) in done:
                continue
            npz = os.path.join(cache_dir, '%s__%s.npz' % (group, name))
            if not os.path.exists(npz):
                raise SystemExit('clip missing from the cache: %s' % npz)
            rows.append([group, name, alerted(det, load_stream(npz))])
            todo += 1
        json.dump({'meta': meta, 'rows': rows}, open(out, 'w'), indent=1)
        print('  %-14s %d clips (%d replayed now)' % (group, len(paths), todo), flush=True)

    json.dump({'meta': meta, 'rows': rows}, open(out, 'w'), indent=1)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
