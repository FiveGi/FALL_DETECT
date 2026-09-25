# -*- coding: utf-8 -*-
"""Run the pose pass once per clip and cache its output, so a classifier A/B is cheap.

The pose pass is essentially the whole cost of this pipeline -- tens of milliseconds a frame
against 0.2 ms for the classifier. Comparing two classifiers with `rule_sweep_perclip.py`
therefore pays for the expensive half twice and measures it twice, which is both slow (hours
per model over the 220-clip set) and noisy: two runs of the same pose model over the same
video do not have to agree to the last keypoint, so a difference of one or two clips between
two classifiers can be the pose pass rather than the classifiers.

This script writes the keypoints out once. `replay_classifiers.py` then feeds the identical
stream to every classifier under test, so the only thing that differs between two results is
the thing being compared. A full sweep drops from hours to under a minute per model.

What it does NOT replace: `rule_sweep_perclip.py` stays the measurement of record for anything
that changes the pose pass itself -- input size, pose checkpoint, confidence, frame rate. A
cache is keyed on all of those, and asking for one that was never cached is an error rather
than a silently wrong answer.

Usage:
    V3_DEVICE=cuda V3_IMGSZ=320 TARGET_FPS=8 CACHE_DIR=... python cache_pose_streams.py
"""
import glob
import hashlib
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'training'))
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
from eval_v3_frame_drop import TRAIN_ADL  # noqa: E402

FPS = float(os.environ.get('TARGET_FPS', 8))
# SIMULATE_DARK: scale every frame's brightness before the detector sees it, to ask a question
# the corpus cannot answer on its own -- how does this behave in a room at night?
#
# Not one clip here is dark except twelve URFD ADL clips and one out-of-domain Test clip, and
# none of the fall clips are. But the deployment is a bedroom at night. Darkening footage whose
# correct answer is already known is the only way to measure the thing that matters without new
# recordings: how much accuracy is lost as the light goes, and how much of it the preprocessing
# gets back.
#
# A scale, plus shot noise that grows as the signal falls. A real sensor in low light does not
# simply output a dimmer picture -- it outputs a dimmer, noisier one, and the noise is what
# actually breaks pose estimation. Leaving it out would make this test too kind.
#
# This lives in the measurement, never in the detector: it is a way of asking a question, not
# something any deployment should do to its own frames.
SIMULATE_DARK = float(os.environ.get('SIMULATE_DARK', 1.0))
DARK_NOISE = float(os.environ.get('SIMULATE_DARK_NOISE', 6.0))


def darken(frame):
    """Scale brightness to SIMULATE_DARK and add sensor noise proportional to the loss."""
    if SIMULATE_DARK >= 0.999:
        return frame
    out = frame.astype(np.float32) * SIMULATE_DARK
    if DARK_NOISE > 0:
        sigma = DARK_NOISE * (1.0 - SIMULATE_DARK)
        out += np.random.normal(0.0, sigma, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
CACHE_DIR = os.environ.get('CACHE_DIR', os.path.join(ROOT, 'training', 'data', 'pose_cache'))


def clip_groups():
    """The same five groups, in the same order, as rule_sweep_perclip.py."""
    return [
        ('urfd_fall', sorted(glob.glob('training/data/urfd/fall-*.mp4')), True),
        ('urfd_adl', sorted(glob.glob('training/data/urfd/adl-*.mp4')), True),
        ('gmdcsa_fall', sorted(glob.glob('training/data/gmdcsa24_fall_raw/*.mp4')), False),
        ('val_adl', sorted(glob.glob('training/data/gmdcsa24_adl_raw_val/*.mp4')), False),
        ('train50_adl', [q for q in (os.path.join('training/data/gmdcsa24_adl_raw_train50', n + '.mp4')
                                     for n in TRAIN_ADL) if os.path.exists(q)], False),
    ]


def cache_key():
    """Everything the cached keypoints depend on.

    A cache that silently answers for the wrong input size would be worse than no cache at
    all, so the key is the configuration itself and `replay_classifiers.py` re-derives it
    rather than trusting a directory name.
    """
    return {
        'pose_model': os.environ.get('V3_POSE_MODEL', 'yolo26s-pose.pt'),
        'imgsz': v3.IMGSZ,
        'pose_conf': v3.POSE_CONF,
        'tracker': v3.TRACKER,
        'num_poses': v3.NUM_POSES,
        'fps': FPS,
        # V3_PREPROCESS alters the frame before the pose model sees it, so it changes the
        # keypoints as surely as the input size does. Without it in the key, a replay would
        # answer for a cache built with a different setting and never say so.
        'preprocess': list(v3.PREPROCESS),
        'dark_below': v3.PREPROCESS_DARK_BELOW,
        # A darkened run is a different question, not the same one measured twice.
        'simulate_dark': SIMULATE_DARK,
        'dark_noise': DARK_NOISE if SIMULATE_DARK < 0.999 else 0.0,
    }


def cache_dir_for(key):
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, digest)


def sampled_frames(path, fps, rgb_half):
    """Yields the frames the live loop would classify, at `fps`, matching rule_sweep_perclip."""
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, last_slot = 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        slot = int(i * fps / src)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        yield darken(frame[:, frame.shape[1] // 2:] if rgb_half else frame)
    cap.release()


def main():
    # Same seed every run, so "darkened to 40%" is the same footage each time and two settings
    # can be compared on it rather than on two different draws of noise.
    np.random.seed(int(os.environ.get('SIMULATE_DARK_SEED', 0)))
    key = cache_key()
    out_dir = cache_dir_for(key)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'key.json'), 'w') as fh:
        json.dump(key, fh, indent=1)
    print('cache', out_dir, key, flush=True)

    if v3.TRACKER != 'hip':
        raise SystemExit('only the hip tracker is cacheable here: bytetrack assigns identities '
                         'inside the pose call, so its output is not a pure function of the frame')

    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    for group, paths, rgb_half in clip_groups():
        n_new = 0
        for path in paths:
            name = os.path.basename(path)
            npz = os.path.join(out_dir, '%s__%s.npz' % (group, name))
            if os.path.exists(npz):
                continue
            counts, kpts = [], []
            for frame in sampled_frames(path, FPS, rgb_half):
                people = det.extract_all_keypoints(frame)
                counts.append(len(people))
                for kp, _hip in people:
                    kpts.append(kp)
            # Ragged by nature -- a frame holds zero to NUM_POSES people -- so it is stored
            # flat with a per-frame count, not as an object array.
            np.savez_compressed(
                npz,
                counts=np.array(counts, dtype=np.int16),
                kpts=(np.stack(kpts) if kpts else np.zeros((0, v3.NUM_KEYPOINTS, 3), np.float32)))
            n_new += 1
        print('  %-14s %d clips (%d cached now)' % (group, len(paths), n_new), flush=True)
    print('done', out_dir)


if __name__ == '__main__':
    main()
