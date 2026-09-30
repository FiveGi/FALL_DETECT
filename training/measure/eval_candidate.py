# -*- coding: utf-8 -*-
"""Judge one or more candidate classifiers by day and by (simulated) night, fairly.

Fairness here means three things this project learned the hard way:
  1. The threshold is re-chosen PER MODEL. Two models trained the same way land at different
     points on the score axis, so comparing them at one fixed threshold compares operating
     points, not detectors.
  2. It is chosen on URFD half A and reported on half B (training/urfd_split.py, pairs of
     sequences, never odd/even). A number chosen and reported on the same clips flatters.
  3. Night is reported over every IR cache given -- several noise seeds -- as a range, because
     one noise draw moved this measurement by 8 points.

Replays cached pose streams, so the pose model is identical for every candidate: only the
classifier differs. Each candidate is a directory holding a `fall_classifier_v3.onnx`.

Usage:
    python training/measure/eval_candidate.py --day <cache> --ir <cache> [<cache> ...] \
        --model name=<dir> [--model name=<dir> ...]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('V3_POSE_MODEL', os.path.join(ROOT, 'models', 'yolo26s-pose.pt'))
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
from replay_classifiers import load_stream  # noqa: E402
from urfd_split import in_half_a, in_half_b  # noqa: E402

THRESHOLDS = [round(x, 2) for x in np.arange(0.35, 0.86, 0.05)]


def alerted(det, frames, threshold):
    st = v3.V3MultiPersonFallState()
    for people in frames:
        det._replay = [(kp, (kp[v3.LEFT_HIP, :2] + kp[v3.RIGHT_HIP, :2]) / 2.0) for kp in people]
        if any(r[1] for r in v3.detect_v3_fall_multi(None, st, det, config=None,
                                                     threshold=threshold)):
            return True
    return False


def load_group(cache, group):
    out = {}
    for f in sorted(os.listdir(cache)):
        if f.startswith(group + '__') and f.endswith('.npz'):
            out[f.split('__', 1)[1].replace('.npz', '')] = load_stream(os.path.join(cache, f))
    return out


def score(det, clips, threshold):
    return {name: alerted(det, frames, threshold) for name, frames in clips.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True)
    ap.add_argument('--ir', nargs='*', default=[])
    ap.add_argument('--model', action='append', required=True)
    ap.add_argument('--curve', action='store_true',
                    help='print every threshold instead of choosing one: compare models at a '
                         'MATCHED day false-alarm count, not at whatever each one chose')
    a = ap.parse_args()

    day = {g: load_group(a.day, g) for g in ('urfd_fall', 'urfd_adl', 'val_adl')}
    night = [{g: load_group(c, g) for g in ('urfd_fall', 'urfd_adl')} for c in a.ir]

    for spec_ in a.model:
        name, mdir = spec_.split('=', 1)
        det = v3.V3PoseFallDetector(model_dir=mdir)
        det.extract_all_keypoints = lambda frame, d=det: d._replay

        if a.curve:
            # No threshold is chosen here, so using every clip introduces no selection bias.
            print('=== %s  (all clips; day false alarms = URFD ADL 40 + val ADL 16)' % name)
            print('  thr   day falls/60  day false alarms/56   night falls/60 per seed   '
                  'night false alarms/40 per seed')
            for thr in THRESHOLDS:
                df = sum(score(det, day['urfd_fall'], thr).values())
                dfa = (sum(score(det, day['urfd_adl'], thr).values())
                       + sum(score(det, day['val_adl'], thr).values()))
                nf = [sum(score(det, n['urfd_fall'], thr).values()) for n in night]
                nfa = [sum(score(det, n['urfd_adl'], thr).values()) for n in night]
                print('  %.2f  %6d        %8d              %-22s  %s' % (
                    thr, df, dfa, ' '.join(map(str, nf)), ' '.join(map(str, nfa))))
                sys.stdout.flush()
            continue

        # Choose on half A, by day: falls caught + ADL clean, ties to the lower threshold.
        best = None
        for thr in THRESHOLDS:
            f = score(det, {k: v for k, v in day['urfd_fall'].items() if in_half_a(k)}, thr)
            ad = score(det, {k: v for k, v in day['urfd_adl'].items() if in_half_a(k)}, thr)
            bal = sum(f.values()) / max(len(f), 1) + (len(ad) - sum(ad.values())) / max(len(ad), 1)
            if best is None or bal > best[0] + 1e-9:
                best = (bal, thr)
        thr = best[1]

        def falls(clips, sel):
            s = score(det, {k: v for k, v in clips.items() if sel(k)}, thr)
            return sum(s.values()), len(s)

        def clean(clips, sel=lambda k: True):
            s = score(det, {k: v for k, v in clips.items() if sel(k)}, thr)
            return len(s) - sum(s.values()), len(s)

        db = falls(day['urfd_fall'], in_half_b)
        dall = falls(day['urfd_fall'], lambda k: True)
        cb = clean(day['urfd_adl'], in_half_b)
        cv = clean(day['val_adl'])
        nb = [falls(n['urfd_fall'], in_half_b) for n in night]
        nall = [falls(n['urfd_fall'], lambda k: True) for n in night]
        ncl = [clean(n['urfd_adl']) for n in night]

        print('=== %s   threshold %.2f (chosen on URFD half A, by day)' % (name, thr))
        print('  DAY    URFD falls half B %2d/%-2d   all %2d/%-2d   URFD ADL clean half B %2d/%-2d'
              '   val ADL clean %2d/%-2d' % (db + dall + cb + cv))
        if night:
            print('  NIGHT  URFD falls half B %s   all %s   URFD ADL clean %s' % (
                ' '.join('%d/%d' % x for x in nb), ' '.join('%d/%d' % x for x in nall),
                ' '.join('%d/%d' % x for x in ncl)))
        sys.stdout.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main())
