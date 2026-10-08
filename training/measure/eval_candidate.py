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
import json
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
# THRESHOLD_GRID=extended adds 0.90/0.95/0.975/0.99, pre-registered 2026-10-01 for every Stage 1
# model incl. the deployed baseline (Codex), after the 0.35-0.85 grid's ceiling was reached.
if os.environ.get('THRESHOLD_GRID') == 'extended':
    THRESHOLDS = THRESHOLDS + [0.9, 0.95, 0.975, 0.99]


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


def pooled(det, day_caches, ir_caches, fixed=None):
    """Choose ONE threshold on the union of URFD half-A clips over several caches (the crop
    schedule's phases), then report every cache separately. Frozen in AI_HANDOFF.md before any
    pooled result was seen (Codex + Gemini: taking the mode of per-phase choices after seeing
    them adds selection bias; phases are correlated, so each clip-phase is weighted equally)."""
    days = [{g: load_group(c, g) for g in ('urfd_fall', 'urfd_adl', 'val_adl')} for c in day_caches]
    best = None
    for thr in THRESHOLDS:
        f = [a for d in days for k, v in d['urfd_fall'].items() if in_half_a(k)
             for a in [alerted(det, v, thr)]]
        ad = [a for d in days for k, v in d['urfd_adl'].items() if in_half_a(k)
              for a in [alerted(det, v, thr)]]
        bal = sum(f) / max(len(f), 1) + (len(ad) - sum(ad)) / max(len(ad), 1)
        if best is None or bal > best[0] + 1e-9:
            best = (bal, thr)
    thr = best[1]
    print('threshold %.2f chosen on pooled URFD half A of %d caches' % (thr, len(days)))
    if fixed is not None:
        # Report-only: the deployed operating point, for the gates. Chooses nothing.
        thr = fixed
        print('REPORTING AT FIXED threshold %.2f (--fixed-threshold; not a choice)' % thr)
    rows = []
    # PER_CLIP_OUT=<file>: also save every reported clip's outcome (alerted True/False) per cache, so two runs can be
    # compared clip by clip -- equal totals can hide a fall lost here and another gained there (Codex, 8 Oct E1).
    per_clip = {}
    for c, d in zip(day_caches, days):
        oc = {g: {k: bool(alerted(det, v, thr)) for k, v in d[g].items()
                  if g == 'val_adl' or in_half_b(k)} for g in ('urfd_fall', 'urfd_adl', 'val_adl')}
        per_clip[os.path.basename(c)] = oc
        fb = sum(oc['urfd_fall'].values())
        nb = len(oc['urfd_fall'])
        cb = sum(not a for a in oc['urfd_adl'].values())
        na = len(oc['urfd_adl'])
        cv = sum(not a for a in oc['val_adl'].values())
        rows.append((fb, cb, cv))
        print('  %s  half B falls %d/%d  ADL clean %d/%d  val ADL clean %d/%d'
              % (os.path.basename(c), fb, nb, cb, na, cv, len(d['val_adl'])))
        sys.stdout.flush()
    for i, what in enumerate(('half B falls', 'half B ADL clean', 'val ADL clean')):
        v = [r[i] for r in rows]
        print('  mean %-17s %.1f  range %d-%d' % (what, sum(v) / len(v), min(v), max(v)))
    for c in ir_caches:
        n = {g: load_group(c, g) for g in ('urfd_fall', 'urfd_adl')}
        oc = {g: {k: bool(alerted(det, v, thr)) for k, v in n[g].items()} for g in ('urfd_fall', 'urfd_adl')}
        per_clip['NIGHT ' + os.path.basename(c)] = oc
        print('  NIGHT %s  falls %d/%d  ADL false alarms %d/%d' % (
            os.path.basename(c), sum(oc['urfd_fall'].values()), len(oc['urfd_fall']),
            sum(oc['urfd_adl'].values()), len(oc['urfd_adl'])))
    if os.environ.get('PER_CLIP_OUT'):
        if os.path.exists(os.environ['PER_CLIP_OUT']):   # never overwrite the other arm's evidence (Codex, 8 Oct)
            raise SystemExit('PER_CLIP_OUT exists, refusing to overwrite: %s' % os.environ['PER_CLIP_OUT'])
        json.dump({'threshold': thr, 'caches': per_clip}, open(os.environ['PER_CLIP_OUT'], 'w'), indent=1)
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', nargs='+', metavar='CACHE',
                    help='choose one threshold on pooled half A of these caches; with --ir, also '
                         'report night falls/false alarms at it. Replaces --day.')
    ap.add_argument('--fixed-threshold', type=float,
                    help='with --pool: report at this threshold instead of the chosen one '
                         '(the deployed operating point, for comparison only)')
    ap.add_argument('--day')
    ap.add_argument('--ir', nargs='*', default=[])
    ap.add_argument('--model', action='append', required=True)
    ap.add_argument('--curve', action='store_true',
                    help='print every threshold instead of choosing one: compare models at a '
                         'MATCHED day false-alarm count, not at whatever each one chose')
    a = ap.parse_args()
    if a.pool:
        for spec_ in a.model:
            name, mdir = spec_.split('=', 1)
            det = v3.V3PoseFallDetector(model_dir=mdir)
            det.extract_all_keypoints = lambda frame, d=det: d._replay
            print('=== %s' % name)
            pooled(det, a.pool, a.ir, a.fixed_threshold)
        return 0
    if not a.day:
        ap.error('--day or --pool is required')

    day ={g: load_group(a.day, g) for g in ('urfd_fall', 'urfd_adl', 'val_adl')}
    night = [{g: load_group(c, g) for g in ('urfd_fall', 'urfd_adl')} for c in a.ir]

    for spec_ in a.model:
        name, mdir = spec_.split('=', 1)
        det = v3.V3PoseFallDetector(model_dir=mdir)
        det.extract_all_keypoints = lambda frame, d=det: d._replay

        if a.curve:
            # No threshold is chosen here, so using every clip introduces no selection bias.
            print('=== %s  (all clips; day false alarms = URFD ADL 40 + val ADL 16)' % name)
            print('  thr   day falls/60  day false alarms/56 (URFD ADL /40 + val ADL /16)   night falls/60 per seed   '
                  'night false alarms/40 per seed')
            for thr in THRESHOLDS:
                df = sum(score(det, day['urfd_fall'], thr).values())
                ufa = sum(score(det, day['urfd_adl'], thr).values())
                vfa = sum(score(det, day['val_adl'], thr).values())
                dfa = '%d (%d+%d)' % (ufa + vfa, ufa, vfa)
                nf = [sum(score(det, n['urfd_fall'], thr).values()) for n in night]
                nfa = [sum(score(det, n['urfd_adl'], thr).values()) for n in night]
                print('  %.3f %6d        %-14s                    %-22s  %s' % (
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
