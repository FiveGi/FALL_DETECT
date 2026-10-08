# -*- coding: utf-8 -*-
"""Plan round 3 D3 (G4): track-level multi-person metric on the 17 frozen multi-person owner segments.

The segment scorer credits ANY alert inside a fall segment. Here each alert is attributed to a person:
the alerting track's hip centroid (normalized 0-1, as the replay feeds it) is compared with the blind,
two-assistant faller marks (training/data/multi_diag_v2/faller_marks_v1.json, positions at t=95%).
An alert is
  far             if it is more than R_MAX from every mark (checked first; never correct),
  correct-person  if strictly nearer the faller mark than every other-person mark (both_fell: any alert within R_MAX),
  wrong-person    if strictly nearer another person's mark,
  tie             if equally near (uncertain; never correct), invalid if the centroid is not finite.
Segments marked exclude_wrong_person (helper overlaps the faller) are AMBIGUOUS: neither correct nor wrong credit
(Codex P2); their any-alert is reported separately. Per segment and phase: correct = >= 1 correct-person alert,
wrong = >= 1 wrong-person alert. Means over phases; denominators = scored (non-excluded) segments.
Only alerts in the SECOND HALF of the segment are attributed (marks are positions at t=95%; an earlier alert can
be a person who left, Codex P2); earlier alerts are counted as 'early' and never credited. Marks are body cells,
the centroid is the hip midpoint. Inputs are validated (frozen 17, distinct phases, one pose profile, arrays).

Usage: V3_THRESHOLD=0.65 python training/measure/track_metric.py <model_dir> <owner_cache_1> [... <owner_cache_8>]
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
spec = importlib.util.spec_from_file_location('v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
R_MAX = 0.25
MARKS = json.load(open('training/data/multi_diag_v2/faller_marks_v1.json'))['segments']
FROZEN17 = ['1.mp4#5', '1.mp4#10', '1.mp4#12', '1.mp4#13', '1.mp4#15', '4.mp4#1', '4.mp4#7', '4.mp4#16', '5.mp4#3',
            '5.mp4#16', '8.mp4#1', '8.mp4#6', '8.mp4#12', '9.mp4#5', '9.mp4#8', '9.mp4#17', '9.mp4#20']


def frames(npz):
    with np.load(npz) as z:
        counts, kpts, t = z['counts'], z['kpts'], z['t']
    if not np.issubdtype(counts.dtype, np.integer) and not np.all(np.mod(counts, 1) == 0):
        raise SystemExit('non-integer counts: %s' % npz)
    if np.any(counts < 0) or len(t) != len(counts) or int(counts.sum()) != len(kpts) or kpts.shape[1:] != (17, 3)             or not np.all(np.isfinite(t)) or np.any(np.diff(t) < 0):
        raise SystemExit('inconsistent cache arrays: %s' % npz)   # Codex P2: fail closed
    out, at = [], 0
    for n in counts:
        out.append(kpts[at:at + int(n)])
        at += int(n)
    return out, t


def alert_centroids(det, npz):
    fr, t = frames(npz)
    half = t[0] + 0.5 * (t[-1] - t[0])
    st = v3.V3MultiPersonFallState()
    cents = []
    for people, ti in zip(fr, t):
        det._replay = [(kp, (kp[v3.LEFT_HIP, :2] + kp[v3.RIGHT_HIP, :2]) / 2.0) for kp in people]
        for track_id, detected, prob, label, centroid in v3.detect_v3_fall_multi(None, st, det, config=None):
            if detected:
                cents.append(np.asarray(centroid, dtype=float)[:2] if ti >= half else None)
    return cents


def classify(c, m):
    if c is None:
        return 'early'
    if not np.all(np.isfinite(c)):
        return 'invalid'
    d_f = np.linalg.norm(c - np.asarray(m['faller_xy']))
    d_o = min([np.linalg.norm(c - np.asarray(o)) for o in m['other_xy']] or [np.inf])
    if min(d_f, d_o) > R_MAX:
        return 'far'
    if m['both_fell'] or d_f < d_o:
        return 'correct'
    return 'wrong' if d_o < d_f else 'tie'


def main(model_dir, caches):
    if len(set(map(os.path.abspath, caches))) != len(caches):
        raise SystemExit('duplicate phase caches')                       # Codex P2: fail closed
    for c in caches:
        if not os.path.exists(os.path.join(c, 'key.json')):
            raise SystemExit('not a pose cache: %s' % c)
    if sorted(MARKS) != sorted(FROZEN17):
        raise SystemExit('marks do not match the frozen 17 segments')
    keys = [json.load(open(os.path.join(c, 'key.json'))) for c in caches]
    if len({k.get('roi_phase') for k in keys}) != len(keys):
        raise SystemExit('phase caches are not distinct phases')
    if len({json.dumps({a: b for a, b in k.items() if a != 'roi_phase'}, sort_keys=True) for k in keys}) != 1:
        raise SystemExit('phase caches differ in pose profile')
    det = v3.V3PoseFallDetector(model_dir=model_dir)
    det._replay = []
    det.extract_all_keypoints = lambda frame: det._replay
    scored = [s for s, m in MARKS.items() if not m['exclude_wrong_person']]
    per_seg = {s: [] for s in MARKS}
    tot = {k: [] for k in ('any_scored', 'correct', 'wrong', 'any_excluded', 'far_alerts', 'tie_alerts', 'invalid_alerts', 'early_alerts')}
    for cache in caches:
        n = dict.fromkeys(tot, 0)
        for seg, m in MARKS.items():
            f = os.path.join(cache, seg + '.npz')
            if not os.path.exists(f):
                raise SystemExit('missing cache file %s' % f)
            kinds = [classify(c, m) for c in alert_centroids(det, f)]
            for k in ('far', 'tie', 'invalid', 'early'):
                n[k + '_alerts'] += kinds.count(k)
            if m['exclude_wrong_person']:
                n['any_excluded'] += bool(kinds)
                per_seg[seg].append('X' if kinds else '-')
                continue
            ok, bad = 'correct' in kinds, 'wrong' in kinds
            n['any_scored'] += bool(kinds); n['correct'] += ok; n['wrong'] += bad
            per_seg[seg].append(('C' if ok else '') + ('W' if bad else '') + ('f' if 'far' in kinds else '')
                                + ('t' if 'tie' in kinds else '') or '-')
        for k in tot:
            tot[k].append(n[k])
    out = {'model_dir': model_dir, 'threshold': v3.THRESHOLD, 'phases': len(caches), 'n_scored': len(scored),
           'n_excluded': len(MARKS) - len(scored)}
    out.update(tot)
    out.update({'mean_' + k: round(float(np.mean(v)), 2) for k, v in tot.items()})
    print(json.dumps(out))
    for seg, v in per_seg.items():
        print('  %-10s %s' % (seg, ' '.join('%-3s' % x for x in v)))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
