# -*- coding: utf-8 -*-
"""Overnight plan E1 (8 Oct, agreed by Claude + Codex + Gemini): P3 held-track rule, paired OFF vs ON, with a
person-specific gap/event ledger on the 17 frozen multi-person owner segments x 8 phases = 136 segment-phases.

Reused-data DIAGNOSTICS for option A (exploration; nothing here ships before 9 Oct). Both arms run in ONE process on
the SAME cache frames, so the only difference is v3.HELD_ALERT_MAX (0 = today's behaviour, K = the rule).

Per segment-phase it records, against the blind two-assistant faller marks (faller_marks_v1.json, t = 95 %):
  faller coverage   share of SECOND-HALF frames with a detection within R_MAX of the faller mark
  longest gap       longest run of second-half frames with no such detection (frames and seconds)
  faller ids        distinct SEEN track ids within R_MAX of the faller mark in the second half (id switches = ids - 1)
  per arm           first correct-person alert time (s from segment start), alert kinds (correct / wrong / far / tie /
                    early), and whether the segment-phase counts as correct / wrong (track_metric.py's rule)
and the paired verdict per segment-phase: correct kept / lost / gained, wrong removed / added.
These are SPATIAL PROXIES (positions vs one mark at t=95 %), not validated identities; segments marked
exclude_wrong_person are recorded but left out of every summary count.
Codex's question it answers: when the faller is missed, was the faller undetected (coverage, gap) or detected but
under another id (faller ids)? i.e. detection loss vs association failure.

Usage: V3_THRESHOLD=0.70 python training/measure/p3_ledger.py K MODEL_DIR OWNER_CACHE_1 [... OWNER_CACHE_8] > out.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import track_metric as tm   # noqa: E402  (loads v3, MARKS, FROZEN17, frames, classify, R_MAX; validates nothing on import)

v3 = tm.v3


def replay(det, npz, mark, k):
    """One arm on one segment-phase -> dict of alert kinds + first correct alert time + per-frame faller facts."""
    v3.HELD_ALERT_MAX = k
    fr, t = tm.frames(npz)
    half = t[0] + 0.5 * (t[-1] - t[0])
    st = v3.V3MultiPersonFallState()
    kinds, first_correct, near, ids = [], None, [], set()
    for people, ti in zip(fr, t):
        hips = [(kp[v3.LEFT_HIP, :2] + kp[v3.RIGHT_HIP, :2]) / 2.0 for kp in people[:v3.NUM_POSES]]
        det._replay = [(kp, h) for kp, h in zip(people, hips)]
        res = v3.detect_v3_fall_multi(None, st, det, config=None)
        if ti >= half:
            # The faller = a position track_metric.classify calls 'correct': within R_MAX AND nearer the faller mark
            # than every other-person mark (Codex P2, 8 Oct: plain distance let a bystander count as the faller).
            near.append(any(tm.classify(np.asarray(h, float), mark) == 'correct' for h in hips))
            for tid, _det, _p, _l, c in res:
                tr = st.tracker.tracks.get(tid)
                if tr is not None and tr['missed'] == 0 and tm.classify(np.asarray(c, float)[:2], mark) == 'correct':
                    ids.add(int(tid))
        for tid, detected, prob, label, c in res:
            if detected:
                kind = tm.classify(np.asarray(c, float)[:2] if ti >= half else None, mark)
                kinds.append(kind)
                if kind == 'correct' and first_correct is None:
                    first_correct = round(float(ti - t[0]), 2)
    gap = run = 0
    for n in near:
        run = 0 if n else run + 1
        gap = max(gap, run)
    fps = (len(t) - 1) / max(t[-1] - t[0], 1e-6)
    return {'kinds': kinds, 'first_correct_s': first_correct,
            'coverage': round(sum(near) / max(len(near), 1), 3), 'longest_gap_frames': gap,
            'longest_gap_s': round(float(gap / fps), 2), 'faller_ids': len(ids)}


def main(k, model_dir, caches):
    if sorted(tm.MARKS) != sorted(tm.FROZEN17):
        raise SystemExit('marks do not match the frozen 17 segments')
    if len(caches) != 8 or len(set(map(os.path.abspath, caches))) != 8:
        raise SystemExit('need 8 distinct phase caches')
    keys = [json.load(open(os.path.join(c, 'key.json'))) for c in caches]
    if sorted(kk.get('roi_phase') for kk in keys) != list(range(8)):
        raise SystemExit('caches are not phases 0-7')
    det = v3.V3PoseFallDetector(model_dir=model_dir)
    det._replay = []
    det.extract_all_keypoints = lambda frame: det._replay
    rows = []
    for cache, key in zip(caches, keys):
        for seg, mark in tm.MARKS.items():
            f = os.path.join(cache, seg + '.npz')
            if not os.path.exists(f):
                raise SystemExit('missing cache file %s' % f)
            off, on = replay(det, f, mark, 0), replay(det, f, mark, k)
            if (off['coverage'], off['longest_gap_frames'], off['faller_ids']) != \
               (on['coverage'], on['longest_gap_frames'], on['faller_ids']):
                raise SystemExit('detection facts differ between arms on %s -- the arms are not paired' % f)
            row = {'phase': key['roi_phase'], 'segment': seg, 'excluded': bool(mark['exclude_wrong_person']),
                   'coverage': off['coverage'], 'longest_gap_frames': off['longest_gap_frames'],
                   'longest_gap_s': off['longest_gap_s'], 'faller_ids': off['faller_ids']}
            for arm, r in (('off', off), ('on', on)):
                row[arm] = {'correct': 'correct' in r['kinds'], 'wrong': 'wrong' in r['kinds'],
                            'first_correct_s': r['first_correct_s'],
                            'counts': {x: r['kinds'].count(x) for x in ('correct', 'wrong', 'far', 'tie', 'early')}}
            if not row['excluded']:
                a, b = row['off'], row['on']
                row['verdict'] = ('correct_lost' if a['correct'] and not b['correct'] else
                                  'correct_gained' if b['correct'] and not a['correct'] else
                                  'correct_kept' if a['correct'] else 'missed_both')
                row['wrong_change'] = ('removed' if a['wrong'] and not b['wrong'] else
                                       'added' if b['wrong'] and not a['wrong'] else 'same')
            rows.append(row)
    scored = [r for r in rows if not r['excluded']]
    summary = {'k': k, 'model_dir': model_dir, 'threshold': v3.THRESHOLD, 'segment_phases': len(rows),
               'scored_segment_phases': len(scored),
               'correct_off': sum(r['off']['correct'] for r in scored), 'correct_on': sum(r['on']['correct'] for r in scored),
               'wrong_off': sum(r['off']['wrong'] for r in scored), 'wrong_on': sum(r['on']['wrong'] for r in scored),
               'verdicts': {v: sum(r['verdict'] == v for r in scored)
                            for v in ('correct_kept', 'correct_lost', 'correct_gained', 'missed_both')},
               'wrong_changes': {v: sum(r['wrong_change'] == v for r in scored) for v in ('removed', 'added', 'same')},
               'missed_both_low_coverage': sum(r['verdict'] == 'missed_both' and r['coverage'] < 0.5 for r in scored),
               'missed_both_multi_ids': sum(r['verdict'] == 'missed_both' and r['faller_ids'] > 1 for r in scored)}
    print(json.dumps({'summary': summary, 'rows': rows}, indent=1))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3:])
