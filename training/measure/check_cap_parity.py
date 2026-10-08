# -*- coding: utf-8 -*-
"""plan_v5 T2c: an uncapped pose cache must reproduce the pinned cap-4 cache exactly when cut to 4.

The cap-4-vs-8 comparison replays the SAME uncapped pose stream at both caps, which is only a fair
stand-in for the pinned gates if its first four people per frame are the pinned cache's people. A pose
pass re-run on the same CPU with the same profile should give identical keypoints; if it does not, the
comparison is answering a different question and must stop here (Codex: parity assert, stop on mismatch).

Also reports the pre-cap people counts the cap decision needs: frames with more than 4 detections.

Usage: python training/measure/check_cap_parity.py [--expect N] PINNED_DIR UNCAPPED_DIR [PINNED_DIR UNCAPPED_DIR ...]
Exit 1 on any mismatch: segment sets not identical (or empty, or not N when --expect is given), different frame
counts, different frame timestamps, or first-4 keypoints not identical (Codex review 2026-10-07).
"""
import glob
import json
import os
import sys

import numpy as np


def frames(npz):
    with np.load(npz) as z:
        counts, kpts = z['counts'], z['kpts']
        t = z['t'] if 't' in z.files else None   # owner caches carry frame times; replay uses them
    out, at = [], 0
    for n in counts:
        out.append(kpts[at:at + int(n)])
        at += int(n)
    if at != len(kpts):
        raise ValueError('%s: counts sum %d != %d keypoint rows' % (npz, at, len(kpts)))
    return counts, out, t


def check(pinned, uncapped, expect=None):
    kp, ku = (json.load(open(os.path.join(d, 'key.json'))) for d in (pinned, uncapped))
    differ = sorted(k for k in set(kp) | set(ku) if kp.get(k) != ku.get(k) and k != 'num_poses')
    if differ:
        return ['profile differs beyond num_poses: %s' % differ], {}
    bad, stats = [], {'segments': 0, 'frames': 0, 'frames_over_4': 0, 'max_people': 0, 'segments_over_4': []}
    names = sorted(os.path.basename(f) for f in glob.glob(os.path.join(pinned, '*.npz')))
    unames = sorted(os.path.basename(f) for f in glob.glob(os.path.join(uncapped, '*.npz')))
    if not names:
        return ['pinned cache is empty'], {}
    if names != unames:
        return ['segment sets differ: %d pinned, %d uncapped, only-pinned %s, only-uncapped %s'
                % (len(names), len(unames), sorted(set(names) - set(unames))[:3],
                   sorted(set(unames) - set(names))[:3])], {}
    if expect is not None and len(names) != expect:
        return ['%d segments, expected %d' % (len(names), expect)], {}
    for name in names:
        u = os.path.join(uncapped, name)
        cp, fp, tp = frames(os.path.join(pinned, name))
        cu, fu, tu = frames(u)
        if len(cp) != len(cu):
            bad.append('%s: %d vs %d frames' % (name, len(cp), len(cu)))
            continue
        if (tp is None) != (tu is None) or (tp is not None and not np.array_equal(tp, tu)):
            bad.append('%s: frame timestamps differ' % name)
            continue
        over = 0
        for i, (a, b) in enumerate(zip(fp, fu)):
            if len(a) != min(len(b), 4) or not np.array_equal(a, b[:4]):
                bad.append('%s frame %d: pinned %d people, uncapped first-4 differ' % (name, i, len(a)))
                break
            over += len(b) > 4
            stats['max_people'] = max(stats['max_people'], len(b))
        stats['segments'] += 1
        stats['frames'] += len(cu)
        stats['frames_over_4'] += over
        if over:
            stats['segments_over_4'].append((name, over))
    return bad, stats


def main():
    args = sys.argv[1:]
    expect = None
    if args[:1] == ['--expect']:
        expect, args = int(args[1]), args[2:]
    if not args or len(args) % 2:
        raise SystemExit(__doc__)
    failed = False
    for pinned, uncapped in zip(args[::2], args[1::2]):
        bad, st = check(pinned, uncapped, expect)
        tag = '%s vs %s' % (os.path.basename(os.path.normpath(pinned)), os.path.basename(os.path.normpath(uncapped)))
        if bad:
            failed = True
            print('MISMATCH', tag, '|', '; '.join(bad[:5]), '(%d problems)' % len(bad))
        else:
            print('PARITY OK', tag, '| segments %d frames %d | frames with >4 people %d | max %d | segments %s'
                  % (st['segments'], st['frames'], st['frames_over_4'], st['max_people'], st['segments_over_4'][:10]))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
