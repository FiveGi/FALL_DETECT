# -*- coding: utf-8 -*-
"""FALLPOSE-v1 primary metric: is the person detected WHILE falling and while lying after it?

URFD cam0 only: the dataset's own per-frame labels (urfall-cam0-falls.csv, column 3: -1 not lying,
0 falling, 1 lying; frame numbers match our cam0 mp4 frame counts exactly). URFD has one person
per clip, so "presence" is a fresh detection on that 8 fps sample (count > 0 in a pose cache);
held tracks do not count. Spans: descent = label 0 frames; after = the 2 s following descent.
Averaged within clip, then over clips. Half B is the gate (urfd_split.half); A is reported too.

Usage: python training/measure/fall_presence.py <cache_dir> [<cache_dir> ...]
"""
import collections
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'training'))
from urfd_split import half  # noqa: E402

LABELS = 'D:/project/PROJECT/datasets/urfd_labels/urfall-cam0-falls.csv'
SRC_FPS = 30.0


def spans():
    rows = collections.defaultdict(list)
    for r in csv.reader(open(LABELS)):
        rows[r[0]].append((int(r[1]), int(r[2])))
    out = {}
    for seq, fr in rows.items():
        fall = [f for f, l in fr if l == 0]
        if fall:
            t0, t1 = (fall[0] - 1) / SRC_FPS, fall[-1] / SRC_FPS
            out[seq] = (t0, t1, t1 + 2.0)
    return out


def presence(cache):
    res = {}
    for seq, (t0, t1, t2) in spans().items():
        f = os.path.join(cache, 'urfd_fall__%s-cam0.mp4.npz' % seq)
        if not os.path.exists(f):
            # Codex review: a missing clip must fail loudly, not shrink the denominator
            raise SystemExit('fall_presence: %s missing from %s' % (seq, cache))
        z = np.load(f)
        # cache_pose_streams yields the first source frame of each 8 fps slot: sample k is at k/8 s
        c = z['counts']
        t = np.arange(len(c)) / 8.0
        d = [c[i] > 0 for i in range(len(c)) if t0 <= t[i] < t1]
        a = [c[i] > 0 for i in range(len(c)) if t1 <= t[i] < t2]
        res[seq] = (np.mean(d) if d else np.nan, np.mean(a) if a else np.nan)
    return res


if __name__ == '__main__':
    for cache in sys.argv[1:]:
        res = presence(cache)
        for h in 'AB':
            v = [res[s] for s in res if half(s + '-cam0.mp4') == h]
            print('%s half %s n=%d  descent %.3f  after %.3f' % (cache, h, len(v), np.nanmean([x[0] for x in v]),
                                                                 np.nanmean([x[1] for x in v])))
