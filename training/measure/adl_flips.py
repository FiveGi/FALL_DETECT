# -*- coding: utf-8 -*-
"""Which URFD ADL clips alert under pose cache B but not under pose cache A (same classifier, fixed
threshold)? Counts over all given phase pairs. Usage: adl_flips.py <thr> <A1> <B1> [<A2> <B2> ...]"""
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import eval_candidate as ec  # noqa: E402

thr = float(sys.argv[1]); pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
det = ec.v3.V3PoseFallDetector(model_dir=os.path.join(ec.ROOT, 'models'))
det._replay = []
det.extract_all_keypoints = lambda frame: det._replay
gain, lose = collections.Counter(), collections.Counter()
for a, b in pairs:
    sa = ec.score(det, ec.load_group(a, 'urfd_adl'), thr)
    sb = ec.score(det, ec.load_group(b, 'urfd_adl'), thr)
    for k in sa:
        if sb[k] and not sa[k]: gain[k] += 1
        if sa[k] and not sb[k]: lose[k] += 1
print('new false alarms (clip: phases):', dict(gain.most_common()))
print('false alarms removed:', dict(lose.most_common()))
