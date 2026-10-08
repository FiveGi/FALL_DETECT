# -*- coding: utf-8 -*-
"""Plan round 3 D5: why are falls missed, per size bucket (owner_size_buckets_v4.json)? For each fall segment
a config misses in >= 6 of 8 phases: 'pose absent' if the pose cache (phase 0) shows nobody in >= 50% of the
segment's last 2 s (the person on the floor), else 'classifier rejects' (poses present, peak score under the
threshold). Configs: deployed (stock pose) and N1 s44 (POSE-IR s44 pose + deployed classifier).
Usage: python training/measure/miss_causes.py"""
import json, os
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
B = json.load(open('training/data/day_first/owner_size_buckets_v4.json'))
CFG = {'deployed': ('parity_deployed', 'training/data/pose_cache_owner/23de875c4696'),
       'N1_s44': ('N1_nightaug_s44_v2', None)}
lst = open('training/data/stage1/nightaug_s44_cache_lists.txt').read().split('\n')
CFG['N1_s44'] = ('N1_nightaug_s44_v2', [l for l in lst if l.startswith('OWNER')][0].split()[1])
for name, (row, cache) in CFG.items():
    runs = [json.load(open('training/data/stage1/%s_pinned_owner%d.json' % (row, p)))['segments'] for p in range(8)]
    out = {}
    for seg, v in B.items():
        if v['bucket'] == 'unknown': continue
        hits = sum(bool(r[seg]['alerts_at']) for r in runs)
        if hits > 2: continue
        z = np.load(os.path.join(cache, seg + '.npz')); c, t = z['counts'], z['t']
        tail = c[t >= t[-1] - 2.0]
        cause = 'pose absent' if (tail == 0).mean() >= 0.5 else 'classifier rejects'
        peak = max(r[seg]['peak'] for r in runs)
        out.setdefault(v['bucket'], []).append((seg, cause, round(peak, 2)))
    for b in ('near', 'medium', 'far'):
        L = out.get(b, []); n_abs = sum(1 for x in L if x[1] == 'pose absent')
        print('%-9s %-6s missed %2d: pose absent %2d, classifier rejects %2d' % (name, b, len(L), n_abs, len(L) - n_abs))
