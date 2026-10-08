# -*- coding: utf-8 -*-
"""DAY-FIRST-v1 table from existing pinned owner results (8 phases each): falls caught per near/far
bucket (owner_size_buckets_v4.json, frozen from stock poses) and on the frozen 17-segment
residential-adult multi-person list (Codex MULTI-DIAG-v2), mean over phases (min in brackets).
Usage: python training/measure/day_first_table.py name [name ...]   (names in results_pinned.jsonl)"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
B = json.load(open('training/data/day_first/owner_size_buckets_v4.json'))
MULTI17 = ['1.mp4#5', '1.mp4#10', '1.mp4#12', '1.mp4#13', '1.mp4#15', '4.mp4#1', '4.mp4#7', '4.mp4#16',
           '5.mp4#3', '5.mp4#16', '8.mp4#1', '8.mp4#6', '8.mp4#12', '9.mp4#5', '9.mp4#8', '9.mp4#17', '9.mp4#20']
groups = {b: [s for s, v in B.items() if v['bucket'] == b] for b in ('far', 'medium', 'near', 'unknown')}
groups['multi17'] = MULTI17
print('%-28s ' % 'model' + '  '.join('%-14s' % ('%s /%d' % (g, len(v))) for g, v in groups.items()))
for name in sys.argv[1:]:
    runs = [json.load(open('training/data/stage1/%s_pinned_owner%d.json' % (name, p)))['segments'] for p in range(8)]
    cells = []
    for g, segs in groups.items():
        per = [sum(bool(r[s]['alerts_at']) for s in segs) for r in runs]
        cells.append('%5.1f (%2d)    ' % (sum(per) / len(per), min(per)))
    print('%-28s ' % name + '  '.join(cells))
