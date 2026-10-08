# -*- coding: utf-8 -*-
"""Print pinned cache lists (DAY 8 / IR 3 / OWNER 8) for a pose checkpoint tag built on the CPU
(training/data/pose_ir_eval_cpu/<tag>, owner caches keyed by pose_model containing <tag>_p2).
Fails unless every list is complete. Usage: python training/measure/list_model_caches.py <tag>"""
import glob
import json
import os
import sys

tag = sys.argv[1]
day, ir, own = {}, {}, {}
for k in glob.glob('training/data/pose_ir_eval_cpu/%s/*/key.json' % tag):
    j = json.load(open(k)); d = os.path.dirname(k).replace(os.sep, '/')
    if len(glob.glob(d + '/*.npz')) < 220 or j.get('night_alt'):
        continue
    (ir.__setitem__(j['simulation_seed'], d) if j['simulate_ir'] else day.__setitem__(j['roi_phase'], d))
for k in glob.glob('training/data/pose_cache_owner/*/key.json'):
    j = json.load(open(k))
    if (tag + '_p2') in j.get('pose_model', '') and os.path.exists(os.path.join(os.path.dirname(k), 'done')):
        own[j['roi_phase']] = os.path.dirname(k).replace(os.sep, '/')
assert sorted(day) == list(range(8)) and sorted(ir) == [0, 7, 13] and sorted(own) == list(range(8)), \
    'incomplete: day %s ir %s owner %s' % (sorted(day), sorted(ir), sorted(own))
print('DAY', ' '.join(day[p] for p in range(8)))
print('IR', ' '.join(ir[s] for s in (0, 7, 13)))
print('OWNER', ' '.join(own[p] for p in range(8)))
