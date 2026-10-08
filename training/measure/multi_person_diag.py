# -*- coding: utf-8 -*-
"""Why are multi-person falls missed? Per genuine multi-person fall segment (25, owner list):
how often the deployed model and ensemble E1 alert over the 8 crop phases, their peak score, and
what the pose pass saw (people per frame, frames with nobody) from the stock owner caches.
Usage: python training/measure/multi_person_diag.py > training/data/multi_diag/table.txt"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
segs = [l.strip() for l in open('../.ai_evidence/multi_person_genuine.txt') if l.strip()]
dep = [json.load(open('test_result/incidents/alerts_cpu_320px_8fps_roi256_phase%d.json' % p))['segments'] for p in range(8)]
ens = [json.load(open('training/data/stage1/ensA_42_43_44_owner_phase%d.json' % p))['segments'] for p in range(8)]
caches = {}
for k in glob.glob('training/data/pose_cache_owner/*/key.json'):
    key = json.load(open(k))
    if key['pose_conf'] == 0.3 and 'pose_model' not in key:
        caches[key['roi_phase']] = os.path.dirname(k)
print('%-12s %6s %6s %6s %6s  %5s %5s %5s %5s' % ('segment', 'dep/8', 'dpeak', 'ens/8', 'epeak', 'mean', 'max', 'zero%', 'one%'))
for s in segs:
    hit = lambda runs: sum(bool(r[s]['alerts_at']) for r in runs if s in r)
    peak = lambda runs: max(r[s].get('peak', 0) for r in runs if s in r)
    c = np.load(os.path.join(caches[0], s + '.npz'))['counts'].astype(float)
    print('%-12s %6d %6.2f %6d %6.2f  %5.2f %5d %5.0f %5.0f' % (s, hit(dep), peak(dep), hit(ens), peak(ens),
          c.mean(), c.max(), 100 * (c == 0).mean(), 100 * (c == 1).mean()))
