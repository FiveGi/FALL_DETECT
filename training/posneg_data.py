# -*- coding: utf-8 -*-
"""T3 POSE-NEG training list (Codex design 2026-10-03): COCO-pose train with a fixed 10% share of
PERSON-FREE COCO-train backgrounds (indoor clutter; D:/project/PROJECT/datasets/posneg/images, no label
file = background for ultralytics). Same total images as the control (nightaug, which saw none), so the
training budget is unchanged. 5,000 unique backgrounds, first 5,000 of the pre-shuffled candidates.
Usage: SEED=42 python training/posneg_data.py -> prints the yaml to pass as COCO_POSE_YAML"""
import json
import os
import random

COCO = 'D:/project/PROJECT/datasets/coco-pose'
NEG = 'D:/project/PROJECT/datasets/posneg'
SHARE, N_UNIQUE = 0.10, 5000
SEED = int(os.environ.get('SEED', 42))

coco = [os.path.normpath(os.path.join(COCO, l.strip())).replace(chr(92), '/')
        for l in open(os.path.join(COCO, 'train2017.txt')) if l.strip()]
cands = json.load(open(os.path.join(NEG, 'candidates.json')))
neg = [os.path.join(NEG, 'images', c['file']).replace(chr(92), '/') for c in cands
       if os.path.exists(os.path.join(NEG, 'images', c['file']))][:N_UNIQUE]
assert len(neg) == N_UNIQUE, 'only %d backgrounds downloaded' % len(neg)
for p in neg:   # a background must have NO label file
    lab = p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
    assert not os.path.exists(lab), lab
total = len(coco); n_neg = int(round(SHARE * total))
rng = random.Random(SEED)
lines = rng.sample(coco, total - n_neg) + [neg[i % len(neg)] for i in range(n_neg)]
rng.shuffle(lines)
lst = os.path.join(NEG, 'train_neg_s%d.txt' % SEED)
open(lst, 'w').write('\n'.join(lines) + '\n')
src = open(os.path.join(os.path.dirname(COCO), 'coco-pose.yaml'), encoding='utf-8').read()
yaml = os.path.join(NEG, 'posneg_s%d.yaml' % SEED)
open(yaml, 'w', encoding='utf-8').write('\n'.join(
    ('path: %s' % COCO) if l.startswith('path:') else ('train: %s' % lst) if l.startswith('train:') else l
    for l in src.splitlines()) + '\n')
json.dump({'total': total, 'coco': total - n_neg, 'neg_unique': len(neg), 'neg_draws': n_neg, 'seed': SEED},
          open(os.path.join(NEG, 'train_neg_s%d.json' % SEED), 'w'), indent=1)
print(yaml)
