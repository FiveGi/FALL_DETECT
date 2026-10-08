# -*- coding: utf-8 -*-
"""FALLPOSE-v1 training list: COCO-pose train with a fixed 15% share of fall/lying frames.

Codex review 2026-10-01: same number of training images as the COCO-only control (colour_s<seed>),
so optimisation steps match; 15% of them are fallpose frames (training/fallpose_build.py), the
rest a seeded random subset of COCO train. Fallpose frames are repeated to reach the share --
there are far fewer of them than 15% of COCO. Realised counts are printed and saved.
Validation stays COCO val, colour, so checkpoint selection never sees a fall frame.

Usage: SEED=42 python training/fallpose_data.py  -> prints the yaml path to pass as COCO_POSE_YAML
"""
import glob
import json
import os
import random

COCO = 'D:/project/PROJECT/datasets/coco-pose'
FALL = 'D:/project/PROJECT/datasets/fallpose'
SHARE = 0.15
SEED = int(os.environ.get('SEED', 42))


def main():
    coco = [os.path.normpath(os.path.join(COCO, l.strip())).replace('\\', '/')
            for l in open(os.path.join(COCO, 'train2017.txt')) if l.strip()]
    fall = sorted(p.replace('\\', '/') for p in glob.glob(os.path.join(FALL, 'images', 'train', '*.jpg')))
    total = len(coco)
    n_fall = int(round(SHARE * total))
    reps = [fall[i % len(fall)] for i in range(n_fall)]
    rng = random.Random(SEED)
    keep = rng.sample(coco, total - n_fall)
    lines = keep + reps
    rng.shuffle(lines)
    lst = os.path.join(FALL, 'train_mix_s%d.txt' % SEED)
    open(lst, 'w').write('\n'.join(lines) + '\n')
    yaml = os.path.join(FALL, 'fallpose_mix_s%d.yaml' % SEED)
    src = open(os.path.join(os.path.dirname(COCO), 'coco-pose.yaml'), encoding='utf-8').read()
    out = []
    for l in src.splitlines():
        if l.startswith('path:'):
            l = 'path: %s' % COCO
        elif l.startswith('train:'):
            l = 'train: %s' % lst
        out.append(l)
    open(yaml, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    stats = {'total': len(lines), 'coco': len(keep), 'fall_unique': len(fall), 'fall_draws': len(reps),
             'share': round(len(reps) / len(lines), 4), 'seed': SEED}
    json.dump(stats, open(os.path.join(FALL, 'train_mix_s%d.json' % SEED), 'w'), indent=1)
    print(json.dumps(stats))
    print(yaml)


if __name__ == '__main__':
    main()
