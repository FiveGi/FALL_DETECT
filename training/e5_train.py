# -*- coding: utf-8 -*-
"""E5 (pre-registered .ai_evidence/plan_oct7_8/e5_preregistration.md): continue option A's pose model with background
images (E5) or without (A-cont control). Exploration only -- not in the 9 Oct build.

  ARM=e5     python training/e5_train.py     -> training/data/pose_ir/e5_bg10_s45/weights/last.pt
  ARM=acont  python training/e5_train.py     -> training/data/pose_ir/acont_s45/weights/last.pt
  ARM=list   python training/e5_train.py     -> only writes the background id list + the E5 train list

Same NightDegrade hook and trainer as A (training/finetune_pose_ir.py). Backgrounds: COCO train2017 images with zero
person annotations (crowd included) containing chair / couch / bed / dining table and NOT tv, sorted ids, even stride to
10 % of the 56,599 person images. They have no label file, so Ultralytics trains them as background.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finetune_pose_ir as ft   # noqa: E402  (NightDegrade, trainer_class, PROJECT)
from ultralytics import YOLO   # noqa: E402

ARM = os.environ.get('ARM', 'list')
SEED = 45
COCO = 'D:/project/PROJECT/datasets/coco-pose'
A_WEIGHTS = os.path.join(ft.PROJECT, 'nightaug_s44_p2', 'weights', 'best.pt')
BG_IDS = os.path.join(ft.PROJECT, 'e5_backgrounds.txt')
FURNITURE, EXCLUDE = {'chair', 'couch', 'bed', 'dining table'}, {'tv'}


def write_lists():
    inst = json.load(open('D:/project/PROJECT/datasets/coco_ann/instances_train2017.json'))
    names = {c['id']: c['name'] for c in inst['categories']}
    cats = {}
    for a in inst['annotations']:
        cats.setdefault(a['image_id'], set()).add(names[a['category_id']])
    person_imgs = {a['image_id'] for a in inst['annotations'] if names[a['category_id']] == 'person'}
    pool = sorted(i for i, c in cats.items() if i not in person_imgs and c & FURNITURE and not c & EXCLUDE)
    base = [l.strip() for l in open(os.path.join(COCO, 'train2017.txt')) if l.strip()]
    n = round(0.10 * len(base))
    if len(pool) < n:
        raise SystemExit('pool %d < %d' % (len(pool), n))
    pick = [pool[k] for k in np.linspace(0, len(pool) - 1, n).astype(int)]
    assert len(set(pick)) == n
    files = {im['id']: im['file_name'] for im in inst['images']}
    bg = ['./images/train2017/%s' % files[i] for i in pick]
    in_base = set(base)
    assert not in_base & set(bg), 'a background image is already a person image'
    for p in bg:
        if not os.path.exists(os.path.join(COCO, p[2:])):
            raise SystemExit('missing image %s' % p)
        if os.path.exists(os.path.join(COCO, 'labels', 'train2017', os.path.basename(p)[:-4] + '.txt')):
            raise SystemExit('background has a label file: %s' % p)
    open(BG_IDS, 'w').write('\n'.join(map(str, pick)) + '\n')
    open(os.path.join(COCO, 'train2017_e5.txt'), 'w').write('\n'.join(base + bg) + '\n')
    # A-cont gets the SAME number of images per epoch (Codex P2): 10 % of the person images listed twice, so both arms
    # run identical batch counts and learning-rate schedules; the only difference is what the extra 10 % contains.
    dup = [base[k] for k in np.linspace(0, len(base) - 1, n).astype(int)]
    assert len(set(dup)) == n
    open(os.path.join(COCO, 'train2017_acont.txt'), 'w').write('\n'.join(base + dup) + '\n')
    yaml = open('D:/project/PROJECT/datasets/coco-pose.yaml', encoding='utf-8').read().replace(
        'train: train2017.txt', 'train: train2017_e5.txt')
    assert 'train2017_e5.txt' in yaml
    open('D:/project/PROJECT/datasets/coco-pose-e5.yaml', 'w', encoding='utf-8').write(yaml)
    open('D:/project/PROJECT/datasets/coco-pose-acont.yaml', 'w', encoding='utf-8').write(
        yaml.replace('train2017_e5.txt', 'train2017_acont.txt'))
    print('pool %d, backgrounds %d, person images %d -> train2017_e5.txt %d lines' % (len(pool), n, len(base), len(base) + n))


def train(data, name):
    common = dict(data=data, imgsz=320, batch=64, workers=int(os.environ.get('WORKERS', 2)), seed=SEED,
                  deterministic=False, optimizer='AdamW', project=ft.PROJECT, exist_ok=False, plots=False, verbose=False)
    YOLO(A_WEIGHTS).train(trainer=ft.trainer_class(SEED), epochs=3, lr0=2e-5, lrf=0.1, cos_lr=True,
                          warmup_epochs=0, name=name, **common)
    print('done', os.path.join(ft.PROJECT, name, 'weights', 'last.pt'))


if __name__ == '__main__':
    if ARM == 'list':
        write_lists()
    elif ARM == 'e5':
        train('D:/project/PROJECT/datasets/coco-pose-e5.yaml', 'e5_bg10_s45')
    elif ARM == 'acont':
        train('D:/project/PROJECT/datasets/coco-pose-acont.yaml', 'acont_s45')
    else:
        raise SystemExit(__doc__)
