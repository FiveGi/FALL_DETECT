# -*- coding: utf-8 -*-
"""Phantom people in EMPTY home scenes: stock yolo26s-pose vs option A (pose_nightaug_s44), 8 Oct overnight.

Why: A saw a bench as a person in the empty coffee room (8/12 frames; stock 0/12) and the alone alert (owner: ON) counts
exactly these detections. Le2i half 1 could not be used: every annotated frame has a person in it (amendment logged in
AI_HANDOFF.md before any data was seen).

Images: COCO val2017 with ZERO person annotations (crowd included) and at least one home-furniture object (chair, couch,
bed, dining table, tv) -- human-labelled empty-of-people indoor scenes; not used in pose training (coco-pose holds person
images only). Deterministic sample of N (sorted ids, even stride). Both models at the fall path's input (320) and the
alone count's input (960), conf 0.30. Per model and size: share of images with >= 1 phantom person, max conf, and per
furniture category.

Usage: python training/measure/phantom_probe_coco.py > training/data/system_test/phantom_probe_coco.txt
"""
import json
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
COCO = os.path.join(os.path.dirname(ROOT), 'datasets', 'coco-pose')
N = int(os.environ.get('PHANTOM_N', 300))
FURNITURE = {'chair', 'couch', 'bed', 'dining table', 'tv'}
torch.set_num_threads(4)


def main():
    inst = json.load(open(os.path.join(COCO, 'annotations', 'instances_val2017.json')))
    names = {c['id']: c['name'] for c in inst['categories']}
    person = next(i for i, n in names.items() if n == 'person')
    cats = {}
    for a in inst['annotations']:
        cats.setdefault(a['image_id'], set()).add(names[a['category_id']])
    has_person = {a['image_id'] for a in inst['annotations'] if a['category_id'] == person}
    pool = sorted(i for i, c in cats.items() if i not in has_person and c & FURNITURE)
    pick = [pool[k] for k in np.linspace(0, len(pool) - 1, min(N, len(pool))).astype(int)]
    if os.environ.get('IDS_FILE'):   # a frozen id list (dev / confirmation, E5 pre-registration) instead of sampling
        pick = [int(l) for l in open(os.environ['IDS_FILE']) if l.strip()]
        if not set(pick) <= set(pool):
            raise SystemExit('ids outside the empty-home pool')
    files = {im['id']: im['file_name'] for im in inst['images']}
    models = {'stock': YOLO('models/yolo26s-pose.pt'), 'A': YOLO('models/pose_nightaug_s44.pt')}
    for extra in filter(None, os.environ.get('EXTRA_MODELS', '').split(',')):   # name=path, e.g. E5=...last.pt
        name, path = extra.split('=', 1)
        models[name] = YOLO(path)
    tally = {(m, s): [0, 0.0] for m in models for s in (320, 960)}
    by_cat = {}
    per_image = {}   # image id -> {"model@size": max person conf or 0} -- paired comparisons (Codex P2, 8 Oct)
    for iid in pick:
        img = cv2.imread(os.path.join(COCO, 'images', 'val2017', files[iid]))
        if img is None:
            raise SystemExit('cannot read %s' % files[iid])
        for mn, model in models.items():
            for sz in (320, 960):
                r = model.predict(img, imgsz=sz, conf=0.30, classes=[0], device='cpu', verbose=False)[0]
                confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
                per_image.setdefault(str(iid), {})['%s@%d' % (mn, sz)] = round(float(confs.max()), 3) if len(confs) else 0.0
                if len(confs):
                    tally[(mn, sz)][0] += 1
                    tally[(mn, sz)][1] = max(tally[(mn, sz)][1], float(confs.max()))
                for c in cats[iid] & FURNITURE:
                    t = by_cat.setdefault((c, mn, sz), [0, 0])
                    t[0] += bool(len(confs)); t[1] += 1
    if os.environ.get('PER_IMAGE_OUT'):
        if os.path.exists(os.environ['PER_IMAGE_OUT']):
            raise SystemExit('PER_IMAGE_OUT exists, refusing to overwrite')
        json.dump(per_image, open(os.environ['PER_IMAGE_OUT'], 'w'), indent=0)
    print('COCO val2017 empty-of-people home scenes: pool %d, sampled %d' % (len(pool), len(pick)))
    for (mn, sz), (hit, mx) in tally.items():
        print('%-5s imgsz %4d: images with a phantom person %3d/%d (%.1f%%), max conf %.2f'
              % (mn, sz, hit, len(pick), 100.0 * hit / len(pick), mx))
    for k in sorted(by_cat):
        h, n = by_cat[k]
        print('  %-13s %-5s %4d: %3d/%d (%.1f%%)' % (k[0], k[1], k[2], h, n, 100.0 * h / n))


if __name__ == '__main__':
    main()
