# -*- coding: utf-8 -*-
"""T3 POSE-NEG metric (owner 2026-10-02: "objects must not be detected as people"): false persons per
1,000 PERSON-FREE images. Test set = COCO val2017 images with no person annotation and at least one
indoor-clutter object (furniture, bags, clothes, pets, ...) -- val, so it never overlaps the person-free
TRAIN images used as hard negatives (Codex D3). Pose model at the deployed 320 px, conf 0.30, full frame.
Also reports how many images had >=1 false person and the mean confidence of the false persons.
Usage: python training/measure/object_false_person.py model@imgsz [model@imgsz ...]"""
import collections
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

ANN = 'D:/project/PROJECT/datasets/coco-pose/annotations/instances_val2017.json'
IMG = 'D:/project/PROJECT/datasets/coco-pose/images/val2017'
INDOOR = {'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'backpack', 'handbag', 'suitcase',
          'umbrella', 'teddy bear', 'potted plant', 'refrigerator', 'oven', 'sink', 'book', 'clock', 'vase',
          'cat', 'dog', 'bench', 'tie'}


def test_set():
    d = json.load(open(ANN))
    cat = {c['id']: c['name'] for c in d['categories']}
    cs = collections.defaultdict(set)
    for a in d['annotations']:
        cs[a['image_id']].add(cat[a['category_id']])
    files = {i['id']: i['file_name'] for i in d['images']}
    ids = sorted(i for i, c in cs.items() if 'person' not in c and c & INDOOR)
    return [os.path.join(IMG, files[i]) for i in ids if os.path.exists(os.path.join(IMG, files[i]))]


if __name__ == '__main__':
    paths = test_set()
    print('person-free indoor-clutter COCO val images found locally:', len(paths), flush=True)
    for arg in sys.argv[1:]:
        name, imgsz = arg.rsplit('@', 1)
        m = YOLO(name)
        n_det, n_img, confs = 0, 0, []
        for p in paths:
            b = m.predict(cv2.imread(p), verbose=False, conf=0.30, classes=[0], imgsz=int(imgsz), device='cpu')[0].boxes
            k = 0 if b is None else len(b)
            n_det += k; n_img += k > 0
            if k: confs += b.conf.cpu().numpy().tolist()
        print('%-60s false persons %.1f /1000 imgs; images with any %d/%d; mean conf %.2f' % (
            arg, 1000.0 * n_det / max(len(paths), 1), n_img, len(paths), np.mean(confs) if confs else 0), flush=True)
