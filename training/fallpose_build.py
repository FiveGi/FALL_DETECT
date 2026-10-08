# -*- coding: utf-8 -*-
"""FALLPOSE-v1 data: pseudo-labelled keypoints of people falling and lying, from CAUCAFall.

The deployed pose model (yolo26s-pose, 320 px) loses a person lying on the floor most of the time
(URFD, 2 s after the fall: 0.27-0.62 of frames detected by day, 0.04-0.18 at night;
training/measure/fall_presence.py). yolo26x-pose at 1280 px sees them (0.86, teacher_presence.py).
This writes the teacher's keypoints as training labels so the small model can learn those poses.

Why CAUCAFall only (v1, Codex review 2026-10-01): it is the one source whose frames carry a box and
a fall class drawn by people (a YOLO txt beside every png), so the teacher is checked against a
human box rather than trusted on its own confidence, and spans come from the dataset, not from
pose-motion guesses. Le2i stays out until its licence terms are on file.

Frames, per subject and activity (10 subjects; 5 fall + 5 everyday activities), sampled evenly:
  fall activity: 8 standing frames before the fall (class 0), the first 16 fallen-class frames at
                 every 2nd frame (the descent at 20 fps), 16 more over the rest of the fallen span
  everyday activity (hop, kneel, pick up, sit down, walk): 12 frames -- unusual poses that are NOT
                 falls, so the pose model is not taught that low bodies only happen in falls
Accepted when the teacher's best box overlaps the human box at IoU >= 0.5 and >= 8 keypoints have
confidence >= 0.5. Keypoints under 0.5 are written as not visible (0 0 0), never as invented
positions. The box written is the HUMAN box. Every rejected frame is listed with its reason: that
list is the teacher's blind-spot audit.
Split by subject: 1-8 train, 9-10 held out (val), never mixed.

Out: D:/project/PROJECT/datasets/fallpose/{images,labels}/{train,val}/, manifest.json, rejects.csv
Usage: python training/fallpose_build.py
"""
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = 'D:/project/PROJECT/Dataset CAUCAFall/CAUCAFall'
OUT = 'D:/project/PROJECT/datasets/fallpose'
TEACHER = os.path.join(ROOT, 'models', 'yolo26x-pose.pt')
TEACHER_IMGSZ = 1280
VAL_SUBJECTS = {'Subject.9', 'Subject.10'}
MIN_IOU, MIN_KPT_CONF, MIN_KPTS = 0.5, 0.5, 8


def evenly(items, n):
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def read_label(txt):
    s = open(txt).read().split()
    if len(s) < 5:
        return None, None
    return int(s[0]), [float(v) for v in s[1:5]]   # class, (xc, yc, w, h) normalised


def pick_frames(folder, is_fall):
    # A few frames are duplicated as '<name> - copia.png' with no label beside them: skip those.
    pngs = sorted(p for p in glob.glob(os.path.join(folder, '*.png'))
                  if 'copia' not in p and os.path.exists(p[:-4] + '.txt'))
    lab = [(p, *read_label(p[:-4] + '.txt')) for p in pngs]
    lab = [x for x in lab if x[1] is not None]
    if not is_fall:
        return [(p, 'everyday') for p, _, _ in evenly(lab, 12)]
    fallen = [x for x in lab if x[1] == 1]
    before = [x for x in lab if x[1] == 0 and (not fallen or x[0] < fallen[0][0])]
    descent = fallen[0:32:2]
    rest = [x for x in fallen[32:]]
    out = [(p, 'standing') for p, _, _ in evenly(before, 8)]
    out += [(p, 'descent') for p, _, _ in descent]
    out += [(p, 'lying') for p, _, _ in evenly(rest, 16)]
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-9)


def main():
    from ultralytics import YOLO
    teacher = YOLO(TEACHER)
    for split in ('train', 'val'):
        for kind in ('images', 'labels'):
            os.makedirs(os.path.join(OUT, kind, split), exist_ok=True)
    rejects, stats = [], {}
    for folder in sorted(glob.glob(os.path.join(SRC, 'Subject.*', '*'))):
        subject, activity = folder.replace('\\', '/').split('/')[-2:]
        split = 'val' if subject in VAL_SUBJECTS else 'train'
        for png, stratum in pick_frames(folder, activity.startswith('Fall')):
            key = (split, stratum)
            stats.setdefault('%s/%s' % key, [0, 0])
            stats['%s/%s' % key][1] += 1
            img = cv2.imread(png)
            h, w = img.shape[:2]
            _, (xc, yc, bw, bh) = read_label(png[:-4] + '.txt')
            human = ((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)
            r = teacher.predict(img, verbose=False, conf=0.25, classes=[0], imgsz=TEACHER_IMGSZ)[0]
            name = '%s_%s_%s' % (subject.replace('.', ''), activity.replace(' ', ''), os.path.basename(png)[:-4])
            if r.boxes is None or len(r.boxes) == 0:
                rejects.append([split, stratum, png, 'no_detection', '', '']); continue
            boxes = r.boxes.xyxy.cpu().numpy()
            ious = [iou(b, human) for b in boxes]
            i = int(np.argmax(ious))
            kc = r.keypoints.conf[i].cpu().numpy()
            kxy = r.keypoints.xy[i].cpu().numpy()
            n_ok = int((kc >= MIN_KPT_CONF).sum())
            if ious[i] < MIN_IOU:
                rejects.append([split, stratum, png, 'iou', round(ious[i], 3), n_ok]); continue
            if n_ok < MIN_KPTS:
                rejects.append([split, stratum, png, 'few_keypoints', round(ious[i], 3), n_ok]); continue
            kp = []
            for (x, y), c in zip(kxy, kc):
                kp += [x / w, y / h, 2] if c >= MIN_KPT_CONF else [0, 0, 0]
            line = '0 %.6f %.6f %.6f %.6f ' % (xc, yc, bw, bh) + ' '.join(
                ('%.6f' % v) if isinstance(v, float) or isinstance(v, np.floating) else str(v) for v in kp)
            cv2.imwrite(os.path.join(OUT, 'images', split, name + '.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            open(os.path.join(OUT, 'labels', split, name + '.txt'), 'w').write(line + '\n')
            stats['%s/%s' % key][0] += 1
        print(subject, activity, flush=True)
    with open(os.path.join(OUT, 'rejects.csv'), 'w', newline='') as fh:
        csv.writer(fh).writerows([['split', 'stratum', 'frame', 'reason', 'iou', 'kpts_ok']] + rejects)
    json.dump({'teacher': TEACHER, 'teacher_imgsz': TEACHER_IMGSZ, 'min_iou': MIN_IOU,
               'min_kpt_conf': MIN_KPT_CONF, 'min_kpts': MIN_KPTS, 'val_subjects': sorted(VAL_SUBJECTS),
               'accepted_of_selected': stats}, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    sys.exit(main())
