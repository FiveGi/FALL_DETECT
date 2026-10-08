# -*- coding: utf-8 -*-
"""Phantom people in EMPTY frames: stock yolo26s-pose vs option A (pose_nightaug_s44). Overnight finding 8 Oct: A sees a
bench as a person in the empty coffee room (8/12 frames) where stock sees nobody. No gate measures this, and the alone
alert (owner: ON) counts exactly these detections.

Frames: Le2i HALF 1 only (training/data/le2i_holdout_manifest.json half1_final_test; half 2 stays reserved and is never
opened). Le2i annotation rows are 'frame,label,x1,y1,x2,y2'; a row with an all-zero box = nobody in the frame. A frame is
used only if it is >= GUARD frames away from every frame with a person (no one half-way through the door), sampled
evenly, at most PER_VIDEO per video. Each frame goes through both models at the fall path's input (320, conf 0.30) and
the alone count's input (960, conf 0.30).

Usage: python training/measure/phantom_probe.py > training/data/system_test/phantom_probe.txt
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
LE2I = os.path.join(os.path.dirname(ROOT), 'datasets', 'le2i')
GUARD, PER_VIDEO = 15, 10
torch.set_num_threads(4)


def video_and_annotation(v):
    scene, name = v.split('/')
    n = name.split('_')[1]
    vid = next((p for p in (os.path.join(LE2I, scene, 'Videos', 'video (%s).avi' % n),
                            os.path.join(LE2I, scene, 'video (%s).avi' % n)) if os.path.exists(p)), None)
    ann = next((p for p in (os.path.join(LE2I, scene, d, 'video (%s).txt' % n)
                            for d in ('Annotation_files', 'Annotations_files')) if os.path.exists(p)), None)
    return vid, ann


def empty_frames(ann):
    rows = [l.strip().split(',') for l in open(ann) if l.count(',') == 5]
    person = {int(r[0]) for r in rows if any(int(x) for x in r[2:6])}
    empty = sorted(int(r[0]) for r in rows if not any(int(x) for x in r[2:6]))
    ok = [f for f in empty if all(abs(f - p) >= GUARD for p in person)]
    if len(ok) > PER_VIDEO:
        ok = [ok[i] for i in np.linspace(0, len(ok) - 1, PER_VIDEO).astype(int)]
    return ok


def main():
    m = json.load(open('training/data/le2i_holdout_manifest.json'))
    videos = m['half1_final_test']['videos']
    reserved = set(m.get('half2_reserved', {}).get('videos', []))
    assert not reserved & set(videos), 'half 2 video in the half-1 list'
    models = {'stock': YOLO('models/yolo26s-pose.pt'), 'A': YOLO('models/pose_nightaug_s44.pt')}
    tally = {(mn, sz): [0, 0, 0.0] for mn in models for sz in (320, 960)}   # frames with a person, frames, max conf
    per_scene, no_ann, used = {}, 0, 0
    for v in videos:
        vid, ann = video_and_annotation(v)
        if vid is None or ann is None:
            no_ann += 1
            continue
        frames = empty_frames(ann)
        cap = cv2.VideoCapture(vid)
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)       # annotation frames are 1-based
            ok, img = cap.read()
            if not ok:
                raise SystemExit('cannot decode %s frame %d' % (vid, f))
            used += 1
            for mn, model in models.items():
                for sz in (320, 960):
                    r = model.predict(img, imgsz=sz, conf=0.30, classes=[0], device='cpu', verbose=False)[0]
                    confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
                    t = tally[(mn, sz)]
                    t[1] += 1
                    if len(confs):
                        t[0] += 1
                        t[2] = max(t[2], float(confs.max()))
                        key = (v.split('/')[0], mn, sz)
                        per_scene[key] = per_scene.get(key, 0) + 1
        cap.release()
    print('empty frames used: %d from %d half-1 videos (%d without annotation/video skipped)' % (used, len(videos) - no_ann, no_ann))
    for (mn, sz), (hit, n, mx) in tally.items():
        print('%-5s imgsz %4d: frames with a phantom person %3d/%d (%.1f%%), max conf %.2f' % (mn, sz, hit, n, 100.0 * hit / max(n, 1), mx))
    for k in sorted(per_scene):
        print('  %-16s %-5s %4d: %d frames' % (k[0], k[1], k[2], per_scene[k]))


if __name__ == '__main__':
    main()
