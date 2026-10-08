# -*- coding: utf-8 -*-
"""OmniFall staged datasets (Le2i first) -> pose files, the same way OF-Syn was extracted.

Video comes from the original dataset; labels from OmniFall's unified frame segments
(`labels/<dataset>.csv`; "fall" and "fallen" = 1, as for OF-ItW / OF-Syn). Poses from the deployed
detector at the CPU profile (320 px, crop 256 every 8, conf 0.30), crop state reset per video,
frames sampled at 8 fps by source time with the camera loop's slot rule, so the file needs no
resampling (it stores fps = 8). One person followed per video (extract_omnifall_syn.follow).

Group for the train/val split = the scene (Le2i has no subject ids), stored as `subject`, so a room
never straddles train and validation (Codex DATA-DESIGN #2).

Le2i: UBFC doi:10.25666/DATAUBFC-2024-04-09, CC BY-NC-SA 3.0. 190 videos, 25 fps, 320x240.

Usage: V3_DEVICE=cuda DATASET=le2i python training/extract_omnifall_staged.py
"""
import glob
import importlib.util
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for k, v in (('V3_IMGSZ', '320'), ('V3_ROI_IMGSZ', '256'), ('V3_ROI_FULL_EVERY', '8'), ('V3_POSE_CONF', '0.3')):
    os.environ.setdefault(k, v)
DATASET = os.environ.get('DATASET', 'le2i')
LABELS = 'D:/project/PROJECT/datasets/omnifall_labels/labels/%s.csv' % DATASET
# TARGET_FPS: 8 matches the camera loop; 15 matches recipe A (~15 fps by stride, AI_HANDOFF.md
# 2026-10-01), whose stride cannot be applied to files already at 8 fps. Non-default rates get their
# own directory so the two never mix.
FPS = float(os.environ.get('TARGET_FPS', 8))
FALL_LABEL_IDS = {1, 2}
OUT = os.path.join(ROOT, 'training', 'data', 'poses_%s%s' % (DATASET, '' if FPS == 8 else '_%dfps' % FPS))

spec = importlib.util.spec_from_file_location('v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)


def le2i_videos():
    """OmniFall path 'Coffee_room_01/video_14' -> 'Coffee_room_01/Videos/video (14).avi'."""
    root = 'D:/project/PROJECT/datasets/le2i'
    out = {}
    for f in glob.glob(os.path.join(root, '**', '*.avi'), recursive=True):
        rel = os.path.relpath(f, root).replace(os.sep, '/')
        scene = rel.split('/')[0].replace(' ', '_')
        out['%s/video_%s' % (scene, re.search(r'\((\d+)\)', rel).group(1))] = f
    return out


VIDEOS = {'le2i': le2i_videos}


def follow(per_frame):
    # Same rule as extract_omnifall_syn.follow (kept local: importing that module would build a
    # second detector at import time).
    out = np.zeros((len(per_frame), 17, 3), dtype=np.float32)
    prev = None
    for i, people in enumerate(per_frame):
        if not people:
            continue
        k, hip = people[0] if prev is None else min(
            people, key=lambda p: float(np.linalg.norm(np.asarray(p[1]) - prev)))
        out[i] = k
        prev = np.asarray(hip)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    segs = {p: g[['label', 'start', 'end']].values for p, g in lab.groupby('path')}
    videos = VIDEOS[DATASET]()
    missing = [p for p in segs if p not in videos]
    if missing:
        raise SystemExit('labels without a video: %d, e.g. %s' % (len(missing), missing[:3]))
    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    print('dataset', DATASET, 'videos', len(segs), 'roi', v3.ROI_IMGSZ, 'conf', v3.POSE_CONF, flush=True)
    for n, p in enumerate(sorted(segs)):
        npz = os.path.join(OUT, '%s.npz' % p.replace('/', '__'))
        if os.path.exists(npz):
            continue
        cap = cv2.VideoCapture(videos[p])
        src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        det.reset_roi_state(0)
        people, t, i, last = [], [], 0, -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            slot = int(i * FPS / src)
            i += 1
            if slot == last:
                continue
            last = slot
            people.append(det.extract_all_keypoints(frame))
            t.append((i - 1) / src)
        cap.release()
        if len(people) < 4:
            print('SKIP', p, len(people), flush=True)
            continue
        t = np.asarray(t)
        fl = np.zeros(len(t), dtype=np.int64)
        for l, a, b in segs[p]:
            if int(l) in FALL_LABEL_IDS:
                fl[(t >= a) & (t < b)] = 1
        np.savez_compressed(npz, keypoints=follow(people), frame_labels=fl, label=int(fl.max()),
                            subject='%s_%s' % (DATASET, p.split('/')[0]), fps=FPS, source_fps=src)
        if n % 20 == 0:
            print(n, p, len(t), 'frames', int(fl.sum()), 'fall', flush=True)
    print('done', OUT)


if __name__ == '__main__':
    sys.exit(main())
