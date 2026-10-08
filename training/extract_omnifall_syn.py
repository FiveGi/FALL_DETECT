# -*- coding: utf-8 -*-
"""OmniFall OF-Syn (12,000 synthetic 5 s clips, 16 fps) -> pose files for training.

Design (AI_HANDOFF.md, Codex DATA-DESIGN 2026-10-01):
  - Poses come from the DEPLOYED detector at the CPU profile -- V3PoseFallDetector with
    V3_IMGSZ=320, V3_ROI_IMGSZ=256, V3_ROI_FULL_EVERY=8, V3_POSE_CONF=0.30 -- so training
    keypoints have the noise, crop behaviour and missed detections the camera loop has.
    Crop state is reset per clip (reset_roi_state); nothing carries between videos.
  - 16 fps -> 8 fps by taking alternate frames; BOTH offsets are kept, as two files of the same
    clip (`..._o0`, `..._o1`), and share the clip's split group so they never straddle a split.
  - Labels: OmniFall's 16-class frame segments; "fall" and "fallen" (ids 1, 2) -> 1, as
    extract_ofitw_poses*.py already does. Stored per frame, so windows are labelled at their
    centre, and the file carries `fps` = 8 so dataset.RESAMPLE_FPS needs no lookup.
  - One person per clip (each OF-Syn video shows one person). The person followed is the most
    confident detection on the first frame that has one, then the detection nearest the previous
    hip. A frame with nobody gets zeros (confidence 0), as yolopose_extractor does.

Licence: OF-Syn videos and labels, CC BY-NC-SA 4.0 (non-commercial).

Usage:  V3_DEVICE=cuda SHARD=0 SHARDS=3 python training/extract_omnifall_syn.py
"""
import importlib.util
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('V3_IMGSZ', '320')
os.environ.setdefault('V3_ROI_IMGSZ', '256')
os.environ.setdefault('V3_ROI_FULL_EVERY', '8')
os.environ.setdefault('V3_POSE_CONF', '0.3')
SYN = os.path.join(ROOT, 'training', 'data', 'omnifall_syn')
OUT = os.environ.get('OUT_DIR') or os.path.join(ROOT, 'training', 'data', 'poses_omnifall_syn')
# CLIP_LIST: a file of OF-Syn paths (one per line) to extract instead of all 12,000 -- used to
# re-extract a subset with another pose checkpoint (V3_POSE_MODEL) into its own OUT_DIR.
CLIP_LIST = os.environ.get('CLIP_LIST')
FALL_LABEL_IDS = {1, 2}
SHARD, SHARDS = int(os.environ.get('SHARD', 0)), int(os.environ.get('SHARDS', 1))
# Extraction order and offsets: the GPU is the bottleneck (~81 pose calls per clip), so the clips
# most like the deployment (older people first) are extracted first, and the second 8 fps offset
# can be deferred with OFFSETS=0. Toddlers and children come last: the audit rejected them for an
# elderly-care camera, and they are kept extractable only so that decision stays revisitable.
AGE_ORDER = ['elderly_65_plus', 'middle_aged_35_64', 'young_adults_18_34', 'teenagers_13_17',
             'children_5_12', 'toddlers_1_4']
OFFSETS = [int(o) for o in os.environ.get('OFFSETS', '0,1').split(',')]

spec = importlib.util.spec_from_file_location('v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)


def follow(per_frame):
    """per_frame: list of [(kpts17, hip)] -> (T,17,3), one person followed through the clip."""
    out = np.zeros((len(per_frame), 17, 3), dtype=np.float32)
    prev = None
    for i, people in enumerate(per_frame):
        if not people:
            continue
        if prev is None:
            k, hip = people[0]                       # most confident (extract_all_keypoints order)
        else:
            k, hip = min(people, key=lambda p: float(np.linalg.norm(np.asarray(p[1]) - prev)))
        out[i] = k
        prev = np.asarray(hip)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    labels = pd.read_csv(os.path.join(SYN, 'labels', 'of-syn.csv'))
    meta = labels.groupby('path').first()
    segs = {p: g[['label', 'start', 'end']].values for p, g in labels.groupby('path')}
    if CLIP_LIST:
        keep = {l.strip() for l in open(CLIP_LIST) if l.strip()}
        segs = {p: v for p, v in segs.items() if p in keep}
    paths = sorted(segs, key=lambda p: (AGE_ORDER.index(meta.loc[p, 'age_group']), p))[SHARD::SHARDS]
    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    print('roi', v3.ROI_IMGSZ, v3.ROI_FULL_EVERY, 'conf', v3.POSE_CONF, 'imgsz', v3.IMGSZ,
          'shard', SHARD, '/', SHARDS, 'clips', len(paths), flush=True)
    done = 0
    for p in paths:
        name = p.replace('/', '__')
        if all(os.path.exists(os.path.join(OUT, '%s_o%d.npz' % (name, o))) for o in OFFSETS):
            continue
        cap = cv2.VideoCapture(os.path.join(SYN, 'videos', p + '.mp4'))
        native = cap.get(cv2.CAP_PROP_FPS) or 16.0
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if len(frames) < 4:
            print('SKIP unreadable', p, len(frames), flush=True)
            continue
        m = meta.loc[p]
        for o in OFFSETS:
            if os.path.exists(os.path.join(OUT, '%s_o%d.npz' % (name, o))):
                continue
            idx = np.arange(o, len(frames), int(round(native / 8.0)))
            det.reset_roi_state(0)
            kp = follow([det.extract_all_keypoints(frames[i]) for i in idx])
            t = idx / native
            fl = np.zeros(len(idx), dtype=np.int64)
            for lab, a, b in segs[p]:
                if int(lab) in FALL_LABEL_IDS:
                    fl[(t >= a) & (t < b)] = 1
            np.savez_compressed(
                os.path.join(OUT, '%s_o%d.npz' % (name, o)), keypoints=kp, frame_labels=fl,
                label=int(fl.max()), subject='syn_' + name, fps=8.0, offset=o,
                age_group=str(m['age_group']), environment=str(m['environment_category']),
                camera_elevation=str(m['camera_elevation']), camera_distance=str(m['camera_distance']))
        done += 1
        if done % 200 == 0:
            print('shard', SHARD, 'done', done, '/', len(paths), flush=True)
    print('shard', SHARD, 'finished', done, flush=True)


if __name__ == '__main__':
    sys.exit(main())
