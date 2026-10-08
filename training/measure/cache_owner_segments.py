# -*- coding: utf-8 -*-
"""Pose streams of the owner's 126 compilation segments, cached once per crop phase.

eval_incidents_cpu.py runs the pose model on every segment for every configuration: ~1 h per
phase on one CPU thread. Stage 1 trains several classifiers that share one pose front-end
(320 px, crop 256 every 8, conf 0.30, 8 fps), so the poses are computed once here and each
classifier is replayed over them (replay_owner_segments.py) -- the same split as
cache_pose_streams.py / compare_caches.py for URFD.

Sampling is eval_incidents_cpu.segment_alerts's exactly: seek to the segment start, keep a frame
when its 8 fps slot changes, reset the crop state to ROI_PHASE at the segment start.

Usage: V3_DEVICE=cpu ROI_PHASE=0 python training/measure/cache_owner_segments.py
Writes training/data/pose_cache_owner/<key>/<clip>#<segment>.npz (+ key.json).
"""
import hashlib
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
for k, v in (('V3_IMGSZ', '320'), ('V3_ROI_IMGSZ', '256'), ('V3_ROI_FULL_EVERY', '8'),
             ('V3_POSE_CONF', '0.3')):
    os.environ.setdefault(k, v)
spec = importlib.util.spec_from_file_location('v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

FPS = float(os.environ.get('TARGET_FPS', 8))
ROI_PHASE = int(os.environ.get('ROI_PHASE', 0))
INCIDENTS = os.path.join(ROOT, 'test_result', 'incidents', 'incidents.json')


def key():
    k = {'imgsz': v3.IMGSZ, 'pose_conf': v3.POSE_CONF, 'fps': FPS, 'roi_imgsz': v3.ROI_IMGSZ,
         'roi_full_every': v3.ROI_FULL_EVERY, 'roi_phase': ROI_PHASE, 'roi_state': 'reset-per-segment',
         'device': os.environ.get('V3_DEVICE', 'auto'), 'preprocess': list(v3.PREPROCESS),
         'sampling': 'eval_incidents_cpu.segment_alerts'}
    # A different pose checkpoint is a different pose pass. Keyed only when not the stock model,
    # so every cache built before POSE-IR keeps its key (stage1_eval finds them by that key).
    pose = os.environ.get('V3_POSE_MODEL', 'yolo26s-pose.pt')
    if pose != 'yolo26s-pose.pt':
        k['pose_model'] = pose
    # Same rule for the people cap: a cache built at another V3_NUM_POSES holds different frames,
    # and without this it would land in -- and overwrite -- the pinned cap-4 cache directory.
    if v3.NUM_POSES != 4:
        k['num_poses'] = v3.NUM_POSES
    return k


def cache_dir():
    k = key()
    return os.path.join(ROOT, 'training', 'data', 'pose_cache_owner',
                        hashlib.sha1(json.dumps(k, sort_keys=True).encode()).hexdigest()[:12])


def segment_frames(path, start_s, end_s):
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * src))
    i, last_slot = int(start_s * src), -1
    while i < int(end_s * src):
        ok, frame = cap.read()
        if not ok:
            break
        slot = int(i * FPS / src)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        yield i, src, frame
    cap.release()


def main():
    out = cache_dir()
    os.makedirs(out, exist_ok=True)
    json.dump(key(), open(os.path.join(out, 'key.json'), 'w'), indent=1)
    print('cache', out, key(), flush=True)
    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    data = json.load(open(INCIDENTS, encoding='utf-8'))
    for clip, rows in data['clips'].items():
        for r in rows:
            npz = os.path.join(out, '%s#%d.npz' % (clip, r['segment']))
            if os.path.exists(npz):
                continue
            det.reset_roi_state(ROI_PHASE)
            counts, kpts, t_rel = [], [], []
            for i, src, frame in segment_frames(os.path.join('Test', clip), r['start_s'], r['end_s']):
                people = det.extract_all_keypoints(frame)
                counts.append(len(people))
                t_rel.append(round(i / src - r['start_s'], 2))   # the time eval_incidents reports
                for kp, _hip in people:
                    kpts.append(kp)
            np.savez_compressed(npz, counts=np.array(counts, dtype=np.int16),
                                kpts=np.array(kpts, dtype=np.float32).reshape(-1, 17, 3),
                                t=np.array(t_rel, dtype=np.float32))
        print('  %-8s %d segment(s)' % (clip, len(rows)), flush=True)
    open(os.path.join(out, 'done'), 'w').write('ok')
    print('done', out)


if __name__ == '__main__':
    sys.exit(main())
