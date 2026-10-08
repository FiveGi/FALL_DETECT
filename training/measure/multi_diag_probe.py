# -*- coding: utf-8 -*-
"""MULTI-DIAG-v2 probe (Codex design 2026-10-01, training/data/multi_diag/codex_design_final.md).

Why is the faller lost in a multi-person fall? Replays a few owner segments through the deployed
pipeline exactly as eval_incidents_cpu.segment_alerts does (8 fps slots, crop state reset per
segment, hip tracker, deployed classifier at its threshold), under several pose-pass arms, and
writes a per-frame trace of what each stage did:

  roi        the crop the pose pass actually ran on (None = whole frame)
  dets       every detection the pose model returned, BEFORE the NUM_POSES cap, whole-frame
             pixels + box confidence, ranked by confidence (rank < NUM_POSES were kept)
  tracks     per active track: id, seen this frame, hip centre, classifier probability, alert

Nothing here is a scoring run: three segments are a diagnosis, not an evaluation. Output goes to
its own directory, outside every automatic cache discovery.

Arms (Codex): B = deployed (crop 256, full every 8, 320, conf .30); A = crop off; C = crop off,
conf .15; R = crop off, 640 px.

Usage:
  V3_DEVICE=cpu python training/measure/multi_diag_probe.py --segments 1.mp4#15,9.mp4#20,12.mp4#10 \
      --arms B,A,C,R --phase 0 --out training/data/multi_diag_v2
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
for k, v in (('V3_IMGSZ', '320'), ('V3_ROI_IMGSZ', '256'), ('V3_ROI_FULL_EVERY', '8'),
             ('V3_POSE_CONF', '0.3'), ('V3_TRACKER', 'hip')):
    os.environ.setdefault(k, v)
spec = importlib.util.spec_from_file_location('v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

FPS = 8.0
INCIDENTS = os.path.join(ROOT, 'test_result', 'incidents', 'incidents.json')
ARMS = {  # roi_imgsz, imgsz, pose_conf
    'B': (256, 320, 0.30),
    'A': (0, 320, 0.30),
    'C': (0, 320, 0.15),
    'R': (0, 640, 0.30),
}


def sha(path):
    return hashlib.sha1(open(path, 'rb').read()).hexdigest()[:12]


class Spy:
    """Wraps the pose model and the crop helper to record what one frame actually did."""

    def __init__(self, det):
        self.det, self.frame = det, None
        real_predict, real_roi = det.pose_model.predict, v3._roi_from_boxes

        def predict(source, **kw):
            res = real_predict(source, **kw)
            self.frame['imgsz'] = kw.get('imgsz')
            r = res[0]
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                roi = self.frame['roi'] or (0, 0, 0, 0)
                order = np.argsort(-conf)
                self.frame['dets'] = [[round(float(xyxy[i][0] + roi[0]), 1), round(float(xyxy[i][1] + roi[1]), 1),
                                       round(float(xyxy[i][2] + roi[0]), 1), round(float(xyxy[i][3] + roi[1]), 1),
                                       round(float(conf[i]), 3)] for i in order]
            return res

        def roi_from_boxes(boxes, w, h):
            out = real_roi(boxes, w, h)
            self.frame['roi'] = list(out) if out is not None else None
            return out

        det.pose_model.predict = predict
        v3._roi_from_boxes = roi_from_boxes


def run_segment(det, spy, path, start_s, end_s, phase, threshold):
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * src))
    state = v3.V3MultiPersonFallState()
    det.reset_roi_state(phase)
    i, last_slot, frames, secs = int(start_s * src), -1, [], 0.0
    while i < int(end_s * src):
        ok, frame = cap.read()
        if not ok:
            break
        slot = int(i * FPS / src)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        spy.frame = {'t': round(i / src - start_s, 3), 'roi': None, 'imgsz': None, 'dets': []}
        t0 = time.perf_counter()
        res = v3.detect_v3_fall_multi(frame, state, det, config=None, threshold=threshold)
        secs += time.perf_counter() - t0
        seen = {tid for tid, st in state.tracker.tracks.items() if st['missed'] == 0}
        spy.frame['tracks'] = [[int(tid), tid in seen, [round(float(c[0]), 3), round(float(c[1]), 3)],
                                round(float(p), 3), bool(d)] for tid, d, p, _, c in res]
        spy.frame['shape'] = list(frame.shape[:2])
        frames.append(spy.frame)
    cap.release()
    return frames, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments', required=True)
    ap.add_argument('--arms', default='B,A,C,R')
    ap.add_argument('--phase', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = json.load(open(INCIDENTS, encoding='utf-8'))
    rows = {'%s#%d' % (c, r['segment']): r for c, rs in data['clips'].items() for r in rs}
    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    spy = Spy(det)
    threshold = v3.THRESHOLD
    manifest = {'design': 'training/data/multi_diag/codex_design_final.md', 'segments': a.segments.split(','),
                'phase': a.phase, 'fps': FPS, 'threshold': threshold, 'num_poses': v3.NUM_POSES,
                'tracker': v3.TRACKER, 'device': os.environ.get('V3_DEVICE'), 'roi_full_every': v3.ROI_FULL_EVERY,
                'roi_pad': v3.ROI_PAD, 'preprocess': list(v3.PREPROCESS), 'dark_below': v3.PREPROCESS_DARK_BELOW,
                'grey_below': v3.PREPROCESS_GREY_BELOW, 'max_track_distance': v3.MAX_TRACK_DISTANCE,
                'max_missed_frames': v3.MAX_MISSED_FRAMES, 'arms': {k: ARMS[k] for k in a.arms.split(',')},
                'sha': {'v3_fall_detection.py': sha('app/detection/v3_fall_detection.py'),
                        'pose': sha(os.path.join('models', 'yolo26s-pose.pt')),
                        'classifier': sha(os.path.join('models', 'fall_classifier_v3.onnx'))
                        if os.path.exists(os.path.join('models', 'fall_classifier_v3.onnx')) else None},
                'runs': {}}
    for arm in a.arms.split(','):
        v3.ROI_IMGSZ, v3.IMGSZ, v3.POSE_CONF = ARMS[arm]
        for seg in a.segments.split(','):
            r = rows[seg]
            frames, secs = run_segment(det, spy, os.path.join('Test', seg.split('#')[0]), r['start_s'], r['end_s'],
                                       a.phase, threshold)
            name = '%s_%s_p%d.json' % (arm, seg.replace('.mp4#', '_'), a.phase)
            json.dump({'segment': seg, 'arm': arm, 'frames': frames}, open(os.path.join(a.out, name), 'w'))
            alerts = [f['t'] for f in frames if any(t[4] for t in f['tracks'])]
            peak = max([0.0] + [t[3] for f in frames for t in f['tracks']])
            manifest['runs'][name] = {'alerts_at': alerts, 'peak': round(peak, 3), 'frames': len(frames),
                                      'ms_per_frame': round(1000 * secs / max(len(frames), 1), 1)}
            print('%s %-10s alerts %d peak %.2f  %.0f ms/frame' % (arm, seg, len(alerts), peak,
                  manifest['runs'][name]['ms_per_frame']), flush=True)
    json.dump(manifest, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
