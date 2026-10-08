# -*- coding: utf-8 -*-
"""PLAN step 2 (agreed 2026-10-04): FULL-pipeline CPU processing time per frame at the deployed profile
(320 px, crop 256 every 8, conf 0.30, 8 fps, hip tracker, NUM_POSES=4), CPU_THREADS=4, for several
configurations on the SAME frames, interleaved segment by segment so background load hits all alike.
Frames: owner segments with 1-4 people (from the stock owner cache counts), decoded once.
Reports per config: median / p95 ms per frame, implied sustained fps, and ratio to the first config.

Usage: python training/measure/pipeline_timing.py name=POSE_MODEL[:ENSEMBLE_DIRS][@ROI_IMGSZ] ...
  (@ROI_IMGSZ overrides the crop size for that config only, e.g. crop320=-@320; plan round 3 D7)
  e.g. deployed=- s44=training/data/pose_ir/nightaug_s44_p2/weights/best.pt
       s44_ens3=training/data/pose_ir/nightaug_s44_p2/weights/best.pt:dirA,dirB
"""
import glob
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)   # Codex P1: app.services.cpu_tuning must import, or thread pinning silently falls back
for k, v in (('V3_IMGSZ', '320'), ('V3_ROI_IMGSZ', '256'), ('V3_ROI_FULL_EVERY', '8'), ('V3_POSE_CONF', '0.3'),
             ('V3_DEVICE', 'cpu'), ('CPU_THREADS', '4'), ('OMP_NUM_THREADS', '4')):
    os.environ[k] = v
import cv2  # noqa: E402
import numpy as np  # noqa: E402


def load_v3(pose, ens, roi='256'):
    os.environ['V3_ROI_IMGSZ'] = roi   # read when the module executes below
    # Codex P1: a relative path would be resolved under models/; pass an absolute one
    os.environ['V3_POSE_MODEL'] = os.path.abspath(pose) if pose != '-' else 'yolo26s-pose.pt'
    os.environ['V3_ENSEMBLE'] = ens
    spec = importlib.util.spec_from_file_location('v3_%d' % len(sys.modules), 'app/detection/v3_fall_detection.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, m.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))


def segments(max_n=12):
    inc = json.load(open('test_result/incidents/incidents.json', encoding='utf-8'))
    C = 'training/data/pose_cache_owner/23de875c4696'
    out = []
    for clip, rows in inc['clips'].items():
        for r in rows:
            f = os.path.join(C, '%s#%d.npz' % (clip, r['segment']))
            if not os.path.exists(f):
                continue
            c = np.load(f)['counts']
            if len(c) and 1 <= np.median(c) and c.max() <= 4 and c.mean() >= 1.3:
                out.append((clip, r['start_s'], r['end_s'], float(c.mean())))
    out.sort(key=lambda x: -x[3])
    return out[:max_n]


def frames(clip, a, b):
    cap = cv2.VideoCapture(os.path.join('Test', clip)); src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a * src)); i, last, fr = int(a * src), -1, []
    while i < int(b * src):
        ok, f = cap.read()
        if not ok:
            break
        slot = int(i * 8 / src); i += 1
        if slot != last:
            last = slot; fr.append(f)
    return fr


if __name__ == '__main__':
    cfgs = []
    for a in sys.argv[1:]:
        name, spec = a.split('=', 1)
        spec, _, roi = spec.partition('@')
        pose, _, ens = spec.partition(':')
        ens = ','.join(os.path.join(ROOT, d, 'fall_classifier_v3.onnx') for d in ens.split(',') if d)
        cfgs.append((name,) + load_v3(pose, ens, roi or '256'))
        assert cfgs[-1][1].ROI_IMGSZ == int(roi or 256), 'crop size not applied'
    import torch
    print('torch threads', torch.get_num_threads(), flush=True)
    assert torch.get_num_threads() == 4, 'thread pinning not applied'
    segs = segments(int(os.environ.get('TIMING_SEGS', 12)))
    warm = frames(*segs[0][:3])[:24]          # Codex P2: untimed warm-up per config
    for name, m, det in cfgs:
        st = m.V3MultiPersonFallState(); det.reset_roi_state(0)
        for f in warm:
            m.detect_v3_fall_multi(f, st, det, config=None)
    print('segments', len(segs), 'mean people', round(np.mean([s[3] for s in segs]), 2), flush=True)
    times = {c[0]: [] for c in cfgs}
    for clip, a, b, _ in segs:
        fr = frames(clip, a, b)
        for name, m, det in cfgs:
            st = m.V3MultiPersonFallState(); det.reset_roi_state(0)
            for k, f in enumerate(fr):
                t = time.perf_counter(); m.detect_v3_fall_multi(f, st, det, config=None)
                # Codex P2: the first window of each segment (classifier not yet running) is excluded
                if k >= m.WINDOW_SIZE:
                    times[name].append(1000 * (time.perf_counter() - t))
    base = None
    for name, _, _ in cfgs:
        v = np.array(times[name]); med, p95 = np.median(v), np.percentile(v, 95)
        base = base or (med, p95)
        print('%-12s n=%d median %.1f ms p95 %.1f ms -> %.1f fps sustained; vs first %.0f%% / %.0f%%' % (
            name, len(v), med, p95, 1000 / np.mean(v), 100 * med / base[0], 100 * p95 / base[1]), flush=True)
