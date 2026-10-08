# -*- coding: utf-8 -*-
"""Score the detector on the owner's real compilation footage, at the CPU server's settings.

The 126 incident segments in test_result/incidents/incidents.json carry `alerts_at` from
cut_compilations.py, which defaults to 20 fps -- the GPU profile. The production server has no
GPU. Re-running that script would re-cut the segments and overwrite every label, so this reads
the segment list and writes its own results beside it.

Each segment is one continuous shot, so the detector starts fresh for each. Frames are sampled
by SOURCE time at TARGET_FPS, as the fixed camera loop does.

Truth: `label` where a person has set one, else Gemini's proposal (`antigravity`). Proposals are
not ground truth, and the report says which each count rests on.

Usage (CPU profile, on the CPU):
    V3_DEVICE=cpu V3_IMGSZ=320 TARGET_FPS=8 python training/measure/eval_incidents_cpu.py
"""
import importlib.util
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

FPS = float(os.environ.get('TARGET_FPS', 8))
ROI_PHASE = int(os.environ.get('ROI_PHASE', 0))   # see V3PoseFallDetector.reset_roi_state
INCIDENTS = os.path.join(ROOT, 'test_result', 'incidents', 'incidents.json')
OUT = os.path.join(ROOT, 'test_result', 'incidents',
                   'alerts_%s_%dpx_%gfps%s.json' % (os.environ.get('V3_DEVICE', 'auto'),
                                                    v3.IMGSZ, FPS,
                                                    ('_roi%d' % v3.ROI_IMGSZ if v3.ROI_IMGSZ else '')
                                                    + ('_grey%g' % v3.PREPROCESS_GREY_BELOW
                                                       if v3.PREPROCESS_GREY_BELOW else '')
                                                    + ('_conf%g' % v3.POSE_CONF
                                                       if v3.POSE_CONF != 0.3 else '')
                                                    + ('_thr%g' % v3.THRESHOLD
                                                       if v3.THRESHOLD != 0.65 else '')
                                                    + ('_phase%d' % ROI_PHASE
                                                       if v3.ROI_IMGSZ else '')))


def segment_alerts(det, path, start_s, end_s):
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * src))
    state = v3.V3MultiPersonFallState()
    det.reset_roi_state(ROI_PHASE)   # each segment is its own source; no crop carried over
    i, last_slot, hits, peak = int(start_s * src), -1, [], 0.0
    while i < int(end_s * src):
        ok, frame = cap.read()
        if not ok:
            break
        slot = int(i * FPS / src)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        res = v3.detect_v3_fall_multi(frame, state, det, config=None)
        peak = max([peak] + [r[2] for r in res])
        if any(r[1] for r in res):
            hits.append(round(i / src - start_s, 2))
    cap.release()
    return hits, round(peak, 3)


def main():
    data = json.load(open(INCIDENTS, encoding='utf-8'))
    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    out = {'profile': {'device': os.environ.get('V3_DEVICE'), 'imgsz': v3.IMGSZ, 'fps': FPS,
                       'threshold': v3.THRESHOLD, 'roi_imgsz': v3.ROI_IMGSZ,
                       'roi_full_every': v3.ROI_FULL_EVERY, 'pose_conf': v3.POSE_CONF,
                       'roi_phase': ROI_PHASE if v3.ROI_IMGSZ else None,
                       'roi_state': 'reset-per-segment'}, 'segments': {}}
    for clip, rows in data['clips'].items():
        for r in rows:
            hits, peak = segment_alerts(det, os.path.join('Test', clip), r['start_s'], r['end_s'])
            out['segments']['%s#%d' % (clip, r['segment'])] = {'alerts_at': hits, 'peak': peak}
        print('  %-8s %d segment(s)' % (clip, len(rows)), flush=True)
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)

    # Score against truth: a person's label first, Gemini's proposal otherwise.
    tally = {'human': [0, 0, 0, 0], 'gemini': [0, 0, 0, 0]}   # caught, missed, false alarm, quiet
    missed = []
    for clip, rows in data['clips'].items():
        for r in rows:
            g = r.get('antigravity') or {}
            if r.get('label'):
                src, ood, fall = 'human', r['label'] == 'out_of_domain', r['label'] == 'fall'
            elif g and not g.get('error'):
                src, ood, fall = 'gemini', bool(g.get('out_of_domain')), bool(g.get('fall'))
            else:
                continue
            if ood:
                continue
            alert = bool(out['segments']['%s#%d' % (clip, r['segment'])]['alerts_at'])
            k = 0 if fall and alert else 1 if fall else 2 if alert else 3
            tally[src][k] += 1
            if k == 1:
                missed.append('%s#%d %s' % (clip, r['segment'], (g.get('description') or '')[:60]))
    print()
    print('profile: %s' % out['profile'])
    print('in-domain segments   caught  missed  false-alarm  quiet')
    for src in ('human', 'gemini'):
        print('  truth=%-7s      %5d  %6d  %11d  %5d' % ((src,) + tuple(tally[src])))
    c = [tally['human'][i] + tally['gemini'][i] for i in range(4)]
    print('  combined           %5d  %6d  %11d  %5d   recall %.0f%%'
          % (c[0], c[1], c[2], c[3], 100.0 * c[0] / max(c[0] + c[1], 1)))
    print('\nmissed falls:')
    for m in missed:
        print('  ' + m)
    print('\nwrote', OUT)


if __name__ == '__main__':
    sys.exit(main())
