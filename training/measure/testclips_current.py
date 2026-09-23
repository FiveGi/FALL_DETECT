# -*- coding: utf-8 -*-
"""The Test/ clips through the two deployed profiles as they stand today.

The stored `all17_results.json` predates partial-window scoring and the CPU profile, so the
comparison workbook needs these measured fresh rather than reused.
"""
import importlib.util, json, os
import cv2

ROOT = r'D:\project\PROJECT\Backend-Elderly-Surveillance-main'
os.chdir(ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = ['Test/%d.mp4' % n for n in range(1, 18)]
OUT = os.path.join(HERE, 'testclips_current.json')

CONFIGS = [
    ('deployed GPU  960 @ 20fps, partial 4', 960, 20, 4),
    ('deployed CPU  320 @ 8fps,  partial 4', 320, 8, 4),
]


def load(imgsz, partial):
    os.environ['V3_DEVICE'] = 'cuda'
    os.environ['V3_IMGSZ'] = str(imgsz)
    os.environ['V3_PARTIAL_MIN'] = str(partial)
    spec = importlib.util.spec_from_file_location(
        'v3_%d_%d' % (imgsz, partial), os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, m.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))


def run(mod, det, clip, fps):
    st = mod.V3MultiPersonFallState()
    cap = cv2.VideoCapture(clip)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, last_slot, peak, last, alerts = 0, -1, 0.0, None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        slot = int(i * fps / src)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        res = mod.detect_v3_fall_multi(f, st, det, config=None)
        for r in res:
            peak = max(peak, float(r[2]))
        label = 'fall' if any(r[1] for r in res) else 'no_fall'
        if label != last:
            alerts += 1 if label == 'fall' else 0
            last = label
    cap.release()
    return alerts, peak


results = json.load(open(OUT)) if os.path.exists(OUT) else {}
for label, imgsz, fps, partial in CONFIGS:
    if label in results and len(results[label]) == len(CLIPS):
        continue
    mod, det = load(imgsz, partial)
    row = results.get(label, {})
    for clip in CLIPS:
        if clip in row:
            continue
        row[clip] = run(mod, det, clip, fps)
        results[label] = row
        json.dump(results, open(OUT, 'w'), indent=1)
        print('  %-38s %-14s alerts=%2d peak=%.2f' % (label, clip, row[clip][0], row[clip][1]),
              flush=True)
print('wrote', OUT)
