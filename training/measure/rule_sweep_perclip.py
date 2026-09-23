# -*- coding: utf-8 -*-
"""Per-clip outcomes for one (model, alerting rule, frame rate), written to JSON.

Choosing an operating point by reading URFD scores would burn URFD as an independent measure --
the same mistake that cost a day already. So this records every clip individually and the
analysis step splits URFD in half by clip index: even-numbered clips are the half a rule may
be chosen on, odd-numbered ones are only ever used to confirm the choice afterwards.

One process per configuration, because the window size and alerting rule are read at import.

Usage:
    V3_DEVICE=cuda V3_WINDOW_SIZE=15 TEST_MODEL_DIR=... V3_SMOOTH_NEED=2 \\
        TARGET_FPS=18 OUT=... python rule_sweep_perclip.py
"""
import glob
import importlib.util
import json
import os
import sys

import cv2

ROOT = r'D:\project\PROJECT\Backend-Elderly-Surveillance-main'
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'training'))
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
from eval_v3_frame_drop import TRAIN_ADL  # noqa: E402

MODEL_DIR = os.environ.get('TEST_MODEL_DIR', os.path.join(ROOT, 'models'))
FPS = float(os.environ.get('TARGET_FPS', 18))
OUT = os.environ['OUT']


def alerted(det, path, rgb_half=False):
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    st = v3.V3MultiPersonFallState()
    i, last, count, last_slot = 0, None, 0, -1
    while True:
        ok, f = cap.read()
        if not ok:
            break
        slot = int(i * FPS / src)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        if rgb_half:
            f = f[:, f.shape[1] // 2:]
        hit = any(r[1] for r in v3.detect_v3_fall_multi(f, st, det, config=None))
        label = 'fall' if hit else 'no_fall'
        if label != last:
            count += 1 if hit else 0
            last = label
    cap.release()
    return count > 0


det = v3.V3PoseFallDetector(model_dir=MODEL_DIR)
meta = {'model': MODEL_DIR, 'window': v3.WINDOW_SIZE, 'need': v3.SMOOTH_NEED,
        'of': v3.SMOOTH_OF, 'threshold': v3.THRESHOLD, 'fps': FPS}

# Written after every clip and resumed from. A full sweep is 220 clips; writing only at the
# end has already lost two long runs to the machine going down mid-sweep.
rows = []
if os.path.exists(OUT):
    prev = json.load(open(OUT))
    if prev.get('meta') == meta:
        rows = prev['rows']
        print('resuming: %d clips already measured' % len(rows), flush=True)
    else:
        raise SystemExit('%s holds a different configuration: %r' % (OUT, prev.get('meta')))
done = {(r[0], r[1]) for r in rows}

groups = [
    ('urfd_fall', sorted(glob.glob('training/data/urfd/fall-*.mp4')), True),
    ('urfd_adl', sorted(glob.glob('training/data/urfd/adl-*.mp4')), True),
    ('gmdcsa_fall', sorted(glob.glob('training/data/gmdcsa24_fall_raw/*.mp4')), False),
    ('val_adl', sorted(glob.glob('training/data/gmdcsa24_adl_raw_val/*.mp4')), False),
    ('train50_adl', [q for q in (os.path.join('training/data/gmdcsa24_adl_raw_train50', n + '.mp4')
                                 for n in TRAIN_ADL) if os.path.exists(q)], False),
]
for group, paths, rgb_half in groups:
    todo = [q for q in paths if (group, os.path.basename(q)) not in done]
    for q in todo:
        rows.append([group, os.path.basename(q), alerted(det, q, rgb_half)])
        json.dump({'meta': meta, 'rows': rows}, open(OUT, 'w'), indent=1)
    print('  %-14s %d clips (%d measured now)' % (group, len(paths), len(todo)), flush=True)

json.dump({'meta': meta, 'rows': rows}, open(OUT, 'w'), indent=1)
print('wrote', OUT, meta)
