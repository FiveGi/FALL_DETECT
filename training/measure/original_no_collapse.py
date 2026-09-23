# -*- coding: utf-8 -*-
"""How much of the original detector's URFD recall is the classifier, and how much is the
"the person vanished" rule?

Fed every frame of the file, the original detector catches 54/60 URFD falls -- 90%, well above
anything deployed since. That number decides how this whole comparison reads, so it is worth
knowing what produced it.

The original has a rule the current pipeline kept but made optional: if the classifier was
above COLLAPSE_CONFIDENCE and person detection then drops to ~zero, the disappearance itself is
reported as a fall, with probability exactly 1.0. MediaPipe loses people once they are prone,
and a URFD fall clip ends within a second or two of the person hitting the floor -- so "the
subject vanished at the end of the clip" describes nearly every fall clip in the set, and would
also describe a person walking out of shot.

This is a straight A/B: the same detector, the same frames, with the collapse rule disabled by
raising its threshold above any reachable probability. The difference is what the rule was
worth, on falls and on the normal clips alike.
"""
import glob
import importlib.util
import json
import os
import sys

import cv2

ROOT = r'D:\project\PROJECT\Backend-Elderly-Surveillance-main'
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'training'))
from eval_v3_frame_drop import TRAIN_ADL  # noqa: E402

TARGET_FPS = float(os.environ.get('TARGET_FPS', 30))
OUT = os.path.join(HERE, 'original_no_collapse.json')

spec = importlib.util.spec_from_file_location('v3_orig', os.path.join(HERE, 'v3_original.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

# Above any probability a sigmoid can return, so the branch can never be taken. The rest of the
# detector -- window, threshold, smoothing, pose extractor -- is untouched.
v3.COLLAPSE_CONFIDENCE = 2.0

det = v3.V3PoseFallDetector(model_dir=os.path.join(HERE, 'model_dir'))
print('ORIGINAL with the collapse rule disabled, fed at %.0f fps' % TARGET_FPS, flush=True)


def alerted(path, rgb_half=False):
    state = v3.V3FallDetectionState()
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, last, count, last_slot = 0, None, 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        slot = int(i * TARGET_FPS / src)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        if rgb_half:
            frame = frame[:, frame.shape[1] // 2:]
        _, _, label, _ = v3.detect_v3_fall(frame, state, det, config=None)
        if label != last:
            count += 1 if label == 'fall' else 0
            last = label
    cap.release()
    return count > 0


META = {'model': 'ORIGINAL ce401fa, collapse rule OFF', 'window': v3.WINDOW_SIZE,
        'threshold': v3.THRESHOLD, 'need': v3.SMOOTH_NEED, 'of': v3.SMOOTH_OF,
        'pose': 'mediapipe, no collapse rule', 'fps': TARGET_FPS}

rows = []
if os.path.exists(OUT):
    prev = json.load(open(OUT))
    if prev.get('meta') == META:
        rows = prev['rows']
        print('resuming: %d clips already measured' % len(rows), flush=True)
done = {(r[0], r[1]) for r in rows}
groups = [
    ('urfd_fall', sorted(glob.glob('training/data/urfd/fall-*.mp4')), True),
    ('urfd_adl', sorted(glob.glob('training/data/urfd/adl-*.mp4')), True),
    ('gmdcsa_fall', sorted(glob.glob('training/data/gmdcsa24_fall_raw/*.mp4')), False),
    ('val_adl', sorted(glob.glob('training/data/gmdcsa24_adl_raw_val/*.mp4')), False),
    ('train50_adl', [p for p in (os.path.join('training/data/gmdcsa24_adl_raw_train50', n + '.mp4')
                                 for n in TRAIN_ADL) if os.path.exists(p)], False),
]
for group, paths, rgb_half in groups:
    todo = [p for p in paths if (group, os.path.basename(p)) not in done]
    for p in todo:
        rows.append([group, os.path.basename(p), alerted(p, rgb_half)])
        json.dump({'meta': META, 'rows': rows}, open(OUT, 'w'), indent=1)
    print('  %-14s %d clips (%d measured now)' % (group, len(paths), len(todo)), flush=True)
print('wrote', OUT)
