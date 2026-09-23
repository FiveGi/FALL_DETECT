# -*- coding: utf-8 -*-
"""Which pose model loses the person more often, at the input size each would deploy at?

The keypoints the two produce agree closely (3% of a torso length), so the train/serve mismatch
that was supposed to handicap yolo26n-pose is not the explanation for it catching fewer falls.
The frames where they disagree are the other kind: one found a person and the other did not,
which is not a slightly different elbow -- it is a zero row in the window, and enough of them
stops the classifier scoring at all.

So this measures the thing that is left. Each model at the input size it would run at on four
CPU cores for the same cost per frame (s at 320, n at 384), over whole evaluation clips rather
than the first few seconds, counting how often each finds anybody. Falls matter most on the
frames after the person is on the ground, which is exactly where pose models fail, so the last
third is reported separately.
"""
import os
import sys

import cv2
import numpy as np

ROOT = r'D:\project\PROJECT\Backend-Elderly-Surveillance-main'
os.chdir(ROOT)
from ultralytics import YOLO  # noqa: E402

CONF = 0.3
CONFIGS = [('yolo26s-pose @ 320', 'yolo26s-pose.pt', 320),
           ('yolo26n-pose @ 384', 'yolo26n-pose.pt', 384)]
GROUPS = [
    ('URFD falls', sorted(__import__('glob').glob('training/data/urfd/fall-*.mp4'))[:24], True),
    ('GMDCSA24 falls', sorted(__import__('glob').glob('training/data/gmdcsa24_fall_raw/*.mp4'))[:20], False),
]


def found_rates(model, imgsz, path, rgb_half):
    cap = cv2.VideoCapture(path)
    flags = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if rgb_half:
            f = f[:, f.shape[1] // 2:]
        r = model.predict(f, verbose=False, conf=CONF, classes=[0], device='cuda',
                          imgsz=imgsz)[0]
        flags.append(1 if (r.keypoints is not None and len(r.keypoints.xy)) else 0)
    cap.release()
    if not flags:
        return None, None
    tail = flags[2 * len(flags) // 3:]
    return float(np.mean(flags)), float(np.mean(tail) if tail else 0.0)


print('person found, whole clip and the last third (after the fall)\n')
print('%-22s %-28s %s' % ('', GROUPS[0][0], GROUPS[1][0]))
rows = {}
for label, weights, imgsz in CONFIGS:
    model = YOLO(os.path.join(ROOT, 'models', weights))
    cells = []
    for gname, paths, rgb_half in GROUPS:
        whole, tail = [], []
        for p in paths:
            w, t = found_rates(model, imgsz, p, rgb_half)
            if w is not None:
                whole.append(w)
                tail.append(t)
        cells.append('%.0f%% whole, %.0f%% last third' % (100 * np.mean(whole), 100 * np.mean(tail)))
        rows[(label, gname)] = (np.mean(whole), np.mean(tail))
    print('%-22s %-28s %s' % (label, cells[0], cells[1]))

print()
for gname, _, _ in GROUPS:
    s = rows[(CONFIGS[0][0], gname)]
    n = rows[(CONFIGS[1][0], gname)]
    print('%-16s yolo26n finds the person %+.1f points less of the time overall, '
          '%+.1f in the last third'
          % (gname, 100 * (n[0] - s[0]), 100 * (n[1] - s[1])))
