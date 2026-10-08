# -*- coding: utf-8 -*-
"""Can a bigger pose model see a person lying on the floor? (FALLPOSE-v1 teacher check)

Same metric as fall_presence.py (URFD cam0 dataset labels, 8 fps samples, descent and the 2 s
after), but running each candidate model directly on full frames on the GPU -- a dev-only teacher
check, so no crop and no CPU profile. URFD half A only (choices may be made on A; B is the gate).
Usage: python training/measure/teacher_presence.py model@imgsz [model@imgsz ...]
"""
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, 'training')
sys.path.insert(0, 'training/measure')
from urfd_split import half  # noqa: E402
from fall_presence import spans  # noqa: E402


def frames(path, t_from, t_to):
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, last, out = 0, -1, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        slot = int(i * 8 / src)
        i += 1
        if slot == last:
            continue
        last = slot
        if t_from <= slot / 8.0 < t_to:
            # Codex review P2: URFD files are depth|RGB side by side -- keep the RGB half, as
            # cache_pose_streams does, and keep each sample's own timestamp.
            out.append((slot / 8.0, f[:, f.shape[1] // 2:]))
    return out


if __name__ == '__main__':
    sp = {s: v for s, v in spans().items() if half(s + '-cam0.mp4') == 'A'}
    for arg in sys.argv[1:]:
        name, imgsz = arg.split('@')
        m = YOLO(name)
        d_all, a_all = [], []
        for seq, (t0, t1, t2) in sorted(sp.items()):
            samples = frames('training/data/urfd/%s-cam0.mp4' % seq, t0, t2)
            ts = [t for t, _ in samples]
            hit = [len(m.predict(f, verbose=False, conf=0.3, classes=[0], imgsz=int(imgsz))[0].boxes) > 0 for _, f in samples]
            d = [h for h, t in zip(hit, ts) if t < t1]
            a = [h for h, t in zip(hit, ts) if t >= t1]
            d_all.append(np.mean(d) if d else np.nan)
            a_all.append(np.mean(a) if a else np.nan)
        print('%-24s half A n=%d  descent %.3f  after %.3f' % (arg, len(sp), np.nanmean(d_all), np.nanmean(a_all)), flush=True)
