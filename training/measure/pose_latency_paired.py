# -*- coding: utf-8 -*-
"""Paired CPU latency of pose checkpoints: the same frames, models interleaved frame by frame so a
busy machine slows every model alike; 4 threads, imgsz 320, conf 0.3 (deployed full-frame pass).
Usage: python training/measure/pose_latency_paired.py name=path [name=path ...]"""
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

torch.set_num_threads(4)
cv2.setNumThreads(1)
models = [(a.split('=')[0], YOLO(a.split('=', 1)[1])) for a in sys.argv[1:]]
frames = []
for p in sorted(glob.glob('training/data/urfd/fall-0[1-6]-cam0.mp4')):
    cap = cv2.VideoCapture(p)
    for _ in range(12):
        ok, f = cap.read()
        if ok:
            frames.append(f[:, f.shape[1] // 2:])
for _, m in models:
    m.predict(frames[0], verbose=False, conf=0.3, classes=[0], device='cpu', imgsz=320)
t = {n: [] for n, _ in models}
for rep in range(2):
    for f in frames:
        for n, m in models:
            s = time.perf_counter(); m.predict(f, verbose=False, conf=0.3, classes=[0], device='cpu', imgsz=320)
            t[n].append(1000 * (time.perf_counter() - s))
for n, v in t.items():
    print('%-12s median %6.1f ms  p95 %6.1f ms  (n=%d)' % (n, np.median(v), np.percentile(v, 95), len(v)))
