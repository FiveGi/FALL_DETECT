# -*- coding: utf-8 -*-
"""NIGHT-NOISE-v1 CPU cost screen: added milliseconds per frame for each preprocessing arm, on
real frames at the deployed decode size, ONE thread (cv2.setNumThreads(1)), gate forced open.
Indicative only when other jobs run on the machine: report it with that caveat and re-run idle.
Usage: python training/measure/preprocess_cost.py"""
import glob
import importlib.util
import os
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
cv2.setNumThreads(1)
ARMS = {'B': 'auto', 'V': 'vflat,auto', 'U': 'auto,unsharp', 'D1': 'bilateral,auto',
        'D2': 'guided,auto', 'D3': 'nlmeans,auto', 'G': 'auto'}


def load(pre):
    os.environ['V3_PREPROCESS'] = pre
    os.environ['V3_PREPROCESS_DARK_BELOW'] = '999'      # force the gate open: cost when it fires
    spec = importlib.util.spec_from_file_location('v3', 'app/detection/v3_fall_detection.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


frames = []
for p in sorted(glob.glob('training/data/urfd/fall-0[1-4]-cam0.mp4')) + sorted(glob.glob('Test/1[34].mp4')):
    cap = cv2.VideoCapture(p)
    for _ in range(10):
        ok, f = cap.read()
        if ok:
            frames.append(f)
print('frames', len(frames), 'sizes', sorted({f.shape[:2] for f in frames}))
for arm, pre in ARMS.items():
    m = load(pre)
    ts = []
    for _ in range(3):
        for f in frames:
            t = time.perf_counter(); m.preprocess_frame(f); ts.append(1000 * (time.perf_counter() - t))
    print('%-3s %-18s mean %6.1f ms  p95 %6.1f ms' % (arm, pre, np.mean(ts), np.percentile(ts, 95)), flush=True)
