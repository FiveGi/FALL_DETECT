# -*- coding: utf-8 -*-
"""Night MOVEMENT kill test (Gemini's top cues, 2026-10-01): can a pixel-motion signal see the fall
when the skeleton is lost? Per URFD cam0 fall clip (dataset labels: descent span, then lying):
signal = foreground fraction per 8 fps sample at pose-input scale (320 px wide, grey, blurred):
  M1 MOG2 (history 200, varThreshold 16, no shadows)   M2 frame difference |f_t - f_t-1| > 15
SNR = peak during the descent / 95th percentile while lying still (0.75 s after the descent to the clip end): the
fall must stand out from the noise of the same scene with the same person on the floor.
Conditions: day, IR (cache_pose_streams.to_infrared, seed 0), NIGHT_ALT (to_night_alt, seed 0).
Kill rule (Gemini): night SNR not clearly above day-level separability -> the cue is dead at night.
Usage: python training/measure/night_motion_probe.py [A|B]"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, 'training'); sys.path.insert(0, 'training/measure')
os.environ.setdefault('SIMULATE_IR_NOISE', '5.0')
import cache_pose_streams as cps  # noqa: E402
from fall_presence import spans  # noqa: E402
from urfd_split import half  # noqa: E402

HALF = sys.argv[1] if len(sys.argv) > 1 else 'A'


def series(path, cond):
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rng = np.random.default_rng(0)
    mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
    i, last, prev, m1, m2 = 0, -1, None, [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        slot = int(i * 8 / src); i += 1
        if slot == last:
            continue
        last = slot
        f = f[:, f.shape[1] // 2:]          # Codex review P2: RGB half only (URFD is depth|RGB)
        if cond == 'ir':
            f = cps.to_infrared(f, rng=rng)
        elif cond == 'alt':
            f = cps.to_night_alt(f, rng=rng)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(cv2.resize(g, (320, int(320 * g.shape[0] / g.shape[1]))), (5, 5), 0)
        m1.append(float((mog.apply(g) > 0).mean()))
        m2.append(float((cv2.absdiff(g, prev) > 15).mean()) if prev is not None else 0.0)
        prev = g
    return np.array(m1), np.array(m2)


if __name__ == '__main__':
    sp = {s: v for s, v in spans().items() if half(s + '-cam0.mp4') == HALF}
    for cond in ('day', 'ir', 'alt'):
        snr = {'M1': [], 'M2': []}
        for seq, (t0, t1, _) in sorted(sp.items()):
            m = dict(zip(('M1', 'M2'), series('training/data/urfd/%s-cam0.mp4' % seq, cond)))
            t = np.arange(len(m['M1'])) / 8.0
            d = (t >= t0) & (t < t1)
            still = t >= t1 + 0.75
            if d.sum() == 0 or still.sum() < 3:
                continue
            for k, v in m.items():
                snr[k].append(v[d].max() / (np.percentile(v[still], 95) + 1e-3))
        print('half %s %-4s n=%d  MOG2 SNR median %.1f (p25 %.1f)  diff SNR median %.1f (p25 %.1f)' % (
            HALF, cond, len(snr['M1']), np.median(snr['M1']), np.percentile(snr['M1'], 25),
            np.median(snr['M2']), np.percentile(snr['M2'], 25)), flush=True)
