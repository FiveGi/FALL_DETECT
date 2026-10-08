# -*- coding: utf-8 -*-
"""Audit OF-Syn before it is trained on (Codex DATA-DESIGN #6): 120 clips, stratified.

Synthetic video can carry things real cameras do not -- framing that drifts or jumps inside a
"static" shot, limbs that morph, falls whose timing disagrees with the label. A pose classifier
reads motion, so camera motion becomes fake body motion. This draws contact sheets for a
person (and Gemini) to judge, and measures camera motion per clip so that, if the measure agrees
with the eye, it can screen all 12,000.

Camera motion: phase correlation between consecutive 8 fps frames on the image BORDER only
(a 15% frame band, where the person rarely is), as a fraction of the frame width; a cut shows as
a near-zero correlation response. Reported per clip: max shift, mean shift, min response.

Writes test_result/syn_audit/sheet_XX.jpg (10 clips each) and syn_audit.json.
Usage: python training/audit_omnifall_syn.py
"""
import json
import os
import random

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, 'training', 'data', 'omnifall_syn')
POSES = os.path.join(ROOT, 'training', 'data', 'poses_omnifall_syn')
OUT = os.path.join(ROOT, 'test_result', 'syn_audit')
N = int(os.environ.get('N', 120))


def border_mask(h, w, band=0.15):
    m = np.zeros((h, w), np.float32)
    bh, bw = int(h * band), int(w * band)
    m[:bh], m[-bh:], m[:, :bw], m[:, -bw:] = 1, 1, 1, 1
    return m


def camera_motion(frames):
    g = [cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    m = border_mask(180, 320)
    win = cv2.createHanningWindow((320, 180), cv2.CV_32F)
    shifts, resp = [], []
    for a, b in zip(g, g[1:]):
        (dx, dy), r = cv2.phaseCorrelate(a * m, b * m, win)
        shifts.append(float(np.hypot(dx, dy)) / 320.0)
        resp.append(float(r))
    return max(shifts), float(np.mean(shifts)), min(resp)


def main():
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(os.path.join(SYN, 'labels', 'of-syn.csv'))
    clips = lab.groupby('path').first().reset_index()
    clips['action'] = clips['path'].str.split('/').str[0]
    rng = random.Random(20261001)
    strata = sorted(clips.groupby(['action', 'age_group']).groups.items())
    pick = []
    # Round-robin over (action, age) strata so every action and every age group is represented.
    pools = {k: rng.sample(list(v), len(v)) for k, v in strata}
    while len(pick) < N:
        for k in list(pools):
            if pools[k] and len(pick) < N:
                pick.append(clips.loc[pools[k].pop(), 'path'])
    segs = {p: g[['label', 'start', 'end']].values for p, g in lab.groupby('path')}
    rows, tiles = [], []
    for i, p in enumerate(pick):
        cap = cv2.VideoCapture(os.path.join(SYN, 'videos', p + '.mp4'))
        F = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            F.append(f)
        f8 = F[::2]
        mx, mean, rmin = camera_motion(f8)
        kp_path = os.path.join(POSES, p.replace('/', '__') + '_o0.npz')
        kp = np.load(kp_path)['keypoints'] if os.path.exists(kp_path) else None
        labels = ' '.join('%d:%.1f-%.1f' % (l, a, b) for l, a, b in segs[p])
        rows.append({'clip': p, 'cam_shift_max': round(mx, 4), 'cam_shift_mean': round(mean, 4),
                     'min_response': round(rmin, 3), 'labels': labels})
        strip = []
        for j in np.linspace(0, len(f8) - 1, 6).astype(int):
            f = f8[j].copy()
            h, w = f.shape[:2]
            if kp is not None and j < len(kp):
                for x, y, c in kp[j]:
                    if c > 0.3:
                        cv2.circle(f, (int(x * w), int(y * h)), 8, (0, 255, 0), -1)
            t = j / 8.0
            lab_now = [int(l) for l, a, b in segs[p] if a <= t < b]
            cv2.putText(f, '%.1fs L%s' % (t, lab_now[0] if lab_now else '-'), (15, 70), 0, 2.2,
                        (0, 0, 255) if lab_now and lab_now[0] in (1, 2) else (255, 255, 255), 5)
            strip.append(cv2.resize(f, (256, 144)))
        head = np.zeros((144, 300, 3), np.uint8)
        for r, txt in enumerate(['#%d %s' % (i, p.split('/')[1]), 'shift max %.3f' % mx,
                                 'shift mean %.3f' % mean, 'resp min %.2f' % rmin]):
            cv2.putText(head, txt, (5, 25 + 32 * r), 0, 0.6, (255, 255, 255), 1)
        tiles.append(np.hstack([head] + strip))
        if len(tiles) == 10 or i == len(pick) - 1:
            cv2.imwrite(os.path.join(OUT, 'sheet_%02d.jpg' % (i // 10)), np.vstack(tiles))
            tiles = []
    json.dump(rows, open(os.path.join(OUT, 'syn_audit.json'), 'w'), indent=1)
    print('wrote %d clips, %d sheets to %s' % (len(rows), (len(rows) + 9) // 10, OUT))


if __name__ == '__main__':
    main()
