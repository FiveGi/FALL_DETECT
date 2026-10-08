# -*- coding: utf-8 -*-
"""DAY-FIRST-v1 near/far buckets for the owner's 75 fall segments, frozen from STOCK poses only
(phase 0 owner cache, 23de875c4696) -- never from a candidate's output.
Person height = vertical extent of keypoints with confidence > 0.3 (normalised to frame height),
a proxy for box height (caches keep keypoints, not boxes; the keypoint span from nose to ankle is
roughly 0.85 of a standing box, so cut-offs are applied to extent / 0.85). Scale (v1) = median over the
first 1.0 s of the segment of the LARGEST visible person; far < 0.25, medium 0.25-0.50, near >= 0.50
(Codex). 'unknown' when nobody is seen in that second. multi = more than one person in >= half of
those frames (faller identity is not resolved: the largest person may not be the faller).
Writes training/data/day_first/owner_size_buckets_v4.json"""
import importlib.util
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
s = importlib.util.spec_from_file_location('si', 'training/measure/score_incidents.py')
si = importlib.util.module_from_spec(s); s.loader.exec_module(si)
truth = si.truth()
C = 'training/data/pose_cache_owner/23de875c4696'
out = {}
for seg, fall in sorted(truth.items()):
    if not fall:
        continue
    z = np.load(os.path.join(C, seg + '.npz'))
    counts, kpts, t = z['counts'], z['kpts'], z['t']
    cap = cv2.VideoCapture(os.path.join('Test', seg.split('#')[0]))
    aspect = cap.get(cv2.CAP_PROP_FRAME_WIDTH) / max(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 1); cap.release()
    at, sizes, multi = 0, [], []
    # v2 (2026-10-02, after looking at misses): the first second measured where people START, and
    # 1.mp4#3 / 8.mp4#5 start near the camera then fall far from it. Scale is now read where the
    # fall happens: the last 2.0 s of the segment, falling back to the last 2 s with anyone seen.
    seen = []
    for n, ti in zip(counts, t):
        people = kpts[at:at + int(n)]; at += int(n)
        # v3: body LENGTH in any orientation -- a lying person has a small vertical extent at any
        # distance (v2 put 58/75 in 'far'). Longest side of the keypoint span in pixels / frame height.
        # v4: the owner's clips are mostly portrait compilations -- the camera picture is a band across
        # the full WIDTH with blurred filler above and below -- so scale is taken against the width,
        # expressed as a fraction of the height of a 16:9 picture of that width (W * 9/16).
        h = [max(float(np.ptp(p[p[:, 2] > 0.3, 0])), float(np.ptp(p[p[:, 2] > 0.3, 1])) / aspect) / (9 / 16) / 0.85
             for p in people if (p[:, 2] > 0.3).sum() >= 4]
        if h:
            seen.append((float(ti), max(h), len(h) > 1))
    if seen:
        tail = [x for x in seen if x[0] >= t[-1] - 2.0] or [x for x in seen if x[0] >= seen[-1][0] - 2.0]
        sizes = [x[1] for x in tail]; multi = [x[2] for x in tail]
    for n, ti in []:
        h = [float(np.ptp(p[p[:, 2] > 0.3, 1])) / 0.85 for p in people if (p[:, 2] > 0.3).sum() >= 4]
        if h:
            sizes.append(max(h)); multi.append(len(h) > 1)
    if not sizes:
        b, m = 'unknown', None
    else:
        m = float(np.median(sizes))
        b = 'far' if m < 0.25 else ('medium' if m < 0.50 else 'near')
    out[seg] = {'bucket': b, 'scale': None if m is None else round(m, 3),
                'multi': bool(np.mean(multi) >= 0.5) if multi else None}
# v4 tertiles: Codex's fixed cut-offs (0.25 / 0.50 of frame height) were proposals, and on these
# compilation clips the proxy scale does not sit on a calibrated axis; buckets are the stock-pose
# scale's tertiles instead (still frozen from stock poses only, never a candidate's output).
known = sorted(v['scale'] for v in out.values() if v['scale'] is not None)
q1, q2 = known[len(known) // 3], known[2 * len(known) // 3]
for v in out.values():
    if v['scale'] is not None:
        v['bucket'] = 'far' if v['scale'] < q1 else ('medium' if v['scale'] < q2 else 'near')
print('tertile cut-offs', round(q1, 3), round(q2, 3))
os.makedirs('training/data/day_first', exist_ok=True)
json.dump(out, open('training/data/day_first/owner_size_buckets_v4.json', 'w'), indent=1)
from collections import Counter
print(Counter(v['bucket'] for v in out.values()), 'multi', sum(1 for v in out.values() if v['multi']))
