# -*- coding: utf-8 -*-
"""Replay a classifier over cached owner-segment pose streams -> an alerts file in
eval_incidents_cpu.py's format, scored by score_incidents.py like any other.

Faithfulness check, run before trusting it: the deployed model replayed over a phase cache must
give the same alerts as eval_incidents_cpu.py's full-pipeline run at that phase.

Usage:
  V3_THRESHOLD=0.60 python training/measure/replay_owner_segments.py <cache_dir> <model_dir> <out.json>
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
spec = importlib.util.spec_from_file_location('v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)


def frames(npz):
    with np.load(npz) as z:
        counts, kpts, t = z['counts'], z['kpts'], z['t']
    out, at = [], 0
    for n in counts:
        # Cut to the live cap; offset advances by the original count (see replay_classifiers.load_stream).
        out.append(kpts[at:at + int(n)][:v3.NUM_POSES])
        at += int(n)
    return out, t


def main(cache, model_dir, out_path):
    det = v3.V3PoseFallDetector(model_dir=model_dir)
    det._replay = []
    det.extract_all_keypoints = lambda frame: det._replay
    key = json.load(open(os.path.join(cache, 'key.json')))
    res = {'profile': dict(key, threshold=v3.THRESHOLD, model_dir=model_dir, replay=True), 'segments': {}}
    # The cap actually applied by frames(); key.json describes the CACHE (it may be uncapped).
    # Recorded only when not the default, so every cap-4 result stays byte-identical.
    if v3.NUM_POSES != 4:
        res['profile']['replay_num_poses'] = v3.NUM_POSES
    for f in sorted(os.listdir(cache)):
        if not f.endswith('.npz'):
            continue
        fr, t = frames(os.path.join(cache, f))
        st = v3.V3MultiPersonFallState()
        hits, peak = [], 0.0
        for people, ti in zip(fr, t):
            det._replay = [(kp, (kp[v3.LEFT_HIP, :2] + kp[v3.RIGHT_HIP, :2]) / 2.0) for kp in people]
            r = v3.detect_v3_fall_multi(None, st, det, config=None)
            peak = max([peak] + [x[2] for x in r])
            if any(x[1] for x in r):
                hits.append(round(float(ti), 2))
        res['segments'][f[:-4]] = {'alerts_at': hits, 'peak': round(float(peak), 3)}
    json.dump(res, open(out_path, 'w'), indent=1)
    print('wrote', out_path, len(res['segments']), 'segments')


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:4]))
