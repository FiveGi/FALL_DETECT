# -*- coding: utf-8 -*-
"""What does night-vision footage do to the SKELETON? Measured, so it can be imitated in training.

On simulated IR the detector falls from ~75% to 25-33% of URFD falls, and a render showed the
pose model still finding a body in most frames. So the classifier is being fed a *degraded*
skeleton rather than none, and training on skeletons degraded the same way is the obvious lever
-- but only if "the same way" is measured rather than guessed.

This compares the same clips, same sampled frames, daytime cache against IR cache, and reports:
  - how often a person found by day is lost at night
  - per joint: how often a joint confident by day drops below the confidence floor at night
  - per joint: positional error at night, in torso lengths, when both are present
  - confidence ratio, night over day

**Measured on GMDCSA24's TRAINING-side clips only** (gmdcsa_fall, train50_adl). URFD is the
held-out test set and must not shape the augmentation that is later judged on it; val_adl feeds
the held-out clean figure, so it is left out too.

Usage:
    python training/measure/ir_keypoint_degradation.py <day_cache_dir> <ir_cache_dir> [...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_classifiers import load_stream  # noqa: E402

GROUPS = ('gmdcsa_fall', 'train50_adl')
CONF_FLOOR = 0.3      # the pose confidence the detector deploys with
NAMES = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 'r_shoulder', 'l_elbow',
         'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle',
         'r_ankle']


def hip(kp):
    return (kp[11, :2] + kp[12, :2]) / 2.0


def torso(kp):
    sh = (kp[5, :2] + kp[6, :2]) / 2.0
    return float(np.linalg.norm(sh - hip(kp))) or 1e-6


def main():
    day_dir, ir_dirs = sys.argv[1], sys.argv[2:]
    if not ir_dirs:
        print(__doc__)
        return 1
    out = {}
    for ir_dir in ir_dirs:
        key = json.load(open(os.path.join(ir_dir, 'key.json')))
        frames = person_lost = 0
        day_conf = np.zeros(17); drop = np.zeros(17); err = [[] for _ in range(17)]
        ratio = [[] for _ in range(17)]
        clips = 0
        for f in sorted(os.listdir(day_dir)):
            if not f.endswith('.npz') or f.split('__')[0] not in GROUPS:
                continue
            p_ir = os.path.join(ir_dir, f)
            if not os.path.exists(p_ir):
                continue
            clips += 1
            d, n = load_stream(os.path.join(day_dir, f)), load_stream(p_ir)
            for pd, pn in zip(d, n):
                if len(pd) == 0:
                    continue
                frames += 1
                kd = pd[0]
                if len(pn) == 0:
                    person_lost += 1
                    continue
                # Match the night person to the day person by hip position, not list order.
                kn = min(pn, key=lambda k: np.linalg.norm(hip(k) - hip(kd)))
                t = torso(kd)
                for j in range(17):
                    if kd[j, 2] < CONF_FLOOR:
                        continue
                    day_conf[j] += 1
                    ratio[j].append(kn[j, 2] / max(kd[j, 2], 1e-6))
                    if kn[j, 2] < CONF_FLOOR:
                        drop[j] += 1
                    else:
                        err[j].append(np.linalg.norm(kn[j, :2] - kd[j, :2]) / t)
        joints = {}
        for j in range(17):
            joints[NAMES[j]] = {
                'drop_rate': round(float(drop[j] / max(day_conf[j], 1)), 3),
                'err_torso_median': round(float(np.median(err[j])), 3) if err[j] else None,
                'err_torso_p90': round(float(np.percentile(err[j], 90)), 3) if err[j] else None,
                'conf_ratio_median': round(float(np.median(ratio[j])), 3) if ratio[j] else None,
            }
        out[os.path.basename(ir_dir.rstrip('/\\'))] = {
            'seed': key.get('simulation_seed'), 'clips': clips, 'frames_with_person_by_day': frames,
            'person_lost_rate': round(person_lost / max(frames, 1), 3), 'joints': joints}

        print('IR cache %s (seed %s): %d clips, %d day frames with a person'
              % (os.path.basename(ir_dir.rstrip('/\\')), key.get('simulation_seed'), clips, frames))
        print('  person lost at night: %.1f%% of those frames' % (100.0 * person_lost / max(frames, 1)))
        print('  %-11s %8s %10s %10s %10s' % ('joint', 'dropped', 'err med', 'err p90', 'conf x'))
        for name, v in joints.items():
            print('  %-11s %7.1f%% %10s %10s %10s' % (
                name, 100 * v['drop_rate'], v['err_torso_median'], v['err_torso_p90'],
                v['conf_ratio_median']))
        print()
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ir_keypoint_degradation.json')
    json.dump(out, open(dest, 'w'), indent=1)
    print('wrote', dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
