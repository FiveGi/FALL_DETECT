# -*- coding: utf-8 -*-
"""Collect real day-to-night skeleton errors, to replay them onto training windows.

`ir_keypoint_degradation.py` measured what simulated night-vision footage does to the pose
model's output: the person is lost in ~12% of frames, and joints -- hips included -- jump by up
to 0.25-0.46 torso lengths at p90 while the model's confidence does not drop at all. Guessing a
noise distribution that matches those numbers would be a second model of a model. This keeps
the actual errors instead: for every frame where the day cache and the night cache both see the
person, the per-joint displacement night-minus-day in torso lengths, plus a flag for frames
where the person found by day is missing at night. dataset.ir_degrade_window() samples from it.

Built from GMDCSA24 TRAINING-side clips only (gmdcsa_fall, train50_adl) and from every IR cache
given, so no URFD clip -- the held-out test set -- contributes a single error.

Usage:
    python training/measure/build_ir_residual_bank.py <day_cache> <ir_cache> [<ir_cache> ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ir_keypoint_degradation import GROUPS, hip, torso  # noqa: E402  same matching rules
from replay_classifiers import load_stream  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'training', 'data', 'ir_residual_bank.npz')


def main():
    day_dir, ir_dirs = sys.argv[1], sys.argv[2:]
    if not ir_dirs:
        print(__doc__)
        return 1
    residuals, lost = [], []
    for ir_dir in ir_dirs:
        for f in sorted(os.listdir(day_dir)):
            if not f.endswith('.npz') or f.split('__')[0] not in GROUPS:
                continue
            p_ir = os.path.join(ir_dir, f)
            if not os.path.exists(p_ir):
                continue
            for pd, pn in zip(load_stream(os.path.join(day_dir, f)), load_stream(p_ir)):
                if len(pd) == 0:
                    continue
                kd = pd[0]
                if len(pn) == 0:
                    residuals.append(np.zeros((17, 2), np.float32))
                    lost.append(True)
                    continue
                kn = min(pn, key=lambda k: np.linalg.norm(hip(k) - hip(kd)))
                residuals.append(((kn[:, :2] - kd[:, :2]) / torso(kd)).astype(np.float32))
                lost.append(False)
    r = np.stack(residuals)
    lost = np.array(lost)
    # A match to the wrong body, or a pose model that put the skeleton somewhere absurd, would
    # otherwise enter the bank as "night-vision error" and teach the model to shrug off
    # anything. Clip at 3 torso lengths -- far beyond the measured p90 of ~0.46.
    r = np.clip(r, -3.0, 3.0)
    np.savez_compressed(OUT, residuals=r, lost=lost)
    seen = r[~lost]
    mag = np.linalg.norm(seen, axis=2)
    print('frames %d (from %d IR cache(s)); person lost in %.1f%%'
          % (len(r), len(ir_dirs), 100.0 * lost.mean()))
    print('joint error, torso lengths: median %.3f  p90 %.3f  p99 %.3f'
          % tuple(np.percentile(mag, [50, 90, 99])))
    print('wrote', OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
