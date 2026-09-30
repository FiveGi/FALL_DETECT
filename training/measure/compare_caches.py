# -*- coding: utf-8 -*-
"""Same classifier, same clips, two pose caches -- what does the footage change cost?

Built to answer one question the corpus cannot answer on its own: this system will be pointed
at a CCTV camera, which after dark switches to its infrared sensor and sends GREY, and not one
clip here is grey. A domain gap nobody has measured is indistinguishable from a working system
until it is deployed.

Takes two cache directories, replays the SAME alerting rule over both, and reports per group
plus the clips whose answer changed, in both directions -- a footage change that only ever
loses falls and never gains one is suspicious, and the per-clip lists are what makes that
checkable rather than assumed.

Usage:
    python training/measure/compare_caches.py <baseline_dir> <changed_dir> [group ...]
"""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
from replay_classifiers import load_stream, alerted  # noqa: E402

TRUTH = {'urfd_fall': True, 'gmdcsa_fall': True,
         'urfd_adl': False, 'val_adl': False, 'train50_adl': False}


def run(cache_dir, groups):
    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    # Exactly what replay_classifiers.py does: the pose model is built so the detector is the
    # object production builds, then never called -- the cached keypoints are handed back
    # instead. Without this, the detector tries to preprocess a frame that is None.
    det._replay = []
    det.extract_all_keypoints = lambda frame: det._replay
    out = {}
    for group in groups:
        for npz in sorted(os.listdir(cache_dir)):
            if not npz.startswith(group + '__') or not npz.endswith('.npz'):
                continue
            out['%s/%s' % (group, npz.split('__', 1)[1])] = alerted(
                det, load_stream(os.path.join(cache_dir, npz)))
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    base_dir, new_dir = sys.argv[1], sys.argv[2]
    groups = sys.argv[3:] or ['urfd_fall', 'urfd_adl']

    for label, path in (('baseline', base_dir), ('changed ', new_dir)):
        with open(os.path.join(path, 'key.json')) as fh:
            k = json.load(fh)
        print('%s  imgsz %s  fps %s  preprocess %s  gate %s  ir %s  dark %s'
              % (label, k.get('imgsz'), k.get('fps'), k.get('preprocess'),
                 k.get('dark_below'), k.get('simulate_ir'), k.get('simulate_dark')))
    print()

    base, new = run(base_dir, groups), run(new_dir, groups)
    shared = sorted(set(base) & set(new))
    if len(shared) < len(base) or len(shared) < len(new):
        print('note: comparing the %d clips both caches hold (baseline %d, changed %d)'
              % (len(shared), len(base), len(new)))
        print()

    print('%-14s %8s %16s %16s' % ('group', 'clips', 'baseline', 'changed'))
    print('-' * 60)
    for group in groups:
        keys = [k for k in shared if k.startswith(group + '/')]
        if not keys:
            continue
        want = TRUTH[group]
        # For a fall group the score is alerts; for an ADL group it is silences.
        b = sum(1 for k in keys if base[k] == want)
        n = sum(1 for k in keys if new[k] == want)
        what = 'falls caught' if want else 'clean (no false alarm)'
        print('%-14s %8d %10d %5s %10d %5s   %s'
              % (group, len(keys), b, '%.0f%%' % (100.0 * b / len(keys)),
                 n, '%.0f%%' % (100.0 * n / len(keys)), what))

    print()
    lost = [k for k in shared if base[k] and not new[k]]
    gained = [k for k in shared if new[k] and not base[k]]
    print('clips that ALERTED before and do NOT now: %d' % len(lost))
    for k in lost[:20]:
        print('   %s%s' % (k, '   <- a real fall, now missed' if TRUTH[k.split('/')[0]] else
                           '   (a false alarm that went away)'))
    print('clips that did NOT alert before and DO now: %d' % len(gained))
    for k in gained[:20]:
        print('   %s%s' % (k, '   <- a fall recovered' if TRUTH[k.split('/')[0]] else
                           '   <- a NEW false alarm'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
