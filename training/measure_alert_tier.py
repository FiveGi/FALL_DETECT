"""Does the alert score say anything about whether the alert is real?

The two-tier rule (`notification_service.alert_tier`) used to answer yes: anything at or above
0.85 was announced to families as a confirmed fall, anything below as "please check". That bar
came from clip *peaks* on GMDCSA24 val alone, and the peak is not what the system uses --
`camera_manager` passes the score **at the instant the alert fires**.

This measures the right quantity on every labelled surface at once: GMDCSA24 falls, GMDCSA24
val + train50 ADL, URFD falls, and the URFD ADL clips held out of training. Each rising edge is
one alert, scored exactly as the camera loop scores it, and labelled by the clip it came from.

The answer turned out to be no, for both the current model and the one before it: a false alarm
is *more* likely to clear a high bar than a real fall is. The tier now keys off escalation
instead. Re-run this before ever reintroducing a score-based tier.

Usage:
    V3_DEVICE=cuda python training/measure_alert_tier.py
    TEST_MODEL_DIR=/path/to/other/model V3_DEVICE=cuda python training/measure_alert_tier.py
"""
import glob
import importlib.util
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'training'))

spec = importlib.util.spec_from_file_location(
    'v3_fall_detection', os.path.join(ROOT, 'app', 'detection', 'v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

MODEL_DIR = os.environ.get('TEST_MODEL_DIR', os.path.join(ROOT, 'models'))
# The rate the live loop is pinned to. Reading every frame of a 30fps file measures a detector
# that is not deployed: the window is a fixed number of frames, so the scores it produces at
# 30 fps are not the scores it produces at 15 (SS50).
TARGET_FPS = float(os.environ.get('TARGET_FPS', os.environ.get('V3_TARGET_FPS', 15)))
DATA = os.path.join(ROOT, 'training', 'data')
OUT = os.path.join(DATA, 'tier_alert_scores.json')
BARS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]


def alerts(det, path, rgb_half=False):
    """Scores at each rising edge -- the same moment camera_manager raises an alert. URFD mp4s
    are depth+RGB side by side, so only the right half is usable."""
    state = v3.V3MultiPersonFallState()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, last, out = 0, None, []
    read, last_slot = 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        slot = int(read * TARGET_FPS / fps)
        read += 1
        if slot == last_slot:
            continue
        last_slot = slot
        if rgb_half:
            frame = frame[:, frame.shape[1] // 2:]
        hit = [r for r in v3.detect_v3_fall_multi(frame, state, det, config=None) if r[1]]
        label = 'fall' if hit else 'no_fall'
        if label != last:
            if hit:
                out.append((round(i / TARGET_FPS, 1), round(max(float(r[2]) for r in hit), 3)))
            last = label
        i += 1
    cap.release()
    return out


def main():
    det = v3.V3PoseFallDetector(model_dir=MODEL_DIR)
    rows = []
    for folder, is_fall in [('gmdcsa24_fall_raw', 1), ('gmdcsa24_adl_raw_val', 0),
                            ('gmdcsa24_adl_raw_train50', 0)]:
        for p in sorted(glob.glob(os.path.join(DATA, folder, '*.mp4'))):
            rows.append([folder, os.path.basename(p), is_fall, alerts(det, p)])

    held_out = set(open(os.path.join(DATA, 'urfd_heldout_adl.txt')).read().split())
    for p in sorted(glob.glob(os.path.join(DATA, 'urfd', '*.mp4'))):
        name = os.path.basename(p)
        if name.startswith('fall'):
            rows.append(['urfd', name, 1, alerts(det, p, rgb_half=True)])
        elif name in held_out:
            # The other half became training negatives, so it cannot be scored here.
            rows.append(['urfd', name, 0, alerts(det, p, rgb_half=True)])

    with open(OUT, 'w') as f:
        json.dump(rows, f, indent=1)

    real = [s for r in rows if r[2] == 1 for _, s in r[3]]
    false = [s for r in rows if r[2] == 0 for _, s in r[3]]
    print(f'model: {MODEL_DIR}   window: {v3.WINDOW_SIZE}   threshold: {v3.THRESHOLD}   '
          f'fps: {TARGET_FPS:.0f}')
    print(f'{len(real)} alerts on fall clips, {len(false)} on no-fall clips')
    print(f'overall alert precision: {len(real) / max(1, len(real) + len(false)):.1%}\n')
    print('  bar   real-fall alerts above it    false alarms above it    precision above it')
    for bar in BARS:
        r, f_ = sum(s >= bar for s in real), sum(s >= bar for s in false)
        prec = f'{r / (r + f_):.0%}' if r + f_ else '--'
        print(f'  {bar:.2f}      {r:3d}/{len(real)} ({r / len(real):4.0%})'
              f'            {f_:3d}/{len(false)} ({f_ / max(1, len(false)):4.0%})'
              f'            {prec}')

    print('\nhighest-scoring false alarms:')
    for r in sorted([r for r in rows if r[2] == 0 and r[3]],
                    key=lambda r: -max(s for _, s in r[3]))[:10]:
        print(f'   {max(s for _, s in r[3]):.2f}  {r[0]}/{r[1]}')
    print(f'\nper-alert scores written to {OUT}')


if __name__ == '__main__':
    main()
