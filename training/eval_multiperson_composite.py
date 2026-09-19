"""Measure the multi-person path on the two-person composites with known labels.

Answers three questions the existing evals cannot, because every clip they use has exactly
one person in it:

  1. Does a second person in frame stop the falling person from being detected?
     (fall+adl composites: the fall must still alert.)
  2. Does a second person who is fine cause alerts that neither person causes alone?
     (adl+adl composites: any alert is wrong, and both halves are clips that do not
     false-alarm on their own.)
  3. Does the tracker keep the two people apart, or does one person's window get fed the
     other's keypoints? (Reported as the number of distinct track ids and whether alerts
     come from the side the fall is actually on.)

The single-person path is run on the same composites as the control: if it does just as
well, multi-person tracking is not earning its keep; if it does worse, the tracking is
doing its job.

Usage:
    python training/eval_multiperson_composite.py
"""
import glob
import importlib.util
import os

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'v3_fall_detection', os.path.join(ROOT, 'app', 'detection', 'v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

CLIP_DIR = os.path.join(ROOT, 'training', 'data', 'multiperson_composite')
MODEL_DIR = os.environ.get('TEST_MODEL_DIR', os.path.join(ROOT, 'models'))
# Defaults to the rate docker-compose.gpu.yml pins the live loop to.
TARGET_FPS = float(os.environ.get('TARGET_FPS', os.environ.get('V3_TARGET_FPS', 15)))


def run(detector, path, multi):
    state = v3.V3MultiPersonFallState() if multi else v3.V3FallDetectionState()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx, last = 0, None
    alerts = []
    tracks = set()
    read, last_slot = 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Feed frames at the rate the camera loop is pinned to, not every frame of the file:
        # the window is a fixed number of frames, so measuring at 30 fps measures a detector
        # that is not deployed (SS50). Integer slots, exact when TARGET_FPS == source.
        slot = int(read * TARGET_FPS / fps)
        read += 1
        if slot == last_slot:
            continue
        last_slot = slot
        width = frame.shape[1]
        if multi:
            results = v3.detect_v3_fall_multi(frame, state, detector, config=None)
            tracks.update(r[0] for r in results)
            firing = [r for r in results if r[1]]
            label = 'fall' if firing else 'no_fall'
            # x of the hip centre says which half of the frame alerted -- the fall is always
            # composited on the left, so this checks the alert is attributed to the right
            # person rather than just happening somewhere.
            side = 'left' if (firing and firing[0][4][0] < 0.5) else 'right'
        else:
            _, _, label, _ = v3.detect_v3_fall(frame, state, detector, config=None)
            side = '?'
        if label != last:
            if label == 'fall':
                alerts.append((round(idx / TARGET_FPS, 1), side))
            last = label
        idx += 1
    cap.release()
    return alerts, len(tracks)


def main():
    detector = v3.V3PoseFallDetector(model_dir=MODEL_DIR)
    fall_adl = sorted(glob.glob(os.path.join(CLIP_DIR, 'falladl_*.mp4')))
    adl_adl = sorted(glob.glob(os.path.join(CLIP_DIR, 'adladl_*.mp4')))

    print('=== fall + bystander: the fall must still be caught ===')
    print(f"{'clip':46} {'multi':>18} {'single':>10}")
    caught_m = caught_s = 0
    left_ok = 0
    for p in fall_adl:
        am, ntracks = run(detector, p, True)
        asg, _ = run(detector, p, False)
        caught_m += len(am) > 0
        caught_s += len(asg) > 0
        if am and any(s == 'left' for _, s in am):
            left_ok += 1
        name = os.path.basename(p).replace('falladl_', '').replace('.mp4', '')[:44]
        sides = ','.join(s for _, s in am[:3])
        print(f'{name:46} {len(am):>3} alerts ({sides:11}) {len(asg):>3} alerts  tracks={ntracks}')
    print(f'\nfall caught: multi {caught_m}/{len(fall_adl)}   single {caught_s}/{len(fall_adl)}')
    print(f'alert attributed to the falling half (left): {left_ok}/{len(fall_adl)}')

    # Control: same 2x-wide frame, second half blank. Separates "the bystander broke
    # detection" from "the person is rendered at half the pixels now", which the
    # composite changes at the same time and which has nothing to do with tracking.
    blanks = sorted(glob.glob(os.path.join(CLIP_DIR, 'fallblank_*.mp4')))
    if blanks:
        print('')
        print('=== control: same wide frame, NO second person ===')
        caught_b = 0
        missed = []
        for bp in blanks:
            ab, _ = run(detector, bp, True)
            caught_b += len(ab) > 0
            if not ab:
                missed.append(os.path.basename(bp).replace('fallblank_', '').replace('.mp4', ''))
        print('fall caught with a blank half: %d/%d' % (caught_b, len(blanks)))
        if missed:
            print('missed with nobody else in frame:', ', '.join(missed))

    print('\n=== two people, nobody falls: any alert is wrong ===')
    clean_m = clean_s = 0
    for p in adl_adl:
        am, ntracks = run(detector, p, True)
        asg, _ = run(detector, p, False)
        clean_m += len(am) == 0
        clean_s += len(asg) == 0
        name = os.path.basename(p).replace('adladl_', '').replace('.mp4', '')[:44]
        print(f'{name:46} {len(am):>3} alerts             {len(asg):>3} alerts  tracks={ntracks}')
    print(f'\nclean: multi {clean_m}/{len(adl_adl)}   single {clean_s}/{len(adl_adl)}')


if __name__ == '__main__':
    main()
