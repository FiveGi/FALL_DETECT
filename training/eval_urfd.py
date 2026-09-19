"""Run the deployed detector on UR Fall Detection (URFD) -- a dataset it has never seen.

Every number this project reports comes from GMDCSA24 (plus the unlabelled Test/ clips).
Settings have been chosen against GMDCSA24 more than once, so it is no longer a neutral
measure of anything. URFD is independent: a different lab, different room, different camera,
different actors, and nothing here has ever been tuned on it.

Two things to know about the footage:
  - each mp4 is a side-by-side composite, depth on the left and RGB on the right, so only the
    right half is usable and the usable frame is 320x240 -- small, which matters, since input
    resolution was measured to be the single biggest lever on recall;
  - the fall clips average 3.3 seconds, which is why this does NOT answer the
    "does the person get back up" question the download was meant to test.

Usage:
    python training/eval_urfd.py
"""
import glob
import importlib.util
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'v3_fall_detection', os.path.join(ROOT, 'app', 'detection', 'v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

CLIPS = os.path.join(ROOT, 'training', 'data', 'urfd')
sys.path.insert(0, os.path.join(ROOT, 'training'))
from frame_sampler import sampled_frames, TARGET_FPS  # noqa: E402


def run(detector, path):
    state = v3.V3MultiPersonFallState()
    alerts, last, peak = 0, None, 0.0
    # rgb_half: the left half is a depth map, and feeding it to a pose model trained on colour
    # images would be measuring the wrong thing. Frames arrive at the rate the live loop is
    # pinned to -- reading every frame measures a detector that is not deployed (frame_sampler).
    for frame in sampled_frames(path, rgb_half=True):
        results = v3.detect_v3_fall_multi(frame, state, detector, config=None)
        for _, _, p, _, _ in results:
            peak = max(peak, float(p))
        label = 'fall' if any(r[1] for r in results) else 'no_fall'
        if label != last:
            alerts += label == 'fall'
            last = label
    return alerts, peak


def main():
    detector = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    falls = sorted(glob.glob(os.path.join(CLIPS, 'fall-*.mp4')))
    adls = sorted(glob.glob(os.path.join(CLIPS, 'adl-*.mp4')))

    caught, fall_peaks, missed = 0, [], []
    for p in falls:
        a, peak = run(detector, p)
        fall_peaks.append(peak)
        if a:
            caught += 1
        else:
            missed.append((os.path.basename(p), round(peak, 2)))

    clean, adl_peaks, noisy = 0, [], []
    for p in adls:
        a, peak = run(detector, p)
        adl_peaks.append(peak)
        if a == 0:
            clean += 1
        else:
            noisy.append((os.path.basename(p), a, round(peak, 2)))

    print(f'\nURFD, never used for tuning (usable frame 320x240), fed at {TARGET_FPS:.0f} fps')
    print(f'  falls caught      : {caught}/{len(falls)} ({caught / max(len(falls),1):.0%})')
    print(f'  ADL with no alert : {clean}/{len(adls)} ({clean / max(len(adls),1):.0%})')
    if fall_peaks:
        fall_peaks.sort()
        print(f'  peak score on fall clips: median {fall_peaks[len(fall_peaks)//2]:.2f}, '
              f'min {fall_peaks[0]:.2f}')
    if adl_peaks:
        adl_peaks.sort()
        print(f'  peak score on ADL clips : median {adl_peaks[len(adl_peaks)//2]:.2f}, '
              f'max {adl_peaks[-1]:.2f}')
    if missed:
        print(f'\n  missed falls ({len(missed)}), with how close they came to {v3.THRESHOLD}:')
        for name, peak in missed[:10]:
            print(f'     {name:26} peak {peak}')
    if noisy:
        print(f'\n  ADL clips that alerted ({len(noisy)}):')
        for name, a, peak in noisy[:10]:
            print(f'     {name:26} {a} alert(s), peak {peak}')


if __name__ == '__main__':
    main()
