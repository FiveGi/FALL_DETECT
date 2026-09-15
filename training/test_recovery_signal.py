"""Does "did the person get back up?" separate real falls from false alarms?

Every signal tried so far (SS33's four geometric ones, SS40's four confidence/size ones) asks
the same question the classifier already asks: is THIS moment a fall. They all overlap genuine
falls because a person mid-bend and a person mid-fall look alike in a one-second window.

This asks a different question, over a longer horizon: after the alert, does the person stay
on the ground? That is what a human uses to tell the two apart, and it is the distinction that
matters for care -- someone who is back on their feet in two seconds does not need help.

Measured per alert as the fraction of the following RECOVERY_WINDOW seconds in which the
alerting person is still "low": their hip centre sits in the lower part of their own standing
height, using the tallest pose seen for that track before the alert as the reference. Height
is measured per person, not in absolute pixels, so someone far from the camera is judged on
the same scale as someone close.

Ground truth comes from the clip verification (Gemini on shot-trimmed clips), so FALL vs
NOT_A_FALL here is the same label used everywhere else.

Usage:
    python training/test_recovery_signal.py
"""
import importlib.util
import json
import os
import re

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'v3_fall_detection', os.path.join(ROOT, 'app', 'detection', 'v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

TEST_DIR = os.environ.get('TEST_DIR', os.path.join(ROOT, 'Test'))
DATA = os.path.join(ROOT, 'training', 'data')
RECOVERY_WINDOW = float(os.environ.get('RECOVERY_WINDOW', 6.0))
# Fraction of standing height below which the hip counts as "down". A standing person's hip
# sits near the middle of their own height; on the ground it is close to the bottom.
LOW_RATIO = float(os.environ.get('LOW_RATIO', 0.72))


def person_height_and_hip(kpts, frame_h):
    """-> (height, hip_y) in normalised units for one pose, or None if too little is visible."""
    vis = kpts[kpts[:, 2] > 0.3]
    if len(vis) < 5:
        return None
    top, bottom = float(vis[:, 1].min()), float(vis[:, 1].max())
    hip = (kpts[v3.LEFT_HIP, :2] + kpts[v3.RIGHT_HIP, :2]) / 2.0
    return (bottom - top), float(hip[1]), top, bottom


def recovery_score(video_path, t_alert, centroid):
    """-> fraction of the window after the alert where the nearest person is still down."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det = DETECTOR

    # Reference height: the tallest this person appeared in the 4s before the alert, which is
    # them upright. Without it, "low" would have to be an absolute number and would mean
    # different things at different distances from the camera.
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((t_alert - 4) * fps)))
    ref_height = 0.0
    for _ in range(int(4 * fps)):
        ok, f = cap.read()
        if not ok:
            break
        for kpts, hip in det.extract_all_keypoints(f):
            if float(np.linalg.norm(np.array(hip) - np.array(centroid))) < 0.25:
                m = person_height_and_hip(kpts, f.shape[0])
                if m and m[0] > ref_height:
                    ref_height = m[0]
    if ref_height <= 0:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_alert * fps))
    low = seen = 0
    for _ in range(int(RECOVERY_WINDOW * fps)):
        ok, f = cap.read()
        if not ok:
            break
        best = None
        for kpts, hip in det.extract_all_keypoints(f):
            d = float(np.linalg.norm(np.array(hip) - np.array(centroid)))
            if d < 0.35 and (best is None or d < best[0]):
                best = (d, kpts)
        if best is None:
            continue
        m = person_height_and_hip(best[1], f.shape[0])
        if not m:
            continue
        _, hip_y, top, bottom = m
        # Where the hip sits within this person's own standing height.
        rel = (hip_y - top) / max(ref_height, 1e-6)
        seen += 1
        low += rel > LOW_RATIO or (bottom - top) < ref_height * 0.55
    cap.release()
    return (low / seen) if seen else None


def main():
    global DETECTOR
    DETECTOR = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))

    alerts = json.load(open(os.path.join(DATA, 'prod_multi_alerts_new.json'), encoding='utf-8'))
    verdicts = json.load(open(os.path.join(DATA, 'prod_newshot_clipverify_results.json'),
                              encoding='utf-8'))
    diag = json.load(open(os.path.join(DATA, 'multi_diagnosis.json'), encoding='utf-8'))
    centroids = {(c, round(a['t'], 1)): a['centroid'] for c, v in diag.items() for a in v}

    rows = []
    for clip, items in sorted(alerts.items(), key=lambda x: int(x[0])):
        path = os.path.join(TEST_DIR, f'{clip}.mp4')
        if not os.path.exists(path):
            continue
        for t, p in items:
            key = f'{clip}/t={t:.1f}s_p={p:.2f}'
            raw = verdicts.get(key, '')
            m = re.search(r'"verdict"\s*:\s*"([A-Z_]+)"', raw or '')
            verdict = m.group(1) if m else None
            if verdict not in ('FALL', 'NOT_A_FALL', 'ALREADY_DOWN'):
                continue
            cen = centroids.get((clip, round(t, 1)))
            if cen is None:
                continue
            score = recovery_score(path, t, cen)
            if score is None:
                continue
            rows.append((verdict, score, clip, t, p))
            print(f'  {verdict:13} still-down={score:.2f}  clip{clip} t={t}s p={p}', flush=True)

    print()
    for label in ('FALL', 'ALREADY_DOWN', 'NOT_A_FALL'):
        vals = [s for v, s, *_ in rows if v == label]
        if vals:
            vals.sort()
            print(f'{label:13} n={len(vals):3}  min={vals[0]:.2f}  median={vals[len(vals)//2]:.2f}'
                  f'  max={vals[-1]:.2f}')

    falls = [s for v, s, *_ in rows if v in ('FALL', 'ALREADY_DOWN')]
    nots = [s for v, s, *_ in rows if v == 'NOT_A_FALL']
    if falls and nots:
        print('\nif alerts were suppressed when the person is back up quickly:')
        for gate in (0.1, 0.2, 0.3, 0.4, 0.5):
            kept_f = sum(1 for s in falls if s >= gate)
            kept_n = sum(1 for s in nots if s >= gate)
            print(f'  require still-down >= {gate:.1f} -> keeps {kept_f}/{len(falls)} real, '
                  f'{kept_n}/{len(nots)} false  (removes {len(nots) - kept_n} false alarms, '
                  f'loses {len(falls) - kept_f} real)')


if __name__ == '__main__':
    main()
