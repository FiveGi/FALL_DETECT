# -*- coding: utf-8 -*-
"""What the detector sees on a CCTV camera's night sensor, beside what it sees by day.

Every clip in this corpus is daytime colour. The deployment is a security camera in a bedroom,
which after dark switches to infrared and sends grey. So the footage the system will spend most
of its life looking at is a kind of image it has never been measured on -- and before measuring
it, it is worth looking at, because a simulation that does not resemble real night footage
would make the measurement meaningless.

Three panels: daytime colour, the same frame as a night sensor would send it, and that night
frame after the shipped lighting fix. Each carries the skeleton and the score, so the cost of
the grey is visible rather than argued about.

Usage:
    V3_DEVICE=cuda python training/measure/render_infrared.py \
        training/data/urfd/fall-15-cam1.mp4 test_result/infrared_demo.mp4
"""
import importlib.util
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('V3_PREPROCESS', 'auto')
os.environ.setdefault('V3_PREPROCESS_DARK_BELOW', '70')
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
os.environ['SIMULATE_IR'] = '1'
import cache_pose_streams as cache  # noqa: E402

FPS = float(os.environ.get('TARGET_FPS', 8))
TILE_W = int(os.environ.get('TILE_W', 400))
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def panel(det, state, frame, title, size, preprocess):
    shown = frame
    ops = []
    if preprocess:
        shown, ops = v3.preprocess_frame(frame.copy())
    results = v3.detect_v3_fall_multi(shown, state, det, config=None)
    people = det.extract_all_keypoints(shown)
    hit = any(r[1] for r in results)
    score = max((r[2] for r in results), default=0.0)

    img = shown.copy()
    h, w = img.shape[:2]
    colour = (0, 0, 255) if hit else (0, 220, 255)
    for kpts, _hip in people:
        pts = [(int(kpts[i, 0] * w), int(kpts[i, 1] * h)) for i in range(17)]
        for a, b in EDGES:
            if kpts[a, 2] > 0.2 and kpts[b, 2] > 0.2:
                cv2.line(img, pts[a], pts[b], colour, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            if kpts[i, 2] > 0.2:
                cv2.circle(img, p, 3, colour, -1, cv2.LINE_AA)
    img = cv2.resize(img, size)
    head = np.zeros((46, size[0], 3), np.uint8)
    head[:] = (0, 0, 170) if hit else (35, 35, 35)
    cv2.putText(head, title, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(head, 'score %.2f   %s   %s' % (
        score, 'ALERT' if hit else 'watching',
        '%d body' % len(people) if people else 'NO BODY'),
        (7, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, img]), hit, len(people), ops


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    source, dest = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    np.random.seed(0)

    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    names = ['daytime colour (what we measured on)',
             'night sensor, no cleanup',
             'night sensor + lighting fix']
    states = [v3.V3MultiPersonFallState() for _ in names]
    ever, seen, frames, ops_seen = [False] * 3, [0] * 3, 0, set()

    cap = cv2.VideoCapture(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer, i, last_slot = None, 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        slot = int(i * FPS / src_fps)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        if frame.shape[1] > frame.shape[0] * 1.7:
            frame = frame[:, frame.shape[1] // 2:]
        ir = cache.to_infrared(frame)
        h, w = frame.shape[:2]
        size = (TILE_W, int(h * TILE_W / w))

        tiles = []
        for n, (src, pre) in enumerate([(frame, False), (ir, False), (ir, True)]):
            tile, hit, found, ops = panel(det, states[n], src, names[n], size, pre)
            ever[n] |= hit
            seen[n] += 1 if found else 0
            ops_seen.update(ops)
            tiles.append(tile)
        frames += 1
        row = np.hstack(tiles)
        if writer is None:
            writer = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*'mp4v'), src_fps / 2,
                                     (row.shape[1], row.shape[0]))
        for _ in range(max(1, int(round(src_fps / FPS / 2)))):
            writer.write(row)
    cap.release()
    if writer is not None:
        writer.release()

    print('wrote %s' % dest)
    print('cleanup steps the night frames triggered: %s' % (sorted(ops_seen) or 'none'))
    print()
    print('%-38s %13s %14s' % ('', 'body found', 'alert'))
    for n, name in enumerate(names):
        print('%-38s %10d/%-3d %14s'
              % (name, seen[n], frames, 'yes' if ever[n] else 'NO -- missed'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
