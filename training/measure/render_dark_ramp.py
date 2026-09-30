# -*- coding: utf-8 -*-
"""The same fall at four levels of darkness, so the point where the system goes blind is visible.

"It works in low light" is not a claim anybody should accept without being shown where it stops
working. This plays one fall four times side by side -- full light, and three levels of dimming
-- with the shipped lighting fix ON in every panel, because the question is not whether the fix
helps, it is how far the fix reaches.

What degrades is not the picture, it is the skeleton. Watch the panels right to left: the pose
model does not stop returning a body, it starts returning a *wrong* body, limbs snapping to
whatever edge survived the dark. The classifier reads that as motion. So the failure at the
dark end is not silence, it is noise that can go either way -- which is exactly why this needs
watching rather than a recall number.

Usage:
    V3_DEVICE=cuda python training/measure/render_dark_ramp.py \
        training/data/urfd/fall-15-cam1.mp4 test_result/dark_ramp.mp4
"""
import importlib.util
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
os.environ.setdefault('V3_PREPROCESS', 'auto')
os.environ.setdefault('V3_PREPROCESS_DARK_BELOW', '70')
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

LEVELS = [float(x) for x in os.environ.get('LEVELS', '1.0,0.5,0.35,0.25').split(',')]
NOISE = float(os.environ.get('DARK_NOISE', 6.0))
FPS = float(os.environ.get('TARGET_FPS', 8))
TILE_W = int(os.environ.get('TILE_W', 360))
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def darken(frame, level):
    if level >= 1.0:
        return frame
    out = frame.astype(np.float32) * level
    if NOISE > 0:
        out += np.random.normal(0.0, NOISE * (1.0 - level), out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def panel(det, state, frame, level, size):
    lum = v3.frame_luminance(frame)
    shown, _ops = v3.preprocess_frame(frame.copy())
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
    cv2.putText(head, 'light %d%%   (brightness %d)' % (level * 100, lum), (7, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(head, 'score %.2f   %s   %s'
                % (score, 'ALERT' if hit else 'watching',
                   '%d body' % len(people) if people else 'NO BODY FOUND'),
                (7, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, img]), hit, len(people)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    source, dest = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    np.random.seed(0)

    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    states = [v3.V3MultiPersonFallState() for _ in LEVELS]
    ever = [False] * len(LEVELS)
    seen = [0] * len(LEVELS)
    frames = 0

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
        h, w = frame.shape[:2]
        size = (TILE_W, int(h * TILE_W / w))
        tiles = []
        for n, level in enumerate(LEVELS):
            tile, hit, found = panel(det, states[n], darken(frame, level), level, size)
            ever[n] |= hit
            seen[n] += 1 if found else 0
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
    print()
    print('%-10s %14s %10s' % ('light', 'body found', 'alert'))
    for n, level in enumerate(LEVELS):
        print('%-10s %11d/%-3d %10s'
              % ('%d%%' % (level * 100), seen[n], frames, 'yes' if ever[n] else 'NO -- missed'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
