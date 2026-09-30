# -*- coding: utf-8 -*-
"""The same fall, dimmed, with the lighting fix off and on. Side by side, in motion.

Six URFD falls are missed at half light without the preprocessing and caught with it. That is
a table row until somebody watches one, and what it looks like is the point: the pose model
does not fail cleanly in the dark, it returns a skeleton that twitches, and a twitching
skeleton is read as fast motion, which is what a fall looks like -- so the classifier is being
fed noise at exactly the moment it matters.

Left: dimmed frame, no cleanup, with whatever skeleton was found and the classifier's score.
Right: the same dimmed frame with the shadows lifted, and the same two things.
The banner turns red on the frame where each side would raise an alert.

Usage:
    V3_DEVICE=cuda DARK=0.5 python training/measure/render_dark_comparison.py \
        training/data/urfd/fall-15-cam1.mp4 test_result/dark_demo.mp4
"""
import importlib.util
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)


def load(preprocess, dark_below):
    os.environ['V3_PREPROCESS'] = preprocess
    os.environ['V3_PREPROCESS_DARK_BELOW'] = str(dark_below)
    spec = importlib.util.spec_from_file_location(
        'v3_' + preprocess, os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))


DARK = float(os.environ.get('DARK', 0.5))
NOISE = float(os.environ.get('DARK_NOISE', 6.0))
FPS = float(os.environ.get('TARGET_FPS', 8))
TILE_W = int(os.environ.get('TILE_W', 460))
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def darken(frame):
    out = frame.astype(np.float32) * DARK
    if NOISE > 0:
        out += np.random.normal(0.0, NOISE * (1.0 - DARK), out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def panel(mod, det, state, frame, title, size):
    results = mod.detect_v3_fall_multi(frame, state, det, config=None)
    people = det.extract_all_keypoints(frame)
    hit = any(r[1] for r in results)
    score = max((r[2] for r in results), default=0.0)
    img = frame.copy()
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
    bar = np.zeros((30, size[0], 3), np.uint8)
    bar[:] = (0, 0, 170) if hit else (35, 35, 35)
    cv2.putText(bar, '%s   score %.2f%s' % (title, score, '   ALERT' if hit else ''),
                (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img]), hit


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    source, dest = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)

    np.random.seed(0)
    mod_off, det_off = load('off', 70)
    mod_on, det_on = load('auto', 70)
    state_off, state_on = mod_off.V3MultiPersonFallState(), mod_on.V3MultiPersonFallState()

    cap = cv2.VideoCapture(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer, i, last_slot, ever = None, 0, -1, [False, False]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        slot = int(i * FPS / src_fps)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        if frame.shape[1] > frame.shape[0] * 1.7:      # URFD: depth beside colour
            frame = frame[:, frame.shape[1] // 2:]
        dim = darken(frame)
        lifted, _ops = mod_on.preprocess_frame(dim.copy())

        h, w = dim.shape[:2]
        size = (TILE_W, int(h * TILE_W / w))
        left, hit_l = panel(mod_off, det_off, state_off, dim, 'dimmed, no cleanup', size)
        right, hit_r = panel(mod_on, det_on, state_on, lifted, 'same frame, shadows lifted', size)
        ever[0] |= hit_l
        ever[1] |= hit_r
        pair = np.hstack([left, right])
        for _ in range(max(1, int(round(src_fps / FPS / 2)))):   # slow enough to watch
            if writer is None:
                writer = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*'mp4v'), src_fps / 2,
                                         (pair.shape[1], pair.shape[0]))
            writer.write(pair)
    cap.release()
    if writer is not None:
        writer.release()
    print('wrote %s   dimmed to %.0f%%   alert without cleanup: %s   with cleanup: %s'
          % (dest, DARK * 100, 'YES' if ever[0] else 'NO -- the fall is missed',
             'YES' if ever[1] else 'NO'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
