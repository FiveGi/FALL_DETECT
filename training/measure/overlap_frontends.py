# -*- coding: utf-8 -*-
"""Discriminating test: which pose front-end separates OVERLAPPING people, and at what CPU cost?

Codex and Gemini disagreed (AI_HANDOFF.md, 2026-09-30): top-down RTMPose behind a detector vs the
one-stage RTMO. Both named the same failure -- two overlapping people returned as one skeleton,
seen by eye on Test/8 segment 1 at 1.4 s. This counts people returned on overlap frames and times
each front-end on the same frames. Run inside a throwaway CPU container with rtmlib installed.

Usage: python training/measure/overlap_frontends.py
"""
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
# (clip, seconds): overlap moments -- 8#1 by eye; the rest are Gemini-verified multi-person falls.
PROBES = [('Test/8.mp4', t) for t in (1.2, 1.4, 1.6, 2.0)] + \
         [('Test/1.mp4', 67.0), ('Test/4.mp4', 16.0), ('Test/8.mp4', 30.0)]


def frames():
    out = []
    for clip, t in PROBES:
        cap = cv2.VideoCapture(clip)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * (cap.get(cv2.CAP_PROP_FPS) or 30)))
        ok, f = cap.read()
        cap.release()
        if ok:
            # Keep the aspect ratio: most compilation clips are VERTICAL phone video, and a first
            # version squashed them to 640x360, which distorted people until YOLO found none.
            k = 640.0 / max(f.shape[:2])
            out.append(('%s@%.1f' % (os.path.basename(clip), t),
                        cv2.resize(f, (int(f.shape[1] * k), int(f.shape[0] * k)))))
    return out


def count_people(kpts, scores, thr=0.3):
    if kpts is None or len(kpts) == 0:
        return 0
    s = np.asarray(scores)
    return int(sum(1 for i in range(len(s)) if (s[i] > thr).sum() >= 5))   # >=5 confident joints


SHEET = []          # (front-end, frame name, image with skeletons) -- judged by eye, not by count


def draw(name, frame_name, im, kpts, scores, thr=0.3):
    out = im.copy()
    colours = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255)]
    for i, (k, s) in enumerate(zip(np.asarray(kpts), np.asarray(scores))):
        if (s > thr).sum() < 5:
            continue
        for (x, y), c in zip(k, s):
            if c > thr:
                cv2.circle(out, (int(x), int(y)), 3, colours[i % 4], -1)
    cv2.putText(out, '%s %s' % (name[:18], frame_name), (4, 16), 0, 0.45, (255, 255, 255), 1)
    SHEET.append(out)


def yolo_pose(res):
    if res.keypoints is None or len(res.keypoints) == 0:
        return np.zeros((0, 17, 2)), np.zeros((0, 17))
    return res.keypoints.xy.cpu().numpy(), res.keypoints.conf.cpu().numpy()


def timed(fn, fs, reps=5):
    for _, f in fs[:2]:
        fn(f)
    t0 = time.perf_counter()
    for _ in range(reps):
        for _, f in fs:
            fn(f)
    return 1000 * (time.perf_counter() - t0) / (reps * len(fs))


def main():
    fs = frames()
    rows = {}
    from ultralytics import YOLO
    yolo = YOLO('models/yolo26s-pose.pt')
    for conf in (0.30, 0.15):
        f = lambda im, c=conf: yolo.predict(im, verbose=False, conf=c, classes=[0], imgsz=320, device='cpu')[0]
        name = 'YOLO26s 320 conf %.2f' % conf
        counts = []
        for n, im in fs:
            k, c = yolo_pose(f(im))
            counts.append(count_people(k, c))
            draw(name, n, im, k, c)
        rows[name] = (counts, timed(f, fs))
    try:
        from rtmlib import RTMO, Body
        rtmo = RTMO('https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
                    'rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip',
                    model_input_size=(640, 640), backend='onnxruntime', device='cpu')
        g = lambda im: rtmo(im)
        counts = []
        for n, im in fs:
            k, c = g(im)
            print('RTMO raw score range %s: %.2f..%.2f' % (n, np.min(c), np.max(c)) if len(c) else 'RTMO %s: none' % n)
            counts.append(count_people(k, c))
            draw('RTMO-s', n, im, k, c)
        rows['RTMO-s 640'] = (counts, timed(g, fs))
        body = Body(mode='lightweight', backend='onnxruntime', device='cpu')   # RTMDet-nano + RTMPose-t
        h = lambda im: body(im)
        counts = []
        for n, im in fs:
            k, c = h(im)
            counts.append(count_people(k, c))
            draw('RTMDet+RTMPose-t', n, im, k, c)
        rows['RTMDet-nano + RTMPose-t'] = (counts, timed(h, fs))
    except Exception as exc:
        print('rtmlib models unavailable: %s' % exc)

    if SHEET:
        h = max(im.shape[0] for im in SHEET)
        cells = [cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT) for im in SHEET]
        per = len(fs)
        rows_img = [np.hstack(cells[i:i + per][:4]) for i in range(0, len(cells), per)]
        w = max(r.shape[1] for r in rows_img)
        rows_img = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows_img]
        cv2.imwrite('test_result/_overlap_frontends.jpg', cv2.resize(np.vstack(rows_img), None, fx=0.6, fy=0.6))
    print('probe frames: ' + ', '.join(n for n, _ in fs))
    print('%-28s %-28s %s' % ('front-end', 'people with >=5 joints > 0.3', 'ms/frame'))
    for name, (counts, ms) in rows.items():
        print('%-28s %-28s %6.1f' % (name, ' '.join(map(str, counts)), ms))
    return 0


if __name__ == '__main__':
    sys.exit(main())
