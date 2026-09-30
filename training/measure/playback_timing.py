# -*- coding: utf-8 -*-
"""Does the worker's file playback feed the classifier different motion than the evaluator?

The worker paces itself to the target rate in WALL-CLOCK time and then reads the next
CONSECUTIVE frame of the file. For an RTSP camera opened with CAP_PROP_BUFFERSIZE=1 that is
correct -- the newest frame is whatever the camera has now, so wall-clock pacing samples source
time. For a FILE it is not: nothing skips, so every frame is classified and the clip simply
plays slower.

The offline evaluator samples source time: `int(i * target_fps / source_fps)`.

The classifier's window is a fixed 15 frames, so what changes is not throughput but how much
real motion a window spans -- and a fall is a motion signature. This runs the deployed detector
over the same clip both ways and reports whether the alert survives, which is the only form of
this question worth answering.

Usage:
    V3_DEVICE=cuda TARGET_FPS=20 python training/measure/playback_timing.py Test/15.mp4
"""
import importlib.util
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

FPS = float(os.environ.get('TARGET_FPS', 20))


def frames_source_time(path, fps):
    """What the evaluator feeds: source time sampled to `fps`."""
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, last_slot, out = 0, -1, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        slot = int(i * fps / src)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        out.append((i - 1, frame))
    cap.release()
    return out, src


def frames_consecutive(path):
    """What the worker feeds a FILE source: every frame, in order, nothing skipped."""
    cap = cv2.VideoCapture(path)
    out = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append((i, frame))
        i += 1
    cap.release()
    return out


def run(frames):
    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    state = v3.V3MultiPersonFallState()
    alerts, peak, first = 0, 0.0, None
    was = False
    for n, (_idx, frame) in enumerate(frames):
        results = v3.detect_v3_fall_multi(frame, state, det, config=None)
        hit = any(r[1] for r in results)
        peak = max(peak, max((r[2] for r in results), default=0.0))
        if hit and not was:
            alerts += 1
            if first is None:
                first = n
        was = hit
    return alerts, peak, first


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]

    sampled, src_fps = frames_source_time(path, FPS)
    every = frames_consecutive(path)

    print('%s   source %.1f fps, %d frames, %.1fs'
          % (path, src_fps, len(every), len(every) / src_fps))
    print('target rate %.0f fps' % FPS)
    print()
    print('%-34s %8s %10s %14s' % ('', 'frames', 'window', 'wall-clock'))
    print('-' * 72)
    print('%-34s %8d %8.2fs %12.1fs   evaluator, and what the model was validated on'
          % ('sampled by SOURCE time', len(sampled), v3.WINDOW_SIZE / FPS, len(sampled) / FPS))
    print('%-34s %8d %8.2fs %12.1fs   worker, on a FILE source'
          % ('every consecutive frame', len(every), v3.WINDOW_SIZE / src_fps, len(every) / FPS))
    print()
    print('A 15-frame window therefore spans %.2fs of real motion one way and %.2fs the other '
          '(%.1fx).' % (v3.WINDOW_SIZE / FPS, v3.WINDOW_SIZE / src_fps,
                        (v3.WINDOW_SIZE / FPS) / (v3.WINDOW_SIZE / src_fps)))
    print()

    print('%-34s %8s %8s %12s' % ('', 'alerts', 'peak', 'first alert'))
    print('-' * 72)
    for label, frames, rate in (('sampled by SOURCE time', sampled, FPS),
                                ('every consecutive frame', every, src_fps)):
        alerts, peak, first = run(frames)
        at = '%.1fs' % (first / rate) if first is not None else 'never'
        print('%-34s %8d %8.2f %12s' % (label, alerts, peak, at))
    return 0


if __name__ == '__main__':
    sys.exit(main())
