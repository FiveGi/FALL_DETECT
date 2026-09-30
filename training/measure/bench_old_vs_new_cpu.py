# -*- coding: utf-8 -*-
"""What frame rate can each detector SUSTAIN on the CPU server? One detector per process.

The original detector (first commit ce401fa: MediaPipe PoseLandmarker lite, IMAGE mode, 30-frame
window, thr 0.50) scored 54/60 URFD falls only when fed 30 fps. The deployed detector is judged
at the rate it sustains. So the decisive question for "is the new one better on the server" is
what rate each sustains HERE: a CPU container with the production quota, on the 640x360
substream the CPU profile uses.

An earlier version of this timed both in ONE process and the YOLO number came out 6x slower than
the live worker achieves -- MediaPipe's thread pool and an over-subscribed torch were fighting.
Now each run times exactly one detector, the new one with threads tuned by cpu_tuning exactly as
the camera loop does, and the loop includes decoding and resizing, as the camera loop does.

Usage (inside the CPU container, one run per detector):
    MODE=old python training/measure/bench_old_vs_new_cpu.py
    MODE=new V3_ROI_IMGSZ=256 python training/measure/bench_old_vs_new_cpu.py
"""
import importlib.util
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
MODE = os.environ.get('MODE', 'new')
SECONDS = float(os.environ.get('SECONDS', 30))
CLIPS = tuple(os.environ.get('CLIPS', 'Test/15.mp4,Test/14.mp4,Test/16.mp4').split(','))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build():
    # Threads are pinned to the container quota BEFORE any model or ONNX session exists, as
    # the worker's process does: an onnxruntime session created first sizes its pool to every
    # host core (20 here) and then fights itself on a 3.5-core quota -- which is what made an
    # earlier run of this script read 6-10x slower than the live camera loop.
    sys.path.insert(0, ROOT)
    from app.services.cpu_tuning import tune_threads
    print('threads before build:', tune_threads('bench, before build'))
    if MODE == 'old':
        v = load('v3_original', os.path.join(ROOT, 'training/measure/v3_original.py'))
        det = v.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'training/data/orig_model'))
        st = v.V3FallDetectionState()
        return lambda f: v.detect_v3_fall(f, st, det, None)
    os.environ.setdefault('V3_DEVICE', 'cpu')
    os.environ.setdefault('V3_IMGSZ', '320')
    v = load('v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
    det = v.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    print('threads after build:', tune_threads('bench, after build'))
    st = v.V3MultiPersonFallState()
    return lambda f: v.detect_v3_fall_multi(f, st, det, config=None)


def main():
    step = build()
    caps = [cv2.VideoCapture(c) for c in CLIPS]
    k = 0

    def next_frame():
        nonlocal k
        for _ in range(len(caps)):
            ok, f = caps[k % len(caps)].read()
            if ok:
                return f
            caps[k % len(caps)].set(cv2.CAP_PROP_POS_FRAMES, 0)
            k += 1
        raise SystemExit('no frames')

    for _ in range(15):                                   # warm up
        step(cv2.resize(next_frame(), (640, 360)))
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < SECONDS:
        step(cv2.resize(next_frame(), (640, 360)))       # decode + resize + detect, like the loop
        n += 1
    ms = 1000 * (time.perf_counter() - t0) / n
    quota = open('/sys/fs/cgroup/cpu.max').read().strip() if os.path.exists('/sys/fs/cgroup/cpu.max') else '?'
    print('RESULT mode=%s roi=%s quota=%s  %.1f ms/frame -> %.1f fps sustained'
          % (MODE, os.environ.get('V3_ROI_IMGSZ', '0'), quota, ms, 1000 / ms))
    return 0


if __name__ == '__main__':
    sys.exit(main())
