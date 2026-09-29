# -*- coding: utf-8 -*-
"""Why does total throughput FALL when a second camera process starts?

multi_camera_capacity.py measures one camera at 10.9 fps on the CPU profile, two at 3.1 each
and three at 0.7 -- a total of 10.9, 6.2 and 2.0. Contention explains a fall in the per-camera
rate. It does not explain the TOTAL falling: three workers sharing a machine should still get
more done than one, unless they are fighting rather than queueing.

Two guesses have already been measured and were both wrong: thread oversubscription (torch is
already at one thread per process, and dividing the quota changed nothing) and OpenCV's pool
(pinning it to one thread changed nothing).

So this stops guessing and reports where the time goes: CPU seconds consumed per process
against wall seconds elapsed. A process that is CPU-bound and merely contending will show close
to one core's worth; a process that is blocked on something -- a lock, the GPU, memory -- will
show far less, and then the missing time is the answer.

It also compares fork with spawn, because these run forked from a parent that has already
imported torch, which is its own documented hazard.

**Must be a file, not `python -c`.** spawn re-imports the main module in the child, and a
`-c` program has none, so the first version of this died without printing anything at all.

Usage:
    V3_DEVICE=cpu V3_IMGSZ=320 python training/measure/diagnose_multi_process.py
"""
import importlib.util
import multiprocessing as mp
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SECONDS = float(os.environ.get('SECONDS', 15))
SOURCE = os.environ.get('SOURCE', os.path.join(ROOT, 'Test', '15.mp4'))


def _frame():
    cap = cv2.VideoCapture(SOURCE)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.6))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit('could not read %s' % SOURCE)
    return cv2.resize(frame, (640, 360))


def work(index, queue, seconds):
    spec = importlib.util.spec_from_file_location(
        'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)
    frame = _frame()
    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    state = v3.V3MultiPersonFallState()
    for _ in range(3):
        v3.detect_v3_fall_multi(frame, state, det, config=None)

    count = 0
    wall0, cpu0 = time.monotonic(), time.process_time()
    while time.monotonic() - wall0 < seconds:
        v3.detect_v3_fall_multi(frame, state, det, config=None)
        count += 1
    wall = time.monotonic() - wall0
    queue.put((index, count / wall, (time.process_time() - cpu0) / wall))


def run(method, n):
    ctx = mp.get_context(method)
    queue = ctx.Queue()
    procs = [ctx.Process(target=work, args=(i, queue, SECONDS)) for i in range(n)]
    for p in procs:
        p.start()
    results = [queue.get() for _ in range(n)]
    for p in procs:
        p.join()
    fps = [r[1] for r in results]
    cores = [r[2] for r in results]
    print('%-6s %d process(es): each %5.2f fps   total %5.2f fps   '
          'cores busy: each %.2f, total %.2f'
          % (method, n, sum(fps) / n, sum(fps), sum(cores) / n, sum(cores)), flush=True)
    return sum(fps), sum(cores)


def main():
    print('source 640x360, %.0fs per point, quota %s cores'
          % (SECONDS, os.environ.get('CPU_QUOTA', '?')), flush=True)
    print(flush=True)
    for method in ('fork', 'spawn'):
        for n in (1, 2, 3):
            try:
                run(method, n)
            except Exception as exc:
                print('%-6s %d process(es): FAILED %s: %s'
                      % (method, n, type(exc).__name__, exc), flush=True)
        print(flush=True)
    print('Read the "cores busy, total" column. If it rises with the process count while the')
    print('fps total falls, the work is being done and thrown away -- thrashing. If it stays')
    print('flat and low, the processes are blocked on each other and the lock is the answer.')


if __name__ == '__main__':
    main()
