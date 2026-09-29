# -*- coding: utf-8 -*-
"""How many cameras can one machine actually run, and what does the second one cost the first?

"One camera" has been the assumption in every document here and was never a measurement.
`docs/next_steps.md` has carried "multi-camera capacity has never been measured" for weeks,
which makes it the oldest unanswered question in the project and the one a buyer asks first.

It matters because on this pipeline **frame rate is recall**: a second camera does not simply
add load, it takes frames away from the first, and at the CPU profile the difference between
8 fps and 6 fps is the difference between catching half the falls and catching a fifth. So the
number worth reporting is not "it ran" but the rate each camera achieves.

What it measures: N detector loops in N threads over the same work, at a given profile, with
the pose pass and the classifier doing exactly what they do live. What it does not measure is
frame decoding from N real RTSP streams, which is real work this skips by replaying one
decoded frame -- so the rates here are an optimistic ceiling, and the note says so rather than
letting the number travel without it.

Usage:
    V3_DEVICE=cpu V3_IMGSZ=320 MAX_CAMERAS=4 python training/measure/multi_camera_capacity.py
"""
import importlib.util
import json
import os
import multiprocessing as mp
import statistics
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

MAX_CAMERAS = int(os.environ.get('MAX_CAMERAS', 4))
SECONDS = float(os.environ.get('SECONDS', 20))
SOURCE = os.environ.get('SOURCE', 'Test/15.mp4')
# The substream the deployment notes require. Measuring at 1080p would measure a mistake.
WIDTH, HEIGHT = 640, 360
TARGET = float(os.environ.get('V3_TARGET_FPS', 8))


# PROCESSES, not threads, because that is how the system actually runs: one Celery task per
# camera in a prefork pool. A threaded version of this measured GPU throughput capping at
# 20 fps no matter how many cameras were added, which is Python's GIL serialising the
# CPU-side work -- an artefact of the harness that production does not have.
#
# Loading a pose model also takes seconds, and N of them load at once. An earlier version built
# the detectors inside the timed region and produced a "curve" where two cameras ran slower
# than three: startup variance, not capacity. Each process now loads and warms its own detector,
# then waits at a barrier, so every camera measures a window its neighbours were competing in.
def one_camera(index, frame, ready, go, seconds, rates, share):
    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    # The same sharing the camera loop applies. Without it this measures the defect rather than
    # the capacity: every process sizes its pools to the whole quota and they fight.
    if share > 1:
        try:
            sys.path.insert(0, ROOT)
            from app.services.cpu_tuning import tune_threads
            tune_threads('capacity measurement', share=share)
        except Exception:
            pass
    # OpenCV keeps its own pool and nothing in this project has ever set it. It defaults to
    # every core on the HOST, ignoring the container quota exactly as torch used to, and each
    # camera is a separate process -- so N cameras ask for N times the machine.
    if os.environ.get('CV2_THREADS'):
        cv2.setNumThreads(int(os.environ['CV2_THREADS']))
    state = v3.V3MultiPersonFallState()
    for _ in range(3):                      # warm: lazily built kernels, an empty tracker
        v3.detect_v3_fall_multi(frame, state, det, config=None)
    ready.wait()
    go.wait()
    processed, started = 0, time.monotonic()
    while time.monotonic() - started < seconds:
        v3.detect_v3_fall_multi(frame, state, det, config=None)
        processed += 1
    rates[index] = processed / max(1e-6, time.monotonic() - started)


def main():
    cap = cv2.VideoCapture(SOURCE)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * 0.6))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit('could not read %s' % SOURCE)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    print('%dx%d, %s, %.0fs per point, target %.0f fps'
          % (WIDTH, HEIGHT, v3.LOADED_DEVICE or os.environ.get('V3_DEVICE', 'auto'),
             SECONDS, TARGET))
    print()
    print('%9s %10s %10s %10s %s' % ('cameras', 'each fps', 'slowest', 'total fps', 'verdict'))
    print('-' * 62)
    results = {}
    for cameras in range(1, MAX_CAMERAS + 1):
        ready = mp.Barrier(cameras + 1)
        go = mp.Event()
        rates = mp.Array('d', cameras)
        procs = [mp.Process(target=one_camera,
                            args=(i, frame, ready, go, SECONDS, rates, cameras))
                 for i in range(cameras)]
        for p in procs:
            p.start()
        ready.wait()        # every camera has loaded and warmed
        go.set()
        for p in procs:
            p.join()
        rates = list(rates)
        mean = statistics.mean(rates)
        worst = min(rates)
        # The slowest camera is the one that decides, not the average: a camera running under
        # target is losing falls whatever its neighbours manage.
        verdict = 'OK' if worst >= TARGET * 0.9 else 'BELOW TARGET -- losing falls'
        print('%9d %10.1f %10.1f %10.1f %s' % (cameras, mean, worst, sum(rates), verdict))
        results[cameras] = {'each': round(mean, 2), 'worst': round(worst, 2),
                            'total': round(sum(rates), 2), 'ok': worst >= TARGET * 0.9}

    usable = [c for c, r in results.items() if r['ok']]
    print()
    print('Cameras that all stay at or above %.0f fps: %s'
          % (TARGET, max(usable) if usable else 'none'))
    print()
    print('This replays one decoded frame, so it leaves out decoding N real RTSP streams --')
    print('real work, and on CPU it was 3% of the loop for one camera. These rates are a')
    print('ceiling, not a promise.')
    out = os.path.join(os.environ.get('RESULTS_DIR', os.path.dirname(os.path.abspath(__file__))),
                       'multi_camera_capacity.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'device': v3.LOADED_DEVICE, 'imgsz': v3.IMGSZ, 'target_fps': TARGET,
                   'source': '%dx%d' % (WIDTH, HEIGHT), 'results': results}, fh, indent=1)
    print('wrote %s' % out)


if __name__ == '__main__':
    main()
