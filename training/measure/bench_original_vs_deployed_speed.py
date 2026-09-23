# -*- coding: utf-8 -*-
"""Can the original detector actually run at the frame rate that makes it good?

Fed every frame, the original catches 54/60 URFD falls against the deployed detector's 41/60 --
and disabling its "the person vanished" rule changes that by nothing, so it is the classifier,
not a trick. That makes one number decisive: the original was only ever *measured* at 30 fps,
and the whole point of SS50 is that a detector is worth what it scores at the rate it can
actually sustain. At 15 fps the same detector collapses to 27/60.

So this times both detectors on the same 1080p frames, the way the camera loop calls them: one
full-resolution frame in, pose extraction and classification out. The answer is a frame rate
per detector on this machine, with the GPU available to the one that can use it.

Nothing here is a deployment proposal. It decides whether "the original is better at 30 fps" is
a real option or an arithmetic curiosity.
"""
import importlib.util
import os
import time

import cv2
import numpy as np

ROOT = r'D:\project\PROJECT\Backend-Elderly-Surveillance-main'
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

CLIP = 'Test/14.mp4'   # 1080p, what a real camera delivers
N = 120


def frames(n):
    cap = cv2.VideoCapture(CLIP)
    out = []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            cap.release()
            cap = cv2.VideoCapture(CLIP)
            continue
        out.append(f)
    cap.release()
    return out


FRAMES = frames(N)
print('%d frames of %dx%d\n' % (len(FRAMES), FRAMES[0].shape[1], FRAMES[0].shape[0]))


def time_detector(label, mod, det, state_factory, call):
    state = state_factory()
    for f in FRAMES[:10]:                      # warm up: first call builds the graph
        call(f, state, det)
    state = state_factory()
    t0 = time.perf_counter()
    for f in FRAMES:
        call(f, state, det)
    dt = (time.perf_counter() - t0) / len(FRAMES)
    print('  %-34s %6.1f ms/frame   %5.1f fps' % (label, dt * 1000, 1.0 / dt))
    return 1.0 / dt


# --- the original: MediaPipe, CPU only ------------------------------------------------------
spec = importlib.util.spec_from_file_location(
    'v3_orig', os.path.join(HERE, 'original', 'v3_original.py'))
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)
odet = orig.V3PoseFallDetector(model_dir=os.path.join(HERE, 'original', 'model_dir'))
orig_fps = time_detector('original (MediaPipe, CPU)', orig, odet, orig.V3FallDetectionState,
                         lambda f, s, d: orig.detect_v3_fall(f, s, d, config=None))

# --- the deployed one: YOLO26s-pose at 960, on the GPU --------------------------------------
os.environ['V3_DEVICE'] = 'cuda'
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
cur = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cur)
cdet = cur.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
cur_fps = time_detector('deployed (YOLO26s-pose 960, GPU)', cur, cdet,
                        cur.V3MultiPersonFallState,
                        lambda f, s, d: cur.detect_v3_fall_multi(f, s, d, config=None))

print('\nThis is detector time only. The live loop also decodes the frame, writes the clip')
print('buffer and talks to the database, which SS56 measured at roughly a third of the budget,')
print('so a sustainable camera rate is meaningfully below each figure above.')
print('\n  original needs 30 fps to reach 54/60 on URFD; at 15 fps it reaches 27/60.')
print('  deployed is pinned at 15 fps and reaches 41/60 there.')
