"""What is URFD recall at the frame rate the system actually runs at?

Every accuracy number quoted for this system -- including the honest 72% on URFD -- is measured
offline, reading every frame of a 30fps file. The live loop does not work that way: it has no
stride and `CAP_PROP_BUFFERSIZE=1` hands back the newest frame, so the achieved frame rate *is*
the sampling rate. On the GPU that is 16.8-20 fps, so the classifier's 30-frame window spans
1.6-1.8x more real time than the 1.0s it was trained on.

SS38 measured this effect, but on GMDCSA24 val -- a set the system scores 15/15 on, where there
is almost no headroom for the damage to show. URFD is at 43/60, so if frame rate costs recall
this is where it becomes visible. Until this is run, "72%" is an offline claim, not a
deployment one.

Frames are selected by integer slot arithmetic (`int(i * target / source)`), which handles
non-integer rates like 18 fps exactly. A float "next due time" comparison was tried first and
silently dropped ~9% of frames even at target == source, because the equality landed on the
wrong side of the rounding -- it made the 30 fps column read 39/60 instead of the known 43/60,
which is what caught it.

Usage:
    V3_DEVICE=cuda python training/eval_urfd_framerate.py
    V3_DEVICE=cuda TARGET_FPS="30 20 15 10" python training/eval_urfd_framerate.py

A model trained for a different frame rate needs its window size too, since that is what keeps
the window covering the same amount of real time:

    V3_DEVICE=cuda V3_WINDOW_SIZE=15 TEST_MODEL_DIR=/path/to/model         TARGET_FPS="20 18 15" python training/eval_urfd_framerate.py
"""
import glob
import importlib.util
import json
import os

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'v3_fall_detection', os.path.join(ROOT, 'app', 'detection', 'v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

DATA = os.path.join(ROOT, 'training', 'data')
URFD = os.path.join(DATA, 'urfd')
RATES = [float(x) for x in os.environ.get('TARGET_FPS', '30 20 18 15 10').split()]


def alerts_at(det, path, target_fps):
    """Run the deployed detector over the clip, keeping only the frames a camera loop running
    at `target_fps` would have seen. Returns the number of alerts raised."""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    state = v3.V3MultiPersonFallState()
    i, last, count, last_slot = 0, None, 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # One frame per output slot: exact at target == source, and evenly spaced otherwise.
        slot = int(i * target_fps / src_fps)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        frame = frame[:, frame.shape[1] // 2:]   # RGB half; left half is a depth map
        hit = any(r[1] for r in v3.detect_v3_fall_multi(frame, state, det, config=None))
        label = 'fall' if hit else 'no_fall'
        if label != last:
            count += 1 if hit else 0
            last = label
    cap.release()
    return count


def main():
    model_dir = os.environ.get('TEST_MODEL_DIR', os.path.join(ROOT, 'models'))
    det = v3.V3PoseFallDetector(model_dir=model_dir)
    print(f'model: {model_dir}   window: {v3.WINDOW_SIZE} frames')
    falls = sorted(glob.glob(os.path.join(URFD, 'fall-*.mp4')))
    adls = sorted(glob.glob(os.path.join(URFD, 'adl-*.mp4')))
    print(f'{len(falls)} fall clips, {len(adls)} normal-activity clips\n')
    print(f'{"camera fps":>11s}  {"falls caught":>13s}  {"no false alarm":>15s}')

    out = {}
    for fps in RATES:
        caught = sum(alerts_at(det, p, fps) > 0 for p in falls)
        clean = sum(alerts_at(det, p, fps) == 0 for p in adls)
        out[fps] = {'falls': caught, 'adl_clean': clean}
        print(f'{fps:9.0f}    {caught:6d}/{len(falls)} ({caught / len(falls):4.0%})'
              f'   {clean:7d}/{len(adls)} ({clean / len(adls):4.0%})', flush=True)

    with open(os.path.join(DATA, 'urfd_framerate.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nThe GPU deployment runs at 16.8-20 fps; the production CPU server at ~2-4 fps.')


if __name__ == '__main__':
    main()
