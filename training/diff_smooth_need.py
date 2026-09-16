"""List exactly which URFD clips change when the smoothing rule changes.

A summary count ("+5 falls, -3 clean") cannot be verified. This names the clips so each one
can be watched and judged, which is the only way to know whether the extra alerts are real
catches or noise.
"""
import glob, importlib.util, json, os, sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'v3_fall_detection', os.path.join(ROOT, 'app', 'detection', 'v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3)
CLIPS = os.path.join(ROOT, 'training', 'data', 'urfd')


def run(det, path):
    st = v3.V3MultiPersonFallState()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    i, last, times = 0, None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = f[:, f.shape[1] // 2:]
        res = v3.detect_v3_fall_multi(f, st, det, config=None)
        lbl = 'fall' if any(r[1] for r in res) else 'no_fall'
        if lbl != last:
            if lbl == 'fall':
                times.append(round(i / fps, 1))
            last = lbl
        i += 1
    cap.release()
    return times


det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
out = {}
for p in sorted(glob.glob(os.path.join(CLIPS, '*.mp4'))):
    out[os.path.basename(p)] = run(det, p)
json.dump(out, open(os.path.join(ROOT, 'training', 'data',
                                 f'urfd_alerts_sn{v3.SMOOTH_NEED}.json'), 'w'), indent=1)
print(f'SMOOTH_NEED={v3.SMOOTH_NEED}: '
      f'{sum(1 for k, v in out.items() if k.startswith("fall") and v)}/60 falls alerted, '
      f'{sum(1 for k, v in out.items() if k.startswith("adl") and not v)}/40 adl clean')
