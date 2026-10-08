"""plan_v5 T2e: deterministic checks for the fall-path people cap (V3_NUM_POSES), no model or video needed.

1. extract_all_keypoints keeps the NUM_POSES highest-confidence people, in confidence order (a fake pose result).
2. Eight people standing still: the hip tracker keeps exactly eight stable ids (no churn) at cap 8.
3. A crowd of 13 with some people missed each frame: retained track states stay bounded, never growing without limit.
4. Two people crossing paths: no crash, and every live track still gets a result each frame.
Run: python tests/test_people_cap.py   (re-runs itself at V3_NUM_POSES=8)
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('V3_NUM_POSES', '4')
for k in ('V3_ROI_IMGSZ', 'V3_TRACK_REASSOC', 'V3_TRACKER'):
    os.environ.pop(k, None)
os.environ['V3_PREPROCESS'] = 'none'
sys.path.insert(0, ROOT)
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location('v3', os.path.join(ROOT, 'app', 'detection', 'v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
CAP = v3.NUM_POSES
fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + '[cap %d] %s' % (CAP, name) + (' -- %s' % (detail,) if detail else ''))
    if not cond:
        fails.append(name)


class _T:
    def __init__(self, a): self.a = np.asarray(a, dtype=np.float32)
    def cpu(self): return self
    def numpy(self): return self.a
    def __getitem__(self, i): return _T(self.a[i])
    def __len__(self): return len(self.a)


class _Result:
    """The fields of an ultralytics pose result that extract_all_keypoints reads."""
    def __init__(self, people):   # people: list of (conf, hip_x, hip_y)
        xy = np.zeros((len(people), 17, 2), np.float32)
        for i, (_, x, y) in enumerate(people):
            xy[i, :, 0], xy[i, :, 1] = x, y
        self.keypoints = type('K', (), {'xy': _T(xy), 'conf': _T(np.ones((len(people), 17)))})()
        self.boxes = type('B', (), {'conf': _T([c for c, _, _ in people]),
                                    'xyxy': _T([[x - 5, y - 5, x + 5, y + 5] for _, x, y in people])})()


class _Pose:
    def __init__(self): self.next = []
    def predict(self, *a, **k): return [_Result(self.next)]


det = v3.V3PoseFallDetector.__new__(v3.V3PoseFallDetector)
det.pose_model, det.device = _Pose(), 'cpu'
frame = np.zeros((100, 200, 3), np.uint8)

# 1. which people survive the cap
det.pose_model.next = [(0.9, 10, 50), (0.35, 30, 50), (0.8, 50, 50), (0.31, 70, 50), (0.6, 90, 50), (0.5, 110, 50)]
kept = [round(float(h[0] * 200)) for _, h in det.extract_all_keypoints(frame)]
want = [x for _, x, _ in sorted(det.pose_model.next, key=lambda p: -p[0])][:CAP]
check('keeps the %d most confident, in confidence order' % min(CAP, 6), kept == want, (kept, want))


def stream(frames_people):
    """Feed lists of (x, y) hip positions (0..1) through the tracker; return per-frame track ids and peak tracks."""
    t = v3.PersonTracker()
    ids, peak = [], 0
    for people in frames_people:
        dets = [(np.zeros((17, 3), np.float32), np.array([x, y], np.float32)) for x, y in people[:CAP]]
        out = t.update(dets)
        ids.append(sorted(tid for tid, _, seen in out if seen))
        peak = max(peak, len(t.tracks))
    return ids, peak


# 2. eight still people
still = [[(0.1 + 0.1 * i, 0.5) for i in range(8)]] * 40
ids, peak = stream(still)
n = min(CAP, 8)
check('%d still people -> %d stable ids, no churn' % (n, n), all(f == ids[0] for f in ids) and len(ids[0]) == n
      and peak == n, (ids[0], peak))

# 3. crowd of 13, a different 3 people missed every frame (detector flicker)
rng = np.random.default_rng(0)
base = [(0.05 + 0.07 * i, 0.5) for i in range(13)]
crowd = []
for f in range(120):
    drop = set(rng.choice(13, 3, replace=False))
    crowd.append([p for i, p in enumerate(base) if i not in drop])
ids, peak = stream(crowd)
bound = CAP * (v3.MAX_MISSED_FRAMES + 1)
check('crowd of 13 with flicker -> retained tracks bounded (peak %d <= %d)' % (peak, bound), peak <= bound, peak)

# 4. two people crossing
cross = [[(0.2 + 0.006 * f, 0.5), (0.8 - 0.006 * f, 0.5)] for f in range(100)]
ids, peak = stream(cross)
check('two people crossing -> 2 seen tracks every frame (count only, not identity continuity)', all(len(f) == 2 for f in ids), peak)

if CAP == 4 and not fails:
    r = subprocess.run([sys.executable, __file__], env=dict(os.environ, V3_NUM_POSES='8'))
    if r.returncode:
        fails.append('cap 8 run')
print('RESULT', 'FAIL' if fails else 'PASS', fails)
sys.exit(1 if fails else 0)
