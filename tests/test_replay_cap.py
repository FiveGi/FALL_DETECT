"""plan_v5 T2b: replay loaders cut each cached frame to V3_NUM_POSES without misreading later frames.

A packed cache stores every frame's people back to back with a per-frame count. Cutting a frame
to the cap must still advance the read offset by the frame's ORIGINAL count; advancing by the cut
count would silently feed frame 2 the leftover people of frame 1.
Run: V3_NUM_POSES=4 python tests/test_replay_cap.py  (it re-runs itself for cap 99)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cap = int(os.environ.get('V3_NUM_POSES', 4))
sys.path.insert(0, os.path.join(ROOT, 'training', 'measure'))
os.chdir(ROOT)
import replay_classifiers  # noqa: E402
import replay_owner_segments  # noqa: E402

counts = [6, 2, 5, 0, 1]
kpts = np.zeros((sum(counts), 17, 3), np.float32)
for frame, n in enumerate(counts):          # tag each person: x of keypoint 0 = frame*10 + rank
    start = sum(counts[:frame])
    for rank in range(n):
        kpts[start + rank, 0, 0] = frame * 10 + rank
path = os.path.join(tempfile.mkdtemp(), 'c.npz')
np.savez_compressed(path, counts=np.array(counts, np.int16), kpts=kpts, t=np.arange(len(counts), dtype=np.float32))

fails = []
for name, frames in (('replay_classifiers.load_stream', replay_classifiers.load_stream(path)),
                     ('replay_owner_segments.frames', replay_owner_segments.frames(path)[0])):
    want = [[f * 10 + r for r in range(min(n, cap))] for f, n in enumerate(counts)]
    got = [[int(p[0, 0]) for p in fr] for fr in frames]
    ok = got == want
    print(('PASS ' if ok else 'FAIL ') + '%s cap %d' % (name, cap), '' if ok else (got, want))
    if not ok:
        fails.append(name)

if cap == 4 and not fails:
    r = subprocess.run([sys.executable, __file__], env=dict(os.environ, V3_NUM_POSES='99'))
    if r.returncode:
        fails.append('cap 99')
print('RESULT', 'FAIL' if fails else 'PASS', fails)
sys.exit(1 if fails else 0)
