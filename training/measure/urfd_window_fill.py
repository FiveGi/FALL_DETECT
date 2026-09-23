# -*- coding: utf-8 -*-
"""For each URFD clip, which frames contain a person -- once, so any (window, rate) follows.

Seven URFD fall clips score exactly 0.00 under the deployed configuration, and it is not the
classifier rejecting them. They are the ceiling camera (`cam1`) on the standing falls, where
the room is empty for two thirds of the clip and the person walks into view only as they land:
43 sampled frames at 15 fps, a person in 14 of them, against a window that needs 15. The clip
ends before the classifier can score anything at all.

That matters for every comparison in this project, because a longer window at a lower rate
needs *more* real time of visible person before it can produce a number -- 30 frames at 15 fps
is 2.0 seconds, which is longer than several URFD clips. Some of what looks like one model
beating another is one of them being able to score at all.

Person detection does not depend on the sampling rate, so this runs the pose extractor over
every frame once and stores a bitmap. Whether a given (window, target fps) could ever fill on a
given clip is then arithmetic, not another GPU pass.

Writes after every clip and resumes.
"""
import glob
import importlib.util
import json
import os

import cv2

ROOT = r'D:\project\PROJECT\Backend-Elderly-Surveillance-main'
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

os.environ.setdefault('V3_DEVICE', 'cuda')
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

OUT = os.path.join(HERE, 'urfd_window_fill.json')
det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))


def person_bitmap(path):
    """-> (source fps, '1'/'0' per frame of the clip: was anybody detected)."""
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    bits = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        bits.append('1' if det.extract_all_keypoints(f[:, f.shape[1] // 2:]) else '0')
    cap.release()
    return src, ''.join(bits)


data = json.load(open(OUT)) if os.path.exists(OUT) else {}
for p in sorted(glob.glob('training/data/urfd/*.mp4')):
    name = os.path.basename(p)
    if name in data:
        continue
    src, bits = person_bitmap(p)
    data[name] = [src, bits]
    json.dump(data, open(OUT, 'w'), indent=1)

print('measured %d clips' % len(data))


def fills(name, window, target_fps):
    """Could a `window`-frame buffer ever fill with observed frames at this rate?"""
    src, bits = data[name]
    kept, last_slot = [], -1
    for i, b in enumerate(bits):
        slot = int(i * target_fps / src)
        if slot == last_slot:
            continue
        last_slot = slot
        kept.append(b)
    return sum(1 for b in kept if b == '1') >= window


falls = sorted(n for n in data if n.startswith('fall-'))
print('\nURFD fall clips where the window can never fill, so no score is possible:\n')
print('  %-28s %s' % ('configuration', 'clips out of %d' % len(falls)))
for label, window, fps in [('deployed 15 frames @ 15 fps', 15, 15),
                           ('30 frames @ 24 fps', 30, 24),
                           ('30 frames @ 15 fps (original)', 30, 15),
                           ('30 frames @ 30 fps (original, every frame)', 30, 30)]:
    bad = [n for n in falls if not fills(n, window, fps)]
    print('  %-28s %2d   %s' % (label, len(bad),
                                ' '.join(n.replace('.mp4', '') for n in bad[:9])))
