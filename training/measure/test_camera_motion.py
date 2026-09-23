# -*- coding: utf-8 -*-
"""Is the camera fixed in the four real-fall clips, and does the subject's scale change?

`Test/13` is now the only real fall separating the deployed configuration from the 30-frame one,
so how much weight that clip can carry depends on what is in it. Watching it suggests a moving,
zooming camera and a subject who ends on hands and knees rather than on the ground -- neither of
which is what this system is built for: the classifier's features are torso-normalised
keypoints from a FIXED camera, and every training clip is a fixed camera.

Rather than assert that from a contact sheet, this measures two things per clip:

  camera motion    median magnitude of the dense optical flow over the frame, on the pixels
                   that are NOT part of the person. A fixed camera leaves the background still,
                   so this is near zero however much the subject moves.
  subject scale    the person's keypoint-box height as a fraction of the frame, first quarter
                   against last quarter. A tracking or zooming shot changes it; a fixed camera
                   changes it only as far as the person walks toward or away.

Frames are read at 8 fps, which is enough for both measures and keeps this cheap.
"""
import importlib.util
import os

import cv2
import numpy as np

ROOT = r'D:\project\PROJECT\Backend-Elderly-Surveillance-main'
os.chdir(ROOT)
os.environ.setdefault('V3_DEVICE', 'cuda')
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))


def measure(path, sample_fps=8.0):
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, i, last_slot = [], 0, -1
    while True:
        ok, f = cap.read()
        if not ok:
            break
        slot = int(i * sample_fps / src)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        frames.append(f)
    cap.release()

    heights, flows = [], []
    small = [cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY) for f in frames]
    for n, f in enumerate(frames):
        people = det.extract_all_keypoints(f)
        if people:
            kp = people[0][0]
            good = kp[kp[:, 2] > 0.3]
            heights.append(float(good[:, 1].max() - good[:, 1].min()) if len(good) >= 5 else None)
        else:
            heights.append(None)
        if n:
            flow = cv2.calcOpticalFlowFarneback(small[n - 1], small[n], None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(flow, axis=2)
            # The person is a minority of the frame; the median over the whole frame is
            # therefore dominated by background, which is what camera motion moves.
            flows.append(float(np.median(mag)))

    seen = [h for h in heights if h]
    q = max(1, len(heights) // 4)
    first = [h for h in heights[:q] if h]
    last = [h for h in heights[-q:] if h]
    return {
        'frames': len(frames),
        'background flow px/frame': float(np.median(flows)) if flows else 0.0,
        'subject height first quarter': float(np.median(first)) if first else 0.0,
        'subject height last quarter': float(np.median(last)) if last else 0.0,
        'scale change': ((float(np.median(last)) / float(np.median(first)))
                         if first and last and np.median(first) else 0.0),
        'person seen': '%.0f%%' % (100.0 * len(seen) / max(1, len(heights))),
    }


print('\n%-14s %8s %10s %8s %8s %7s %7s'
      % ('clip', 'frames', 'bg flow', 'h first', 'h last', 'scale', 'seen'))
for n in range(13, 18):
    p = 'Test/%d.mp4' % n
    m = measure(p)
    print('%-14s %8d %10.2f %8.3f %8.3f %6.2fx %7s'
          % (p, m['frames'], m['background flow px/frame'], m['subject height first quarter'],
             m['subject height last quarter'], m['scale change'], m['person seen']))
print('\nbg flow near 0 = fixed camera. scale change far from 1.00 = the framing moved.')
