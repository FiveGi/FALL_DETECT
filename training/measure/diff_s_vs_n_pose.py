# -*- coding: utf-8 -*-
"""Do yolo26n-pose and yolo26s-pose actually produce different keypoints?

yolo26n-pose was rejected on CPU because it caught fewer falls at matched cost. The objection
to that result is fair: the classifier was trained on keypoints extracted with yolo26s-pose, so
a different extractor is a train/serve mismatch and n was handicapped before it started.

Retraining on n-extracted features is the clean answer and cannot be done -- the source videos
for 97% of the training frames were deleted after extraction (OOPS and CAUCAFall), and the
largest source, FallVision, is external keypoints that neither backend produced.

So this measures the premise instead. Both extractors run over the same frames of the same
clips, and the keypoints are compared the way the classifier sees them: normalised to the
frame, and then torso-normalised, which is what dataset.py does before the model ever sees a
number. Three things matter:

  found agreement   how often the two disagree about whether anybody is there at all. A missed
                    person is a zero row, which is a far bigger feature change than a slightly
                    different elbow.
  raw distance      median per-keypoint distance in frame widths, on frames where both found
                    someone.
  torso distance    the same after torso normalisation -- the scale the classifier works in.

If these are small, the mismatch objection does not hold and the CPU comparison stands. If they
are large, n was handicapped and the comparison has to be redone some other way.

Run at the input size each would deploy with, because that is part of the extractor.
"""
import os
import sys

import cv2
import numpy as np

ROOT = r'D:\project\PROJECT\Backend-Elderly-Surveillance-main'
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'training'))
from ultralytics import YOLO  # noqa: E402

CONF = 0.3          # what the runtime uses, not the extractor script's 0.5
IMGSZ = int(os.environ.get('DIFF_IMGSZ', 320))
CLIPS = [
    'training/data/gmdcsa24_fall_raw/s1_Fall_01.mp4',
    'training/data/gmdcsa24_fall_raw/s2_Fall_10.mp4',
    'training/data/gmdcsa24_adl_raw_val/s1_ADL_01.mp4',
    'Test/14.mp4',
    'Test/16.mp4',
    'training/data/urfd/fall-05-cam0.mp4',
    'training/data/urfd/adl-11-cam0.mp4',
]
LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 5, 6, 11, 12


def kpts(model, frame):
    """-> ((17,3) normalised [x, y, conf], found)."""
    h, w = frame.shape[:2]
    r = model.predict(frame, verbose=False, conf=CONF, classes=[0], device='cuda',
                      imgsz=IMGSZ)[0]
    if r.keypoints is None or len(r.keypoints.xy) == 0:
        return np.zeros((17, 3), dtype=np.float32), False
    i = int(np.argmax(r.boxes.conf.cpu().numpy()))
    xy = r.keypoints.xy[i].cpu().numpy()
    cf = (r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None
          else np.ones(17, dtype=np.float32))
    out = np.zeros((17, 3), dtype=np.float32)
    out[:, 0] = xy[:, 0] / w
    out[:, 1] = xy[:, 1] / h
    out[:, 2] = cf
    return out, True


def torso_normalise(k):
    """Centre on the hips and scale by torso length -- what the classifier is fed."""
    hip = (k[LEFT_HIP, :2] + k[RIGHT_HIP, :2]) / 2.0
    sho = (k[LEFT_SHOULDER, :2] + k[RIGHT_SHOULDER, :2]) / 2.0
    scale = np.linalg.norm(sho - hip)
    if scale < 1e-6:
        return None
    return (k[:, :2] - hip) / scale


s_model = YOLO(os.path.join(ROOT, 'models', 'yolo26s-pose.pt'))
n_model = YOLO(os.path.join(ROOT, 'models', 'yolo26n-pose.pt'))
print('yolo26s-pose vs yolo26n-pose, conf %.2f, imgsz %d\n' % (CONF, IMGSZ))

tot_frames = agree_found = only_s = only_n = both = 0
raw_d, torso_d = [], []
for clip in CLIPS:
    cap = cv2.VideoCapture(clip)
    rgb_half = '/urfd/' in clip.replace('\\', '/')
    i = 0
    while i < 60:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        if rgb_half:
            f = f[:, f.shape[1] // 2:]
        ks, fs = kpts(s_model, f)
        kn, fn = kpts(n_model, f)
        tot_frames += 1
        if fs == fn:
            agree_found += 1
        elif fs:
            only_s += 1
        else:
            only_n += 1
        if not (fs and fn):
            continue
        both += 1
        good = (ks[:, 2] > 0.3) & (kn[:, 2] > 0.3)
        if good.sum() >= 5:
            raw_d.append(float(np.median(np.linalg.norm(ks[good, :2] - kn[good, :2], axis=1))))
        ts, tn = torso_normalise(ks), torso_normalise(kn)
        if ts is not None and tn is not None and good.sum() >= 5:
            torso_d.append(float(np.median(np.linalg.norm(ts[good] - tn[good], axis=1))))
    cap.release()
    print('  %-46s %d frames' % (clip, i))

print('\nframes compared            %d' % tot_frames)
print('agree on whether anyone is there  %d (%.1f%%)' % (agree_found, 100.0 * agree_found / tot_frames))
print('  only yolo26s found a person     %d' % only_s)
print('  only yolo26n found a person     %d' % only_n)
print('\non the %d frames where both found someone:' % both)
print('  median keypoint distance, frame widths   %.4f  (p90 %.4f)'
      % (np.median(raw_d), np.percentile(raw_d, 90)))
print('  median distance after torso normalising  %.4f  (p90 %.4f)'
      % (np.median(torso_d), np.percentile(torso_d, 90)))
print('\nFor scale: the torso itself is 1.0 in the second measure, so 0.05 there is five per')
print('cent of a torso length -- roughly the width of a wrist.')
