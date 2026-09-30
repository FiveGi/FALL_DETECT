# -*- coding: utf-8 -*-
"""A dark clip side by side with what the detector actually sees after the shadows are lifted.

The still comparison already in the repo shows one frame. What it cannot show is the thing that
matters: the keypoints in a dark room do not merely sit in the wrong place, they JITTER, and
jitter is read as fast motion, which is what a fall looks like. That is only visible in motion,
which is why this writes a clip rather than another picture.

Left is the frame as the camera sends it, right is the frame the pose model is given, both with
the skeleton that was actually found on them.

Usage:
    V3_DEVICE=cuda python training/measure/render_preprocess_clip.py \
        training/data/urfd/adl-22-cam0.mp4 test_result/preprocess_demo.mp4
"""
import importlib.util
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
TILE_W = int(os.environ.get('TILE_W', 480))
RGB_HALF = os.environ.get('RGB_HALF', '1') == '1'


def skeleton(img, people, colour):
    out = img.copy()
    h, w = out.shape[:2]
    for kpts, _hip in people:
        pts = [(int(kpts[i, 0] * w), int(kpts[i, 1] * h)) for i in range(17)]
        for a, b in EDGES:
            if kpts[a, 2] > 0.2 and kpts[b, 2] > 0.2:
                cv2.line(out, pts[a], pts[b], colour, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            if kpts[i, 2] > 0.2:
                cv2.circle(out, p, 3, colour, -1, cv2.LINE_AA)
    return out


def banner(width, text, colour=(30, 30, 30)):
    strip = np.zeros((28, width, 3), np.uint8)
    strip[:] = colour
    cv2.putText(strip, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    return strip


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    source, dest = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)

    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = None
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # URFD stores depth and colour side by side; only the right half is a picture.
        if RGB_HALF and frame.shape[1] > frame.shape[0] * 1.7:
            frame = frame[:, frame.shape[1] // 2:]
        after, ops = v3.preprocess_frame(frame)

        before_people = det.extract_all_keypoints(frame.copy())
        after_people = det.extract_all_keypoints(after.copy())

        h, w = frame.shape[:2]
        size = (TILE_W, int(h * TILE_W / w))
        left = cv2.resize(skeleton(frame, before_people, (0, 200, 255)), size)
        right = cv2.resize(skeleton(after, after_people, (0, 255, 0)), size)
        left = np.vstack([banner(size[0], 'as the camera sends it   luminance %.0f'
                                 % v3.frame_luminance(frame)), left])
        right = np.vstack([banner(size[0], 'what the model is given   luminance %.0f   %s'
                                  % (v3.frame_luminance(after), '+'.join(ops) or 'untouched'),
                                  (0, 90, 0) if ops else (30, 30, 30)), right])
        pair = np.hstack([left, right])
        if writer is None:
            writer = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                     (pair.shape[1], pair.shape[0]))
        writer.write(pair)
        frames += 1
    cap.release()
    if writer is not None:
        writer.release()
    print('wrote %s  (%d frames at %.0f fps)' % (dest, frames, fps))
    return 0


if __name__ == '__main__':
    sys.exit(main())
