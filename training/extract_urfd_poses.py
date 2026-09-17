"""Extract pose sequences from URFD, split so the test half stays clean.

Purpose: the remaining false alarms are deep bends -- URFD's `adl-28` is a man bending to tie
his shoes, and the same pattern has been tracked since SS20. Every previous attempt to fix it
with a decision rule failed (eight signals, SS33 and SS40), and the one thing that has ever
worked on this project is targeted hard negatives (SS28). URFD's ADL clips are exactly that
material: indoor, fixed camera, ordinary activity, no falls.

The split matters more than usual here. URFD is currently the only dataset nothing has been
tuned against, and it is where the honest accuracy number comes from. Training on all of it
would destroy that. So the ADL clips are split by index: even-numbered clips become training
negatives, odd-numbered ones are never touched and remain the held-out measure. **Fall clips
are never extracted at all** -- recall must stay measurable on all 60.

Frames are the RGB half only; the left half of each URFD mp4 is a depth map.

Usage:
    python training/extract_urfd_poses.py
"""
import os
import re

import cv2
import numpy as np

from yolopose_extractor import YoloPoseExtractor

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(HERE, 'data', 'urfd')
OUT_DIR = os.path.join(HERE, 'data', 'poses_urfd_adl_train')
MODEL_NAME = os.environ.get('YOLOPOSE_MODEL', 'yolo26s-pose.pt')


def clip_index(name):
    m = re.search(r'adl-(\d+)', name)
    return int(m.group(1)) if m else -1


def extract(path, extractor):
    cap = cv2.VideoCapture(path)
    seq = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = frame[:, frame.shape[1] // 2:]  # RGB half
        kpts, _ = extractor.extract_keypoints(frame)
        seq.append(kpts)
    cap.release()
    return np.stack(seq, axis=0) if seq else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    extractor = YoloPoseExtractor(MODEL_NAME)

    names = sorted(f for f in os.listdir(CLIPS) if f.startswith('adl-') and f.endswith('.mp4'))
    train = [n for n in names if clip_index(n) % 2 == 0]
    held_out = [n for n in names if clip_index(n) % 2 == 1]

    print(f'{len(names)} ADL clips -> {len(train)} for training, {len(held_out)} held out')
    print('held out (never trained on):', ', '.join(held_out))

    made = 0
    for name in train:
        out = os.path.join(OUT_DIR, 'urfd_' + name.replace('.mp4', '.npz'))
        if os.path.exists(out):
            made += 1
            continue
        seq = extract(os.path.join(CLIPS, name), extractor)
        if seq is None or len(seq) < 10:
            print('  skip (too short):', name)
            continue
        # label 0: these are activities of daily living, and the point is to teach the
        # classifier that a deep bend is not a fall.
        np.savez_compressed(out, keypoints=seq, label=0, subject=900 + clip_index(name))
        made += 1
        print(f'  [{made}] {name} -> {seq.shape[0]} frames')

    print(f'\nwrote {made} sequences to {OUT_DIR}')
    with open(os.path.join(HERE, 'data', 'urfd_heldout_adl.txt'), 'w') as f:
        f.write('\n'.join(held_out))
    print('held-out list written to training/data/urfd_heldout_adl.txt')


if __name__ == '__main__':
    main()
