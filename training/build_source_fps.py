# -*- coding: utf-8 -*-
"""Native frame rate of every training pose file -> data/source_fps.json.

The classifier must see motion at the rate the camera loop feeds it (8 fps on the CPU server).
A blanket TEMPORAL_STRIDE=2 assumed every source is 30 fps; CAUCAFall is 20 fps, so it was
trained at 10 fps while the rest ran at ~15. dataset.RESAMPLE_FPS needs each file's real rate,
read here from the evidence available for each source:
  - OF-ItW / OOPS: the segment's start and end seconds are in the file name; rate = frames / span.
  - GMDCSA24: the raw videos are still on disk; rate read from each file.
  - CAUCAFall: 20 fps -- the .avi beside every image sequence says so, and the png count equals
    the pose file's length (Subject.1 / Fall backwards: 126 and 126).
  - FallVision: NOT uniform. Its mask videos (Harvard Dataverse doi:10.7910/DVN/75QPKK, CC0) run at
    30, 24, 15, 60, 60.04, 29.75 and 120 fps, and the keypoint CSVs match their frame counts (95 of
    108 checked). Per-file rates come from FALLVISION_FPS_JSON, made by downloading every mask
    archive and reading each video (scratchpad probe_fps.py); a file it does not cover is unknown.
  - GMDCSA24 files whose raw video is not on disk: the median of the same subject's known files
    (one camera per subject, 29.6-29.9 fps).
Files with an explicit `fps` key (new extractions) need no entry.
"""
import glob
import json
import os
import re

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
FALLVISION_FPS_JSON = os.environ.get('FALLVISION_FPS_JSON', os.path.join(DATA, 'fallvision_fps.json'))


def ofitw_fps(name, n):
    m = re.search(r'_(\d+)-(\d+)_(\d+)-(\d+)\.npz$', name)
    if not m:
        return None
    a = int(m.group(1)) + int(m.group(2)) / 1000.0
    b = int(m.group(3)) + int(m.group(4)) / 1000.0
    return n / (b - a) if b > a else None


def main():
    out, unknown, pending, excluded = {}, [], [], []
    fv = json.load(open(FALLVISION_FPS_JSON)) if os.path.exists(FALLVISION_FPS_JSON) else {}
    fv_count_mismatch = 0
    raw = {}
    for d in ('gmdcsa24_fall_raw', 'gmdcsa24_adl_raw_val', 'gmdcsa24_adl_raw_train50'):
        for p in glob.glob(os.path.join(DATA, d, '*.mp4')):
            raw[os.path.splitext(os.path.basename(p))[0]] = cv2.VideoCapture(p).get(cv2.CAP_PROP_FPS)
    for pose_dir in ('poses_ofitw_yolopose_matched', 'poses_gmdcsa24_v2', 'poses_caucafall_yolopose',
                     'poses_fallvision', 'poses_omnifall_adl', 'poses_urfd_adl_train',
                     'poses_realtest_v1', 'poses_yolopose'):
        for p in sorted(glob.glob(os.path.join(DATA, pose_dir, '*.npz'))):
            name = os.path.basename(p)
            n = int(np.load(p, allow_pickle=True)['keypoints'].shape[0])
            if 'ofitw' in pose_dir or 'omnifall_adl' in pose_dir:
                fps = ofitw_fps(name, n)
            elif 'gmdcsa' in pose_dir or pose_dir == 'poses_yolopose':
                fps = raw.get(os.path.splitext(name)[0])
            elif 'cauca' in pose_dir:
                fps = 20.0
            elif 'fallvision' in pose_dir:
                # fv_<archive>_ke(y|t)points_csv_<video>_keypoints.npz
                m = re.match(r'fv_(.+?)_ke[yt]points_csv_(.+)_keypoints\.npz$', name)
                hit = fv.get('%s/%s' % (m.group(1), m.group(2))) if m else None
                fps = hit[0] if hit else None
                if hit and hit[1] != n:
                    fv_count_mismatch += 1
                    # The pose file is SHORTER than its video in all 800 such files (median 0.92
                    # of the frames, as low as 0.17): frames were dropped, most likely where nobody
                    # was detected, and where is not recorded. The timeline is compressed by an
                    # unknown pattern, so beyond 5% missing the file has no usable time base.
                    if n / float(hit[1]) < 0.95:
                        excluded.append('%s/%s' % (pose_dir, name))
                        continue
            else:
                fps = None
            if fps is None and ('gmdcsa' in pose_dir or pose_dir == 'poses_yolopose'):
                # Resolved after every file is read (second pass below): the subject's median.
                pending.append(('%s/%s' % (pose_dir, name), name.split('_')[0]))
                continue
            if fps is None or not (5 <= fps <= 121):
                unknown.append('%s/%s' % (pose_dir, name))
                continue
            out['%s/%s' % (pose_dir, name)] = round(float(fps), 3)
    for key, subj in pending:
        known = [v for k, v in out.items() if k.split('/')[0] in ('poses_gmdcsa24_v2', 'poses_yolopose')
                 and k.split('/')[1].split('_')[0] == subj]
        if known:
            out[key] = sorted(known)[len(known) // 2]
        else:
            unknown.append(key)
    print('FallVision files whose video frame count differs from the pose file: %d' % fv_count_mismatch)
    print('excluded (no usable time base): %d' % len(excluded))
    json.dump({'fps': out, 'unknown': unknown, 'excluded': excluded,
               'fallvision_count_mismatch': fv_count_mismatch},
              open(os.path.join(DATA, 'source_fps.json'), 'w'), indent=0)
    by = {}
    for k, v in out.items():
        by.setdefault(k.split('/')[0], []).append(v)
    for d, v in by.items():
        print('%-30s n=%5d  fps min %.1f median %.1f max %.1f' % (d, len(v), min(v), sorted(v)[len(v) // 2], max(v)))
    print('unknown:', len(unknown), unknown[:5])


if __name__ == '__main__':
    main()
