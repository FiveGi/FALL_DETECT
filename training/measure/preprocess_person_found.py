# -*- coding: utf-8 -*-
"""Does V3_PREPROCESS make the pose model find the person more often?

This is the question that decides whether frame preprocessing belongs in the pipeline at all.
Everything downstream -- the window, the classifier, the alerting rule -- only ever sees frames
where a person was found. If lifting the shadows does not raise the person-found rate, nothing
further downstream can improve, and the 44 ms a dark frame costs buys nothing.

It is measured per clip and split by how dark the clip is, because the two groups are asking
different questions:

  dark clips    does it help where it is supposed to?
  lit clips     does it leave alone what it is supposed to leave alone? ("auto" should read
                these as bright and pass them through untouched, so the two columns must come
                out identical -- if they do not, the gate is not working.)

Usage:
    V3_DEVICE=cuda V3_IMGSZ=320 python training/measure/preprocess_person_found.py
    V3_DEVICE=cuda V3_IMGSZ=320 SETTINGS=off,auto,clahe,gamma python .../preprocess_person_found.py
"""
import glob
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

OUT_DIR = os.environ.get('RESULTS_DIR', os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(OUT_DIR, 'preprocess_person_found.json')
SETTINGS = [s.strip() for s in os.environ.get('SETTINGS', 'off,auto').split(',') if s.strip()]
# Every 4th frame. The question is a rate over a clip, not a per-frame answer, and the pose
# pass is the entire cost of asking.
FRAME_STEP = int(os.environ.get('FRAME_STEP', 4))
DARK_BELOW = float(os.environ.get('DARK_BELOW', 70))


def load_detector(setting):
    """A fresh module per setting: V3_PREPROCESS is read at import, like every other knob.

    "auto@40" runs auto with V3_PREPROCESS_DARK_BELOW=40. The threshold is the setting's most
    important parameter -- the first measurement showed it helping below 30 and hurting
    between there and 70 -- so it has to be sweepable without editing anything.
    """
    name, _, dark_below = setting.partition('@')
    os.environ['V3_PREPROCESS'] = name
    os.environ['V3_PREPROCESS_DARK_BELOW'] = dark_below or str(DARK_BELOW)
    spec = importlib.util.spec_from_file_location(
        'v3_%s' % setting.replace(',', '_').replace('@', '_'),
        os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))


def clips():
    """URFD's ADL clips, which is where every dark clip in this corpus is, plus the Test clips.

    URFD fall clips are deliberately left out: not one of them is dark (the darkest is at
    luminance 76), so including them would pad the result with clips the setting cannot affect.
    """
    out = [('urfd_adl', p, True) for p in sorted(glob.glob('training/data/urfd/adl-*.mp4'))]
    out += [('test', 'Test/%d.mp4' % n, False) for n in range(1, 18)]
    return [(g, p, h) for g, p, h in out if os.path.exists(p)]


def measure(mod, det, path, rgb_half):
    """-> (frames read, frames with a person, mean luminance of the frames as read)."""
    cap = cv2.VideoCapture(path)
    seen, total, touched, lums, i = 0, 0, 0, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % FRAME_STEP:
            continue
        if rgb_half:
            frame = frame[:, frame.shape[1] // 2:]
        lums.append(mod.frame_luminance(frame))
        # How many frames the setting actually altered. Without this the "lit clips must be
        # identical" control cannot be read: the darkness gate is per FRAME and the grouping is
        # by clip MEAN, so a clip averaging 100 can still contain frames below the threshold,
        # and a difference in the lit row could mean either that or a broken gate.
        if mod.preprocess_frame(frame)[1]:
            touched += 1
        # extract_all_keypoints applies the preprocessing itself, which is the point: this
        # measures the real entry point, not a reimplementation of it.
        total += 1
        if det.extract_all_keypoints(frame):
            seen += 1
    cap.release()
    return total, seen, touched, float(np.mean(lums)) if lums else 0.0


def main():
    results = {}
    if os.path.exists(OUT):
        with open(OUT, encoding='utf-8') as fh:
            results = json.load(fh)

    todo = [s for s in SETTINGS
            if len(results.get(s, {})) < len(clips())]
    for setting in todo:
        mod, det = load_detector(setting)
        print('=== V3_PREPROCESS=%s  (imgsz %d) ===' % (setting, mod.IMGSZ), flush=True)
        row = results.get(setting, {})
        for group, path, rgb_half in clips():
            key = '%s/%s' % (group, os.path.basename(path))
            if key in row:
                continue
            total, seen, touched, lum = measure(mod, det, path, rgb_half)
            row[key] = {'frames': total, 'person_found': seen, 'touched': touched,
                        'luminance': round(lum, 1)}
            results[setting] = row
            with open(OUT, 'w', encoding='utf-8') as fh:
                json.dump(results, fh, indent=1)
            print('  %-28s %4d/%4d frames  luminance %5.1f'
                  % (key, seen, total, lum), flush=True)

    report(results)


def report(results):
    settings = [s for s in SETTINGS if s in results]
    if not settings:
        return
    base = results[settings[0]]
    keys = sorted(base)
    groups = {'dark  (luminance < %g)' % DARK_BELOW: [k for k in keys
                                                      if base[k]['luminance'] < DARK_BELOW],
              'lit   (luminance >= %g)' % DARK_BELOW: [k for k in keys
                                                       if base[k]['luminance'] >= DARK_BELOW]}
    print()
    print('%-26s %6s | %s' % ('group', 'clips', '  '.join('%-18s' % s for s in settings)))
    print('-' * (36 + 20 * len(settings)))
    for label, ks in groups.items():
        if not ks:
            continue
        cells = []
        for s in settings:
            found = sum(results[s][k]['person_found'] for k in ks if k in results[s])
            total = sum(results[s][k]['frames'] for k in ks if k in results[s])
            hit = sum(results[s][k].get('touched', 0) for k in ks if k in results[s])
            cells.append('%-22s' % ('%d/%d %.1f%% (%d altered)'
                                    % (found, total, 100.0 * found / max(total, 1), hit)))
        print('%-26s %6d | %s' % (label, len(ks), '  '.join(cells)))
    print()
    print('per clip, dark clips only:')
    print('%-28s %8s | %s' % ('clip', 'luminance', '  '.join('%-14s' % s for s in settings)))
    for k in groups.get('dark  (luminance < %g)' % DARK_BELOW, []):
        cells = []
        for s in settings:
            r = results[s].get(k)
            cells.append('%-14s' % ('%d/%d' % (r['person_found'], r['frames']) if r else '-'))
        print('%-28s %8.1f | %s' % (k, base[k]['luminance'], '  '.join(cells)))
    print()
    print('The lit row is the control: "auto" must leave those frames untouched, so its number')
    print('has to equal "off" exactly. A difference there means the darkness gate is not working.')


if __name__ == '__main__':
    main()
