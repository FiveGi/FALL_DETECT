# -*- coding: utf-8 -*-
"""Cut `Test/1`-`12` into one-incident segments, so twelve clips that can only be counted
become a test set that can be scored.

Today those twelve are reported as a number of alerts and never as accuracy, because there is
no per-incident ground truth and a denominator would have to be invented. They are also the
largest body of real, non-lab footage this project has: roughly three times the four single-fall
clips it currently scores on.

**A shot is not an incident, and this script does not pretend otherwise.** It proposes
boundaries from hard cuts and writes a contact sheet for each one so a person can see what is
in it. Nothing here becomes ground truth until it has been looked at -- one incident can span
two shots, a shot can contain nothing, and a compilation's end card is a shot like any other.

The detector's own alerts are attached to each segment, not to label them but so the review can
start with the segments where the system did something.

Output, in `test_result/incidents/`:
    incidents.json    proposed segments, their length, and the alerts inside them
    sheets/*.jpg      six frames across each segment, for the review

Usage:
    V3_DEVICE=cuda python training/measure/cut_compilations.py
    MIN_SECONDS=1.5 SHEETS=0 python training/measure/cut_compilations.py
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'training'))
import test_clips as tc          # noqa: E402
from shot_cuts import shots      # noqa: E402

OUT_DIR = os.environ.get('INCIDENT_DIR', os.path.join(ROOT, 'test_result', 'incidents'))
SHEET_DIR = os.path.join(OUT_DIR, 'sheets')
# Shorter than this is a dissolve, a flash frame or a transition, not something anybody filmed.
MIN_SECONDS = float(os.environ.get('MIN_SECONDS', 1.2))
WRITE_SHEETS = os.environ.get('SHEETS', '1') == '1'
SHEET_COLS, SHEET_ROWS = 3, 2
THUMB_W = 320


def contact_sheet(path, start, end, out_jpg, caption):
    """Six frames spread across the segment, captioned. The point is to be looked at."""
    cap = cv2.VideoCapture(path)
    idxs = np.linspace(start, max(start, end - 1), SHEET_COLS * SHEET_ROWS).astype(int)
    tiles = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((180, THUMB_W, 3), np.uint8)
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (THUMB_W, max(1, int(h * THUMB_W / w))))
        cv2.putText(frame, 'f%d' % i, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(frame)
    cap.release()
    height = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    rows = [np.hstack(tiles[r * SHEET_COLS:(r + 1) * SHEET_COLS]) for r in range(SHEET_ROWS)]
    sheet = np.vstack(rows)
    banner = np.zeros((26, sheet.shape[1], 3), np.uint8)
    cv2.putText(banner, caption, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.imwrite(out_jpg, np.vstack([banner, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 80])


WITH_ALERTS = os.environ.get('ALERTS', '1') == '1'


def alert_times(path):
    """Seconds at which the deployed detector raises an alert in this clip, or None if it was
    not asked. Not a label -- a place to start reading.

    ALERTS=0 skips it, which is what to use when the GPU is busy with a measurement: the cuts
    and the contact sheets are the slow, reviewable part and they need no detector.
    """
    if not WITH_ALERTS:
        return None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
        v3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v3)
        det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    except Exception as exc:
        print('  (detector unavailable, segments will carry no alerts: %s)' % exc)
        return None

    fps_target = float(os.environ.get('TARGET_FPS', 20))
    state = v3.V3MultiPersonFallState()
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, last_slot, last, times = 0, -1, None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        slot = int(i * fps_target / src)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        hit = any(r[1] for r in v3.detect_v3_fall_multi(frame, state, det, config=None))
        label = 'fall' if hit else 'no_fall'
        if label != last:
            if hit:
                times.append(round((i - 1) / src, 2))
            last = label
    cap.release()
    return times


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if WRITE_SHEETS:
        os.makedirs(SHEET_DIR, exist_ok=True)

    out = {'note': 'Proposed segments from hard cuts. NOT ground truth: every boundary and '
                   'every label has to be checked against the contact sheet first.',
           'min_seconds': MIN_SECONDS, 'clips': {}}
    total = 0
    for clip in tc.COMPILATION:
        if not os.path.exists(clip):
            continue
        name = os.path.basename(clip)
        # Two passes over the file rather than guessing a frame rate: the first finds the
        # cuts and reports the real fps, the second drops shots shorter than MIN_SECONDS at
        # that rate. Guessing 30 would keep half-second shots in a 60 fps clip.
        _segs, fps = shots(clip)
        segs = [(a, b, secs) for a, b, secs in _segs if secs >= MIN_SECONDS]
        alerts = alert_times(clip)
        rows = []
        for n, (start, end, secs) in enumerate(segs, 1):
            t0, t1 = start / fps, end / fps
            inside = [t for t in (alerts or []) if t0 <= t < t1]
            sheet = ''
            if WRITE_SHEETS:
                sheet = os.path.join('sheets', '%s_%02d.jpg' % (name.replace('.mp4', ''), n))
                contact_sheet(clip, start, end,
                              os.path.join(OUT_DIR, sheet),
                              '%s segment %d   %.1fs-%.1fs   %s'
                              % (name, n, t0, t1,
                                 ('alerts at ' + ', '.join('%.1fs' % t for t in inside))
                                 if inside else 'no alert'))
            rows.append({'segment': n, 'start_s': round(t0, 2), 'end_s': round(t1, 2),
                         'seconds': round(secs, 2), 'alerts_at': inside,
                         'sheet': sheet, 'label': None})
        out['clips'][name] = rows
        total += len(rows)
        print('  %-10s %3d segments, %2d with an alert'
              % (name, len(rows), sum(1 for r in rows if r['alerts_at'])), flush=True)

    with open(os.path.join(OUT_DIR, 'incidents.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print()
    print('%d proposed segments from %d compilations.' % (total, len(out['clips'])))
    print('Every "label" is null. They stay null until somebody looks at the sheet -- a shot is')
    print('not an incident, and a boundary from a hard cut is a proposal, not a fact.')
    print('Written to %s' % OUT_DIR)


if __name__ == '__main__':
    main()
