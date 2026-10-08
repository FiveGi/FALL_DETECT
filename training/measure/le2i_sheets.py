# -*- coding: utf-8 -*-
"""Frame sheets for the Le2i adjudication clips (pre-registration 2026-10-05: judges review the video span, blind to
config). Neither judge can play an mp4 (Gemini reads images only), so the span is shown COMPLETELY as frames:

  * every frame at 4 fps from the start to the end of the clip `le2i_eval.py clips` wrote (no gaps longer than 0.25 s
    -- Codex P2: 12 evenly spaced stills could skip an alert);
  * each tile is stamped with its time inside the clip; a tile within 0.125 s of ANY pooled alert time (union over all
    configs and phases) gets a red border, so the judge looks at what the person is doing at every moment an alert
    fired, without learning which config fired it;
  * pages of 20 tiles (5 x 4): <video>_p1.png, _p2.png ...
  * a frame that cannot be decoded aborts the whole run (Codex P2: no silent black tiles);
  * the clip set must equal the expected set (every non-fall video that alerted in any config/phase -- a missing clip
    would silently drop a false alarm), every pooled alert time must fall inside its clip (a clip le2i_eval.clips
    truncated on a decode failure is rejected), and a sheet that fails to save aborts (Codex P2 x2, 7 Oct).

Usage: python training/measure/le2i_sheets.py A_finalist REF_production
"""
import glob
import json
import os
import sys

import importlib.util

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'training', 'data', 'le2i_final')
DIR = os.path.join(OUT, 'adjudicate')
STEP, COLS, ROWS, W = 0.25, 5, 4, 360


def pooled_alerts(names):
    """video -> sorted union of alert times (s, source video time) over every config and phase."""
    times = {}
    for name in names:
        files = sorted(glob.glob(os.path.join(OUT, '%s_p*.json' % name)))
        if len(files) != 8:
            raise SystemExit('%s: %d phase files, need 8' % (name, len(files)))
        for f in files:
            for v, r in json.load(open(f))['videos'].items():
                times.setdefault(v.replace('/', '_'), set()).update(r['alerts_at'])
    return {v: sorted(t) for v, t in times.items()}


def expected_videos(names):
    """Non-fall videos that alerted in any config/phase -- exactly the set le2i_eval.clips writes."""
    spec = importlib.util.spec_from_file_location('le2i_eval', os.path.join(ROOT, 'training', 'measure', 'le2i_eval.py'))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    falls = ev.fall_labels()
    want = set()
    for name in names:
        for f in glob.glob(os.path.join(OUT, '%s_p*.json' % name)):
            for v, r in json.load(open(f))['videos'].items():
                if not falls[v] and r['alerts_at']:
                    want.add(v.replace('/', '_'))
    return want


def main(names):
    alerts = pooled_alerts(names)
    mp4s = sorted(glob.glob(os.path.join(DIR, '*.mp4')))
    want, have = expected_videos(names), {os.path.basename(m)[:-4] for m in mp4s}
    if want != have:
        raise SystemExit('clip set differs from the expected set: missing %s, unexpected %s'
                         % (sorted(want - have), sorted(have - want)))
    if not mp4s:
        print('no non-fall video alerted in any config/phase: nothing to adjudicate')
        return
    for mp4 in mp4s:
        vid = os.path.basename(mp4)[:-4]
        at = alerts.get(vid)
        if not at:
            raise SystemExit('%s: no pooled alert times -- clip and run outputs disagree' % vid)
        t0 = max(0.0, at[0] - 3)                       # the clip starts here in source time (le2i_eval.clips)
        cap = cv2.VideoCapture(mp4)
        total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        if total <= 0 or not fps:
            raise SystemExit('%s: cannot read frame count / fps' % mp4)
        dur = total / fps
        outside = [a for a in at if not (t0 - 0.01 <= a <= t0 + dur + 0.01)]
        if outside:
            raise SystemExit('%s: alert times %s fall outside the clip (%.2f-%.2f s) -- clip truncated?'
                             % (vid, outside[:3], t0, t0 + dur))
        tiles = []
        for t in np.arange(0, dur, STEP):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(round(t * fps)), total - 1))
            ok, fr = cap.read()
            if not ok:
                raise SystemExit('%s: frame at %.2f s did not decode -- sheet aborted' % (mp4, t))
            fr = cv2.resize(fr, (W, int(fr.shape[0] * W / fr.shape[1])))
            src_t = t0 + t
            if any(abs(src_t - a) <= STEP / 2 for a in at):
                fr = cv2.copyMakeBorder(fr[6:-6, 6:-6], 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(0, 0, 255))
            label = '%.2fs' % t
            cv2.putText(fr, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(fr, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(fr)
        cap.release()
        h = max(x.shape[0] for x in tiles)
        blank = np.full((h, W, 3), 40, np.uint8)
        per = COLS * ROWS
        for p in range(0, len(tiles), per):
            page = tiles[p:p + per]
            page = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in page]
            page += [blank] * (per - len(page))
            grid = np.vstack([np.hstack(page[r:r + COLS]) for r in range(0, per, COLS)])
            png = os.path.join(DIR, '%s_p%d.png' % (vid, p // per + 1))
            if not cv2.imwrite(png, grid):
                raise SystemExit('could not write %s' % png)
        print('%s: %.1f s, %d tiles, %d pooled alert times, %d page(s)'
              % (vid, total / fps, len(tiles), len(at), (len(tiles) + per - 1) // per))


if __name__ == '__main__':
    main(sys.argv[1:] or ['A_finalist', 'REF_production'])
