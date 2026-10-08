# -*- coding: utf-8 -*-
"""Draw what the detector actually sees, so it can be looked at instead of read about.

Every number in this project describes something nobody has watched: a skeleton, a score, and
the moment a rule decides to alert. This renders that. It is not a debugging aid bolted on
afterwards -- looking at frames has changed three conclusions here (Test/17 was carried as a
missed fall for days, a "caught fall" turned out to be aftermath, and a diagnostic once
declared a man lying on the floor to be standing because it read the wrong person out of a
confidence-sorted list).

Draws, per frame: the skeleton the pose model returned, the classifier's score for that
person, whether the alerting rule is currently saying fall, and the torso angle the
still-down signal reads.

Usage:
    V3_DEVICE=cuda python training/measure/render_detection.py Test/15.mp4 out.png
    FRAMES=8 AROUND_ALERT=1 python training/measure/render_detection.py Test/15.mp4 out.png
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

FRAMES = int(os.environ.get('FRAMES', 6))
COLS = int(os.environ.get('COLS', 3))
TILE_W = int(os.environ.get('TILE_W', 420))
FPS = float(os.environ.get('TARGET_FPS', 20))
# Centre the strip on the alert rather than spreading it over the whole clip: the interesting
# seconds are the ones around the decision, and a clip is mostly not those.
AROUND_ALERT = os.environ.get('AROUND_ALERT', '1') == '1'

# COCO-17 skeleton, drawn so a human can see a body rather than a scatter of dots.
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def draw(frame, people, scores, alerting):
    out = frame.copy()
    h, w = out.shape[:2]
    for idx, (kpts, _hip) in enumerate(people):
        score = scores.get(idx)
        hit = bool(alerting.get(idx))
        colour = (0, 0, 255) if hit else (0, 200, 255)
        pts = [(int(kpts[i, 0] * w), int(kpts[i, 1] * h)) for i in range(17)]
        for a, b in EDGES:
            if kpts[a, 2] > 0.2 and kpts[b, 2] > 0.2:
                cv2.line(out, pts[a], pts[b], colour, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            if kpts[i, 2] > 0.2:
                cv2.circle(out, p, 3, colour, -1, cv2.LINE_AA)
        cos = v3.torso_cos(kpts)
        label = 'score %.2f' % score if score is not None else 'no score yet'
        if cos is not None:
            label += '   torso %s' % ('upright' if cos >= v3.UPRIGHT_COS else 'DOWN')
        anchor = (max(4, pts[11][0] - 60), max(18, min(pts[11][1], h - 8)))
        cv2.putText(out, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    source, dest = sys.argv[1], sys.argv[2]

    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    # Draw the people the DETECTOR saw, not a second pose pass. extract_all_keypoints is
    # stateful when V3_ROI_IMGSZ is set (crop cadence + last boxes), and an earlier version
    # called it again here: every frame then ran the pose model twice, the crop cadence ran at
    # double speed, and the render showed a different detector from the one that was scored --
    # it drew "alert at never" on 8.mp4#1, which the scorer had caught at 4.2 s.
    seen = {'people': []}
    _extract = det.extract_all_keypoints

    def _spy(frame):
        seen['people'] = _extract(frame)
        return seen['people']
    det.extract_all_keypoints = _spy
    # Same start as the scorer's segment, so a phase-N evaluation can be rendered as it ran.
    det.reset_roi_state(int(os.environ.get('ROI_PHASE', 0)))
    state = v3.V3MultiPersonFallState()
    cap = cv2.VideoCapture(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # START_S / END_S: render one segment of a compilation instead of the whole file.
    start_s, end_s = float(os.environ.get('START_S', 0)), float(os.environ.get('END_S', 0))
    if start_s:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * src_fps))

    shots, i, last_slot, alert_at = [], int(start_s * src_fps), -1, None
    while True:
        ok, frame = cap.read()
        if not ok or (end_s and i >= int(end_s * src_fps)):
            break
        slot = int(i * FPS / src_fps)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        seen['people'] = []
        results = v3.detect_v3_fall_multi(frame, state, det, config=None)
        people = seen['people']
        # Match each drawn person to their result by hip position, the same way the tracker
        # does, so a score is never drawn on the wrong body -- which has happened.
        scores, alerting = {}, {}
        for idx, (_kp, hip) in enumerate(people):
            best = None
            for track_id, detected, prob, _label, centroid in results:
                d = float(np.linalg.norm(np.asarray(centroid) - hip))
                if best is None or d < best[0]:
                    best = (d, prob, detected)
            if best and best[0] < 0.1:
                scores[idx], alerting[idx] = best[1], best[2]
        # The alert is the detector's, whether or not its track matched a drawn body (a track
        # held through a missed frame has no body this frame) -- the scorer counts it too.
        hit = any(r[1] for r in results)
        if hit and alert_at is None:
            alert_at = len(shots)
        shots.append((len(shots), draw(frame, people, scores, alerting), hit))
    cap.release()
    if not shots:
        print('no frames read from %s' % source)
        return 1

    if AROUND_ALERT and alert_at is not None:
        lo = max(0, alert_at - FRAMES // 3)
        picks = list(range(lo, min(len(shots), lo + FRAMES)))
    else:
        picks = list(np.linspace(0, len(shots) - 1, FRAMES).astype(int))

    tiles = []
    for p in picks:
        idx, img, hit = shots[p]
        h, w = img.shape[:2]
        img = cv2.resize(img, (TILE_W, int(h * TILE_W / w)))
        banner = np.zeros((26, img.shape[1], 3), np.uint8)
        banner[:] = (0, 0, 160) if hit else (40, 40, 40)
        text = '%.1fs   %s' % (idx / FPS, 'ALERT' if hit else 'watching')
        cv2.putText(banner, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(np.vstack([banner, img]))

    height = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT)
             for t in tiles]
    rows = [np.hstack(tiles[r:r + COLS]) for r in range(0, len(tiles), COLS)]
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(dest, np.vstack(rows))
    print('wrote %s  (%d frames, alert at %s)'
          % (dest, len(picks), '%.1fs' % (alert_at / FPS) if alert_at is not None else 'never'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
