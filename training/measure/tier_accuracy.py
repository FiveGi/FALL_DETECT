# -*- coding: utf-8 -*-
"""Is the alert TIER right, and where exactly is it wrong?

The tier has never been measured as a classifier, only defended as a design choice. It decides
what a family is told: 'check' asks someone to look, 'confirmed' says go now. Getting it wrong
in one direction wastes a trip; getting it wrong in the other buries a real fall among
please-checks.

Two facts can raise a fresh alert to 'confirmed', and only one of them can be measured from a
clip: **the person is still on the floor** (notification_service.alert_tier). The other,
"nobody acknowledged", is about the people receiving the alert and no recording can answer it.
So this measures the tier a clip would reach on its own evidence, within STILL_DOWN_SECONDS of
the alert.

The labels need no new ground truth: an alert in a fall clip is the fall, an alert in an ADL
clip is a false alarm. That is the whole labelling, which is what makes this cheap enough to
run after every change.

**Every disagreement is written out as an image**, because a confusion matrix says how often
and never says why, and this project has twice been wrong about a clip it had not looked at.

Usage:
    V3_DEVICE=cuda V3_IMGSZ=320 TARGET_FPS=8 python training/measure/tier_accuracy.py
"""
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'training'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
import cache_pose_streams as cache          # noqa: E402
from replay_classifiers import load_stream  # noqa: E402

MODEL_DIR = os.environ.get('TEST_MODEL_DIR', os.path.join(ROOT, 'models'))
OUT_DIR = os.environ.get('RESULTS_DIR', os.path.dirname(os.path.abspath(__file__)))
SHOT_DIR = os.path.join(OUT_DIR, 'tier_disagreements')
# Matches notification_service.STILL_DOWN_SECONDS; imported rather than repeated would pull in
# Flask, and that module is deliberately importable without it.
STILL_DOWN_SECONDS = float(os.environ.get('STILL_DOWN_SECONDS', 10))


def _load_still_down_rule():
    """notification_service.still_down_confirmed, executed directly so Flask is not needed --
    the camera loop's own rule rather than a restatement of it, which is how this file once
    drifted from the loop (it kept a 0.8 fraction after the loop's rule changed)."""
    path = os.path.join(ROOT, 'app', 'services', 'notification_service.py')
    src = open(path, encoding='utf-8').read().replace(
        'from app.services.line_service import send_line_message_async',
        'send_line_message_async = None')
    ns = {}
    exec(compile(src, path, 'exec'), ns)
    return ns['still_down_confirmed'], ns['seen_down_since_alert']


v3_still_down, v3_seen_since = _load_still_down_rule()


def tier_for_clip(det, frames, fps):
    """-> (alerted, tier, alert_frame, seconds_down, observed_seconds)."""
    state = v3.V3MultiPersonFallState()
    alert_at, alert_track = None, None
    window = int(STILL_DOWN_SECONDS * fps)

    for i, people in enumerate(frames):
        det._replay = [(kp, (kp[v3.LEFT_HIP, :2] + kp[v3.RIGHT_HIP, :2]) / 2.0) for kp in people]
        results = v3.detect_v3_fall_multi(None, state, det, config=None)
        if alert_at is None:
            top = v3.alert_result(results) if results else None
            if top and top[1]:
                alert_at, alert_track = i, top[0]
                p = state.person_states.get(alert_track)
                seen_at_alert = p.frames_seen_down if p else 0
            continue
        if i - alert_at >= window:
            break

    if alert_at is None:
        return False, None, None, 0.0, 0.0

    observed = min(len(frames) - 1 - alert_at, window) / fps
    person = state.person_states.get(alert_track)
    down = (person.frames_since_upright / fps) if person else 0.0
    # Not enough footage after the alert to answer: the tier stays where a fresh alert starts.
    if observed < STILL_DOWN_SECONDS:
        return True, 'check', alert_at, down, observed
    # The live rule, not a copy of it: same function, same frame counts.
    elapsed_frames = int(round(observed * fps))
    tier = ('confirmed' if person is not None and v3_still_down(
        person.frames_since_upright,
        v3_seen_since(person.frames_seen_down, seen_at_alert), elapsed_frames) else 'check')
    return True, tier, alert_at, down, observed


def render(path, frame_index, fps, label, out_png):
    """The frame the alert fired on, so a disagreement can be looked at rather than counted."""
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index * src / fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    if 'urfd' in path.replace('\\', '/'):
        frame = frame[:, frame.shape[1] // 2:]
    frame = cv2.resize(frame, (480, 360))
    cv2.putText(frame, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_png, frame)
    return True


def main():
    key = cache.cache_key()
    cache_dir = cache.cache_dir_for(key)
    if not os.path.exists(os.path.join(cache_dir, 'key.json')):
        raise SystemExit('no cached pose stream for this configuration: %s' % json.dumps(key))
    os.makedirs(SHOT_DIR, exist_ok=True)

    det = v3.V3PoseFallDetector(model_dir=MODEL_DIR)
    det._replay = []
    det.extract_all_keypoints = lambda frame: det._replay

    # (group, is a real fall)
    TRUTH = {'urfd_fall': True, 'gmdcsa_fall': True,
             'urfd_adl': False, 'val_adl': False, 'train50_adl': False}
    rows = []
    for group, paths, _half in cache.clip_groups():
        for path in paths:
            name = os.path.basename(path)
            npz = os.path.join(cache_dir, '%s__%s.npz' % (group, name))
            if not os.path.exists(npz):
                continue
            alerted, tier, at, down, observed = tier_for_clip(
                det, load_stream(npz), key['fps'])
            if not alerted:
                continue
            rows.append({'group': group, 'clip': name, 'path': path, 'is_fall': TRUTH[group],
                         'tier': tier, 'alert_frame': at,
                         'seconds_down': round(down, 1), 'observed_seconds': round(observed, 1)})
        print('  %-14s done' % group, flush=True)

    report(rows, key['fps'])
    with open(os.path.join(OUT_DIR, 'tier_accuracy.json'), 'w', encoding='utf-8') as fh:
        json.dump({'meta': {'still_down_seconds': STILL_DOWN_SECONDS, 'cache': key},
                   'rows': rows}, fh, indent=1)


def report(rows, fps):
    print()
    print('Every alert, by what was really in the clip and the tier it would reach on its own')
    print('evidence within %.0fs. "nobody acknowledged" cannot be measured from a recording,'
          % STILL_DOWN_SECONDS)
    print('so this is the floor of the tier, never the ceiling.')
    print()
    grid = {}
    for r in rows:
        grid.setdefault((r['is_fall'], r['tier']), []).append(r)
    n_fall = sum(1 for r in rows if r['is_fall'])
    n_false = len(rows) - n_fall
    print('%-34s %10s %10s' % ('', "'confirmed'", "'check'"))
    print('-' * 56)
    for is_fall, label, total in ((True, 'alerts on a REAL FALL', n_fall),
                                  (False, 'alerts that are FALSE ALARMS', n_false)):
        c = len(grid.get((is_fall, 'confirmed'), []))
        k = len(grid.get((is_fall, 'check'), []))
        print('%-34s %10s %10s' % ('%s (%d)' % (label, total),
                                   '%d (%.0f%%)' % (c, 100.0 * c / max(total, 1)),
                                   '%d (%.0f%%)' % (k, 100.0 * k / max(total, 1))))
    confirmed = len(grid.get((True, 'confirmed'), [])) + len(grid.get((False, 'confirmed'), []))
    right = len(grid.get((True, 'confirmed'), []))
    print()
    print("When it says 'go now', it is a real fall %d of %d times (%.0f%%)."
          % (right, confirmed, 100.0 * right / max(confirmed, 1)))
    print("A real fall that only reaches 'check' is still an alert -- it asks someone to look.")
    print("A false alarm that reaches 'confirmed' is the costly one: it is the alert that")
    print("teaches people to ignore the channel.")

    bad = grid.get((False, 'confirmed'), [])
    print()
    print('THE COSTLY ONES -- false alarms the system would call urgent: %d' % len(bad))
    for r in sorted(bad, key=lambda r: r['clip']):
        png = os.path.join(SHOT_DIR, '%s__%s.png' % (r['group'], r['clip'].replace('.mp4', '')))
        ok = render(r['path'], r['alert_frame'], fps,
                    '%s  URGENT but no fall  down %.0fs/%.0fs'
                    % (r['clip'], r['seconds_down'], r['observed_seconds']), png)
        print('   %-14s %-24s down %4.1fs of %4.1fs   %s'
              % (r['group'], r['clip'], r['seconds_down'], r['observed_seconds'],
                 png if ok else '(frame could not be read)'))
    missed = grid.get((True, 'check'), [])
    print()
    print('Real falls that stay at "please check": %d (they still alert)' % len(missed))
    print('Images for every costly disagreement are in %s' % SHOT_DIR)


if __name__ == '__main__':
    main()
