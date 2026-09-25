# -*- coding: utf-8 -*-
"""After an alert fires, does the person get back up? And does that separate the false alarms?

Every signal this project has tried against false alarms asks the classifier's own question --
is THIS moment a fall -- and they all overlap, because a person mid-bend and a person mid-fall
look alike inside a one-second window. This asks a different question over a longer horizon:
after the alert, does the person stay down? That is how a human tells the two apart, and it is
the distinction that matters for care. Somebody back on their feet in three seconds does not
need anyone to come.

The measurement runs on the cached pose stream, so it is the same keypoints the detector saw,
and it costs seconds rather than an hour of decoding.

Upright is measured as the torso's angle from vertical: |shoulder_y - hip_y| / |shoulder - hip|
is the cosine of that angle, so it needs no reference height, no calibration and no assumption
about where the camera is. UPRIGHT_COS 0.70 is about 45 degrees.

Read the output as two distributions, not one number. The question is not "do people get up"
but **"does getting up happen after false alarms and not after real falls"** -- and the row
that decides whether this can ever ship is the real falls where the person DOES get up, because
cancelling one of those is cancelling a real call for help.

Usage:
    V3_DEVICE=cuda V3_IMGSZ=320 TARGET_FPS=8 python training/measure/recovery_after_alert.py
"""
import importlib.util
import json
import os
import sys

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
OUT = os.environ.get('OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'recovery_after_alert.json'))
# The detector owns this definition now, so the measurement and the thing it
# measures cannot drift apart.
UPRIGHT_COS = v3.UPRIGHT_COS
# Seconds after the alert to watch. Long enough that someone who was only bending has
# straightened up, short enough that it is still the same event.
WINDOW_S = float(os.environ.get('RECOVERY_WINDOW_S', 10.0))
# How much of that window the person has to be upright for. A single upright frame is jitter.
NEED_FRACTION = float(os.environ.get('RECOVERY_NEED', 0.5))
# A clip that ends three frames after the alert cannot answer "did they get up". The first run
# of this counted those as recoveries and reported that it would cancel 17 real falls, several
# of them judged on two to four frames. Below this many observed frames the answer is UNKNOWN,
# which is a different thing from "no" and has to be reported separately or the rates are
# meaningless.
MIN_FRAMES_AFTER = int(os.environ.get('RECOVERY_MIN_FRAMES', 16))


def run_clip(det, frames, fps):
    """-> dict describing the first alert in this clip and what the person did afterwards."""
    state = v3.V3MultiPersonFallState()
    alert_at, alert_track = None, None
    upright_after, seen_after = 0, 0
    window_frames = int(WINDOW_S * fps)

    for i, people in enumerate(frames):
        det._replay = [(kp, (kp[v3.LEFT_HIP, :2] + kp[v3.RIGHT_HIP, :2]) / 2.0) for kp in people]
        # The centroid each result carries is the same array object the detection supplied, so
        # a track can be matched back to its keypoints without re-running the tracker here.
        by_hip = {tuple(np.round(h, 6)): kp for kp, h in det._replay}
        results = v3.detect_v3_fall_multi(None, state, det, config=None)

        if alert_at is None:
            for track_id, detected, _prob, _label, centroid in results:
                if detected:
                    alert_at, alert_track = i, track_id
                    break
            continue

        if i - alert_at > window_frames:
            break
        for track_id, _detected, _prob, _label, centroid in results:
            if track_id != alert_track:
                continue
            kp = by_hip.get(tuple(np.round(np.asarray(centroid), 6)))
            if kp is None:
                continue          # the track is being held through a dropout: no new evidence
            cos = v3.torso_cos(kp)
            if cos is None:
                continue
            seen_after += 1
            if cos >= UPRIGHT_COS:
                upright_after += 1

    if alert_at is None:
        return None
    fraction = upright_after / seen_after if seen_after else 0.0
    enough = seen_after >= MIN_FRAMES_AFTER
    return {
        'alert_frame': alert_at,
        'frames_after': seen_after,
        'upright_after': upright_after,
        'upright_fraction': round(fraction, 3),
        'enough_evidence': enough,
        'would_cancel': bool(enough and fraction >= NEED_FRACTION),
    }


def main():
    key = cache.cache_key()
    cache_dir = cache.cache_dir_for(key)
    if not os.path.exists(os.path.join(cache_dir, 'key.json')):
        raise SystemExit('no cached pose stream for this configuration: %s' % json.dumps(key))

    det = v3.V3PoseFallDetector(model_dir=MODEL_DIR)
    det._replay = []
    det.extract_all_keypoints = lambda frame: det._replay

    results = {}
    for group, paths, _half in cache.clip_groups():
        for path in paths:
            name = os.path.basename(path)
            npz = os.path.join(cache_dir, '%s__%s.npz' % (group, name))
            if not os.path.exists(npz):
                continue
            row = run_clip(det, load_stream(npz), key['fps'])
            if row is not None:
                results['%s/%s' % (group, name)] = row
        print('  %-14s done' % group, flush=True)

    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump({'meta': {'upright_cos': UPRIGHT_COS, 'window_s': WINDOW_S,
                            'need_fraction': NEED_FRACTION, 'cache': key},
                   'clips': results}, fh, indent=1)
    report(results)


def report(results):
    # A fall clip's alert is (almost always) the real fall; an ADL clip's alert is a false
    # alarm by definition. That is the only labelling this needs, and it needs no new ground
    # truth, which is why this measurement is cheap enough to be worth running first.
    # URFD's two cameras are split out because they are different problems: cam0 is ceiling
    # mounted and cam1 is on a wall. A torso angle read from directly above says almost nothing
    # about whether someone is standing, and the README already requires a wall mount -- so if
    # the signal works at all, it has to work on cam1 and may simply be inapplicable to cam0.
    groups = {
        'real falls URFD cam1 (wall)': [k for k in results if k.startswith('urfd_fall/') and 'cam1' in k],
        'real falls URFD cam0 (ceiling)': [k for k in results if k.startswith('urfd_fall/') and 'cam0' in k],
        'real falls GMDCSA24': [k for k in results if k.startswith('gmdcsa_fall/')],
        'FALSE ALARM URFD cam1': [k for k in results if k.startswith('urfd_adl/') and 'cam1' in k],
        'FALSE ALARM URFD cam0': [k for k in results if k.startswith('urfd_adl/') and 'cam0' in k],
        'FALSE ALARM val ADL': [k for k in results if k.startswith('val_adl/')],
        'FALSE ALARM train50': [k for k in results if k.startswith('train50_adl/')],
    }
    print()
    print('%-32s %7s %8s %14s %22s'
          % ('group', 'alerts', 'usable', 'would cancel', 'upright fraction'))
    print('-' * 90)
    for label, keys in groups.items():
        if not keys:
            continue
        usable = [k for k in keys if results[k]['enough_evidence']]
        cancel = sum(1 for k in usable if results[k]['would_cancel'])
        if not usable:
            print('%-32s %7d %8d %14s' % (label, len(keys), 0, 'no evidence'))
            continue
        fr = sorted(results[k]['upright_fraction'] for k in usable)
        print('%-32s %7d %8d %8d (%3.0f%%) %8s med %.2f  q1 %.2f  q3 %.2f'
              % (label, len(keys), len(usable), cancel, 100.0 * cancel / len(usable), '',
                 fr[len(fr) // 2], fr[len(fr) // 4], fr[(3 * len(fr)) // 4]))
    print()
    real = [k for k in results if 'fall/' in k and results[k]['would_cancel']]
    print('(usable = the clip ran at least %d scored frames past the alert; anything shorter '
          'cannot answer the question)' % MIN_FRAMES_AFTER)
    print('REAL FALLS THIS WOULD CANCEL: %d -- each one is a real call for help suppressed'
          % len(real))
    for k in sorted(real)[:15]:
        print('   %-30s upright %.2f of %d frames after the alert'
              % (k, results[k]['upright_fraction'], results[k]['frames_after']))


if __name__ == '__main__':
    main()
