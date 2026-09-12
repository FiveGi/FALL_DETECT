"""Why does multi-person mode alert? Attributes every alert to a mechanism.

SS25 identified two suspected causes for multi-person mode's extra false positives -- extra
pose slots producing duplicate detections of one person, and track churn forcing repeated
cold-start windows -- but "not yet fixed", and explicitly flagged that a fix needs
before/after measurement rather than a guess. This measures which mechanism is actually
responsible, per alert, so a fix can target the real one.

For each alert it records:
  track_age      frames since this track id was created. A window needs WINDOW_SIZE frames;
                 an alert at an age barely above that fired off a window that just filled,
                 which is SS25's cold-start hypothesis.
  real_frames    how many of the window's frames were genuinely observed rather than held
                 copies of the last good pose (person_flags). A window that is mostly held
                 frames is scoring a synthetic pattern.
  people         people detected in the frame at alert time.
  dup_dist       distance from this track's hip centre to the nearest OTHER track that
                 alerted within 0.5s. Small means two tracks fired for one person.
  kpt_conf       mean keypoint confidence over the window's observed frames, and
  body_span      the pose's bounding height as a fraction of frame height. A marginal
                 detection -- someone far away, half out of frame, or a mis-detection --
                 scores low on both, while a genuine fall close to the camera does not.
                 These separate "hard to see because the person is prone" (which
                 MIN_PERSON_FRACTION cannot tell apart from junk) from "hard to see
                 because there is barely anything there".

Usage:
    python training/diagnose_multi_person.py            # all Test/ clips
    python training/diagnose_multi_person.py 4 15       # specific clips
"""
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'v3_fall_detection', os.path.join(ROOT, 'app', 'detection', 'v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

TEST_DIR = os.environ.get('TEST_DIR', os.path.join(ROOT, 'Test'))
MODEL_DIR = os.environ.get('TEST_MODEL_DIR', os.path.join(ROOT, 'models'))


def run_clip(detector, path):
    state = v3.V3MultiPersonFallState()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    track_born = {}
    last_label = None
    alerts = []
    max_people = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = v3.detect_v3_fall_multi(frame, state, detector, config=None)
        max_people = max(max_people, len(results))

        for track_id, *_ in results:
            track_born.setdefault(track_id, frame_idx)

        firing = [r for r in results if r[1]]
        label = 'fall' if firing else 'no_fall'
        if label != last_label:
            if label == 'fall':
                for track_id, _, prob, _, centroid in firing:
                    person_state = state.person_states.get(track_id)
                    flags = list(person_state.person_flags) if person_state else []
                    buf = list(person_state.raw_buffer) if person_state else []
                    confs, spans = [], []
                    for kp in buf:
                        visible = kp[kp[:, 2] > 0.3]
                        if len(visible) >= 4:
                            confs.append(float(np.mean(kp[:, 2])))
                            spans.append(float(visible[:, 1].max() - visible[:, 1].min()))
                    alerts.append({
                        't': round(frame_idx / fps, 1),
                        'track': track_id,
                        'prob': round(float(prob), 2),
                        'track_age': frame_idx - track_born.get(track_id, frame_idx),
                        'real_frames': int(sum(1 for f in flags if f)),
                        'window': len(flags),
                        'people': len(results),
                        'centroid': [round(float(c), 3) for c in centroid],
                        'kpt_conf': round(float(np.mean(confs)), 3) if confs else 0.0,
                        'body_span': round(float(np.mean(spans)), 3) if spans else 0.0,
                    })
            last_label = label
        frame_idx += 1
    cap.release()

    # Duplicate check: two tracks alerting near-simultaneously at nearly the same place are
    # one person counted twice, not two people falling.
    for a in alerts:
        best = None
        for b in alerts:
            if b is a or abs(b['t'] - a['t']) > 0.5 or b['track'] == a['track']:
                continue
            d = float(np.linalg.norm(np.array(a['centroid']) - np.array(b['centroid'])))
            best = d if best is None else min(best, d)
        a['dup_dist'] = None if best is None else round(best, 3)
    return alerts, max_people, frame_idx


VAL_FALL = ["s1_Fall_02", "s1_Fall_06", "s1_Fall_16", "s2_Fall_02", "s2_Fall_04",
            "s2_Fall_09", "s2_Fall_14", "s2_Fall_20", "s3_Fall_02", "s3_Fall_09",
            "s3_Fall_12", "s3_Fall_13", "s3_Fall_16", "s4_Fall_06", "s4_Fall_17"]
VAL_ADL = ["s1_ADL_01", "s1_ADL_05", "s1_ADL_11", "s1_ADL_13", "s2_ADL_03", "s2_ADL_07",
           "s2_ADL_13", "s2_ADL_15", "s2_ADL_16", "s2_ADL_18", "s2_ADL_20", "s3_ADL_07",
           "s3_ADL_11", "s4_ADL_07", "s4_ADL_08", "s4_ADL_10"]
GM_FALL_DIR = os.path.join(ROOT, 'training', 'data', 'gmdcsa24_fall_raw')
GM_ADL_DIR = os.path.join(ROOT, 'training', 'data', 'gmdcsa24_adl_raw_val')


def gmdcsa_mode():
    """Every alert on an ADL clip is a known-wrong alert, and every alert on a Fall clip is
    the one we must not lose -- so this compares the two populations directly instead of
    guessing which Test/ alerts were real."""
    detector = v3.V3PoseFallDetector(model_dir=MODEL_DIR)
    out = {'fall': [], 'adl': []}
    for kind, names, d in (('fall', VAL_FALL, GM_FALL_DIR), ('adl', VAL_ADL, GM_ADL_DIR)):
        for name in names:
            path = os.path.join(d, name + '.mp4')
            if not os.path.exists(path):
                continue
            alerts, _, _ = run_clip(detector, path)
            for a in alerts:
                a['clip'] = name
            out[kind].extend(alerts)
        print(f'{kind}: {len(out[kind])} alerts', flush=True)
    dest = os.path.join(ROOT, 'training', 'data', 'multi_diagnosis_gmdcsa.json')
    with open(dest, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('wrote', dest)


def main():
    if sys.argv[1:2] == ['--gmdcsa']:
        gmdcsa_mode()
        return
    clips = sys.argv[1:] or [str(i) for i in range(1, 18)]
    detector = v3.V3PoseFallDetector(model_dir=MODEL_DIR)

    all_alerts = {}
    print(f"{'clip':>5} {'people':>7} {'alerts':>7}  breakdown")
    totals = {'cold': 0, 'dup': 0, 'held': 0, 'clean': 0}
    for c in clips:
        path = os.path.join(TEST_DIR, f'{c}.mp4')
        if not os.path.exists(path):
            continue
        alerts, max_people, frames = run_clip(detector, path)
        all_alerts[c] = alerts

        cold = sum(1 for a in alerts if a['track_age'] < v3.WINDOW_SIZE * 2)
        dup = sum(1 for a in alerts if a['dup_dist'] is not None and a['dup_dist'] < 0.15)
        held = sum(1 for a in alerts if a['window'] and a['real_frames'] < a['window'] * 0.7)
        clean = sum(1 for a in alerts
                    if a['track_age'] >= v3.WINDOW_SIZE * 2
                    and not (a['dup_dist'] is not None and a['dup_dist'] < 0.15)
                    and not (a['window'] and a['real_frames'] < a['window'] * 0.7))
        for k, v in (('cold', cold), ('dup', dup), ('held', held), ('clean', clean)):
            totals[k] += v
        print(f"{c:>5} {max_people:>7} {len(alerts):>7}  cold-start={cold} duplicate={dup} "
              f"mostly-held={held} clean={clean}")

    print(f"\nTOTAL alerts={sum(len(v) for v in all_alerts.values())}  " +
          '  '.join(f'{k}={v}' for k, v in totals.items()))
    out = os.path.join(ROOT, 'training', 'data', 'multi_diagnosis.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(all_alerts, f, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
