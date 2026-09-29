# -*- coding: utf-8 -*-
"""Somebody was there, and now they are not. Is that a fall the pose model stopped seeing?

The idea this measures: a tracked person who disappears for a couple of seconds might have
fallen into a position the pose model cannot read -- prone behind a sofa, curled on the floor,
below the frame -- and the system would say nothing at all, because everything downstream only
sees frames where a person was found. A fall we cannot see is the worst failure this system
has, and it leaves no trace in any accuracy number.

A narrow version already ships: `COLLAPSE_ENABLED` fires when detection drops to nothing AND
the classifier was already above 0.6 beforehand. Measured on both profiles it earns nothing --
bit-identical on CPU -- because that precondition almost never holds. The broader version, fire
whenever a tracked person vanishes, would alert every time somebody walks out of the room.

**So the question is whether WHERE they vanish separates the two.** A person who leaves through
a doorway goes out at the edge of the frame; a person who falls out of the model's sight
vanishes in the middle of it. That costs nothing to compute -- the last known hip position is
already in the tracker -- which matters, because the deployment is a CPU-only server with no
room for another model.

Reported as two distributions, fall clips against ADL clips. If vanishing away from the edge
happens in both equally, the idea is dead and this says so.

Usage:
    V3_DEVICE=cuda V3_IMGSZ=320 TARGET_FPS=8 python training/measure/vanish_signal.py
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

OUT = os.environ.get('OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'vanish_signal.json'))
# How long somebody has to be gone before it counts as vanishing rather than a tracking blink.
#
# The ceiling is not free to choose: the tracker forgets a person after MAX_MISSED_FRAMES,
# which is one window -- 15 frames, 1.9s at the CPU profile's 8 fps. Asking for 2.0s therefore
# matched nothing at all in 220 clips, because the evidence is deleted a tenth of a second
# before the question is asked. That is what the first run of this measured: zero everywhere,
# which reads exactly like "the idea does not work".
#
# So the event is "the tracker gave up on somebody it had been following", and the threshold is
# clamped below that horizon. Anything longer needs MAX_MISSED_FRAMES raised first, which is a
# change to the detector and not to this script.
VANISH_SECONDS = float(os.environ.get('VANISH_SECONDS', 0))
# How long they had to be visible first. Someone glimpsed for three frames at the edge of the
# picture was never really tracked, and treating that as a disappearance is noise.
MIN_SEEN_SECONDS = float(os.environ.get('MIN_SEEN_SECONDS', 1.0))
# Fraction of the frame counted as "the edge". A doorway is at the edge; a floor is not.
EDGE = float(os.environ.get('EDGE_FRACTION', 0.15))


def near_edge(hip):
    x, y = float(hip[0]), float(hip[1])
    return x < EDGE or x > 1 - EDGE or y < EDGE or y > 1 - EDGE


def vanishes(frames, fps):
    """-> list of (seconds_visible_first, near_the_edge, seconds_gone) for this clip."""
    state = v3.V3MultiPersonFallState()
    seen_for, last_hip, gone_for = {}, {}, {}
    out = []
    horizon = v3.MAX_MISSED_FRAMES
    vanish_frames = int(VANISH_SECONDS * fps) if VANISH_SECONDS else horizon
    vanish_frames = min(vanish_frames, horizon)
    min_seen = int(MIN_SEEN_SECONDS * fps)

    for people in frames:
        det_replay = [(kp, (kp[v3.LEFT_HIP, :2] + kp[v3.RIGHT_HIP, :2]) / 2.0) for kp in people]
        tracked = state.tracker.update(det_replay)
        alive = set()
        for track_id, kpts, was_seen in tracked:
            alive.add(track_id)
            if was_seen:
                if gone_for.get(track_id, 0) >= vanish_frames and seen_for.get(track_id, 0) >= min_seen:
                    # They came back. Still a disappearance, and the fact that it ended is
                    # exactly what a "check on them" alert would have found out.
                    out.append((seen_for[track_id] / fps, near_edge(last_hip[track_id]),
                                gone_for[track_id] / fps, 'returned'))
                seen_for[track_id] = seen_for.get(track_id, 0) + 1
                gone_for[track_id] = 0
                last_hip[track_id] = (kpts[v3.LEFT_HIP, :2] + kpts[v3.RIGHT_HIP, :2]) / 2.0
            else:
                gone_for[track_id] = gone_for.get(track_id, 0) + 1

        # A track the tracker has dropped entirely: gone and not coming back inside this clip.
        for track_id in list(seen_for):
            if track_id in alive:
                continue
            if gone_for.get(track_id, 0) >= vanish_frames and seen_for[track_id] >= min_seen:
                out.append((seen_for[track_id] / fps, near_edge(last_hip[track_id]),
                            gone_for[track_id] / fps, 'dropped'))
            seen_for.pop(track_id, None)
            gone_for.pop(track_id, None)
            last_hip.pop(track_id, None)
    return out


def main():
    key = cache.cache_key()
    cache_dir = cache.cache_dir_for(key)
    if not os.path.exists(os.path.join(cache_dir, 'key.json')):
        raise SystemExit('no cached pose stream for this configuration: %s' % json.dumps(key))
    fps = key['fps']

    TRUTH = {'urfd_fall': 'fall', 'gmdcsa_fall': 'fall',
             'urfd_adl': 'no fall', 'val_adl': 'no fall', 'train50_adl': 'no fall'}
    rows = {}
    for group, paths, _half in cache.clip_groups():
        for path in paths:
            name = os.path.basename(path)
            npz = os.path.join(cache_dir, '%s__%s.npz' % (group, name))
            if not os.path.exists(npz):
                continue
            rows['%s/%s' % (group, name)] = {
                'truth': TRUTH[group],
                'vanishes': [[round(a, 2), bool(b), round(c, 2), d]
                             for a, b, c, d in vanishes(load_stream(npz), fps)]}
        print('  %-14s done' % group, flush=True)

    report(rows)
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump({'meta': {'vanish_seconds': VANISH_SECONDS, 'edge': EDGE,
                            'min_seen_seconds': MIN_SEEN_SECONDS, 'cache': key},
                   'clips': rows}, fh, indent=1)
    print('wrote %s' % OUT)


def report(rows):
    print()
    horizon_s = v3.MAX_MISSED_FRAMES / float(rows and next(iter(rows)) and 8 or 8)
    print('A tracked person gone long enough for the tracker to forget them (%d frames, '
          '%.1fs at this rate),' % (v3.MAX_MISSED_FRAMES, horizon_s))
    print('after being visible for at least %.0fs.' % MIN_SEEN_SECONDS)
    print('"at the edge" means the last hip seen was within %.0f%% of a frame border -- a '
          'doorway.' % (EDGE * 100))
    print()
    print('%-22s %7s %14s %16s %16s' % ('', 'clips', 'with a vanish', 'AT the edge',
                                        'AWAY from it'))
    print('-' * 80)
    for truth in ('fall', 'no fall'):
        keys = [k for k, v in rows.items() if v['truth'] == truth]
        if not keys:
            continue
        with_any = [k for k in keys if rows[k]['vanishes']]
        at_edge = [k for k in keys if any(v[1] for v in rows[k]['vanishes'])]
        away = [k for k in keys if any(not v[1] for v in rows[k]['vanishes'])]
        print('%-22s %7d %8d (%3.0f%%) %10d (%3.0f%%) %10d (%3.0f%%)'
              % (truth, len(keys), len(with_any), 100.0 * len(with_any) / len(keys),
                 len(at_edge), 100.0 * len(at_edge) / len(keys),
                 len(away), 100.0 * len(away) / len(keys)))
    fall_away = sum(1 for k, v in rows.items()
                    if v['truth'] == 'fall' and any(not x[1] for x in v['vanishes']))
    adl_away = sum(1 for k, v in rows.items()
                   if v['truth'] == 'no fall' and any(not x[1] for x in v['vanishes']))
    print()
    print('The column that decides it is the last one. If a person vanishing AWAY from the')
    print('edge happens about as often in clips with no fall as in clips with one, the idea')
    print('does not separate and no threshold rescues it.')
    print('  fall clips with such a vanish:    %d' % fall_away)
    print('  no-fall clips with such a vanish: %d' % adl_away)


if __name__ == '__main__':
    main()
