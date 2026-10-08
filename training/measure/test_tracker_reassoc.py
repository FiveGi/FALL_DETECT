"""Deterministic MULTI-REASSOC-v1 contract tests; no models or video clips.

Run: python training/measure/test_tracker_reassoc.py
Crossing uses separated paths: centroid-only matching cannot identify people
who exchange positions between consecutive observations.
"""
import importlib.util
import itertools
from pathlib import Path
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "v3", ROOT / "app/detection/v3_fall_detection.py")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)


def update(tracker, *points):
    detections = [(np.zeros((17, 3)), np.array(p, dtype=float)) for p in points]
    result = tracker.update(detections)
    assert len({tid for tid, _, _ in result}) == len(result)
    assert {tid for tid, _, _ in result} == set(tracker.tracks)
    for tid, kpts, seen in result:
        assert seen == (tracker.tracks[tid]["missed"] == 0)
        assert (kpts is not None) == seen
        if seen:
            assert any(kpts is kp for kp, _ in detections)
        else:
            assert kpts is None
    return {tuple(tracker.tracks[tid]["centroid"]): tid
            for tid, _, seen in result if seen}


def lost(missed, reassoc=True):
    tracker = v3.PersonTracker(reassoc=reassoc)
    assert update(tracker, (0., 0.)) == {(0., 0.): 0}
    for _ in range(missed):
        assert update(tracker) == {}
    assert tracker.tracks[0]["missed"] == missed
    return tracker


def test_disabled_legacy_behavior():
    # Old sorted-ID greedy rule, including a lost lower ID winning over an
    # active higher ID. Expected IDs are fixed, not computed by the new matcher.
    tracker = v3.PersonTracker(reassoc=False)
    assert update(tracker, (0., 0.), (.20, 0.)) == {(0., 0.): 0, (.20, 0.): 1}
    assert update(tracker, (.20, 0.)) == {(.20, 0.): 1}
    assert update(tracker, (.12, 0.)) == {(.12, 0.): 0}
    assert tracker.tracks[1]["missed"] == 1
    for distance, expected in [(.149, 0), (.15, 1), (.155, 1)]:
        tracker = lost(1, reassoc=False)
        assert update(tracker, (distance, 0.)) == {(distance, 0.): expected}


def test_lost_reattaches_inside_wider_gate():
    tracker = lost(1)
    assert update(tracker, (.155, 0.)) == {(.155, 0.): 0}
    assert tracker.tracks[0]["missed"] == 0
    assert tracker.next_id == 1


def test_gate_boundary_and_beyond():
    for distance in (.16, .161, .4):
        tracker = lost(1)
        assert update(tracker, (distance, 0.)) == {(distance, 0.): 1}
        assert tracker.tracks[0]["missed"] == 2


def test_gate_growth_and_cap():
    assert v3.MAX_TRACK_DISTANCE == .15
    assert v3.REASSOC_GATE_STEP == .01
    assert v3.REASSOC_GATE_MAX == .30
    assert v3.REASSOC_MARGIN == .05
    # Extend lifetime only here to observe saturation beyond m=15; update()
    # still ages every track naturally. Restore before the expiry tests.
    original = v3.MAX_MISSED_FRAMES
    try:
        v3.MAX_MISSED_FRAMES = max(original, 20)
        for missed, distance, expected in [
            (1, .165, 1), (2, .165, 0),
            (5, .199, 0), (5, .201, 1),
            (14, .295, 1), (15, .295, 0),
            (15, .30, 1), (20, .299, 0), (20, .301, 1),
        ]:
            tracker = lost(missed)
            actual = update(tracker, (distance, 0.))
            assert actual == {(distance, 0.): expected}, (missed, distance, actual)
    finally:
        v3.MAX_MISSED_FRAMES = original


def test_crossing_without_swap():
    tracker = v3.PersonTracker(reassoc=True)
    # Paths cross in x, separated by 0.20 in y. Reverse detector ordering
    # every frame; temporarily lose A during the crossing, then reacquire it.
    for frame in range(9):
        a = (.1 + .04 * frame, .4)
        b = (.42 - .04 * frame, .6)
        points = [b] if frame in (3, 4, 5) else [a, b]
        if frame % 2:
            points.reverse()
        expected = {b: 1} if len(points) == 1 else {a: 0, b: 1}
        assert update(tracker, *points) == expected, frame


def test_competing_lost_tracks_birth():
    for target in (.20, .21):  # Exact tie and non-tie inside the 0.05 margin.
        tracker = v3.PersonTracker(reassoc=True)
        update(tracker, (0., 0.), (.4, 0.))
        for _ in range(10):
            update(tracker)
        assert update(tracker, (target, 0.)) == {(target, 0.): 2}
        assert tracker.tracks[0]["missed"] == 11
        assert tracker.tracks[1]["missed"] == 11


def test_one_lost_track_ambiguous_detections_birth():
    tracker = lost(10)
    assert update(tracker, (.20, 0.), (.22, 0.)) == {(.20, 0.): 1, (.22, 0.): 2}
    assert tracker.tracks[0]["missed"] == 11


def test_previous_visible_centroid_protected():
    tracker = v3.PersonTracker(reassoc=True)
    update(tracker, (0., 0.), (.30, 0.))
    for _ in range(10):
        assert update(tracker, (.30, 0.)) == {(.30, 0.): 1}
    # Active B takes .31; leftover .20 is within the lost A gate and would
    # otherwise be an unambiguous match. B's PREVIOUS centroid protects it.
    assert update(tracker, (.31, 0.), (.20, 0.)) == {(.31, 0.): 1, (.20, 0.): 2}
    assert tracker.tracks[0]["missed"] == 11


def test_detection_order_invariance():
    # Fix initial identities. Birth numbering follows input order by contract;
    # compare retained identities and birth locations, not new numeric IDs.
    for points in itertools.permutations([(.20, 0.), (.80, 0.), (.5, .6)]):
        tracker = v3.PersonTracker(reassoc=True)
        update(tracker, (0., 0.), (1., 0.), (.5, .6))
        for _ in range(10):
            update(tracker, (.5, .6))
        assert update(tracker, *points) == {(.20, 0.): 0, (.80, 0.): 1, (.5, .6): 2}
    for points in itertools.permutations([(.20, 0.), (.22, 0.)]):
        tracker = lost(10)
        actual = update(tracker, *points)
        assert set(actual) == {(.20, 0.), (.22, 0.)}
        assert set(actual.values()) == {1, 2}
        assert tracker.tracks[0]["missed"] == 11


def test_expiry_after_max_missed_frames():
    for enabled in (False, True):
        tracker = lost(v3.MAX_MISSED_FRAMES, reassoc=enabled)
        assert update(tracker, (0., 0.)) == {(0., 0.): 0}
        tracker = lost(v3.MAX_MISSED_FRAMES, reassoc=enabled)
        assert update(tracker) == {}
        assert tracker.tracks == {}
        assert update(tracker, (0., 0.)) == {(0., 0.): 1}


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run without -O: these tests require plain asserts")
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception:
            failures += 1
            print(f"FAIL {test.__name__}", flush=True)
            traceback.print_exc()
        else:
            print(f"PASS {test.__name__}", flush=True)
    print(f"{len(tests) - failures}/{len(tests)} passed; {failures} failed; synthetic only, 0 clips")
    raise SystemExit(bool(failures))
