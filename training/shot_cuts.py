# -*- coding: utf-8 -*-
"""Where a video cuts from one shot to another. One definition, used by everything.

`Test/1`-`12` are social-media compilations that cut between unrelated incidents every few
seconds, and that is not a detail. Asking a question about "the clip" there is asking about
several different events at once: SS49 found Gemini answering about somebody else's fall
because a seven-second window spanned three scenes, and clip-based verification on compilation
footage is invalid without trimming to the shot the moment belongs to.

The detector: mean absolute difference between consecutive frames on a 64x36 grayscale
thumbnail. Small enough that ordinary motion and camera noise stay well under the threshold,
and a change of scene goes far above it. 45 is the value SS49 validated; it lives here so the
two places that cut on it cannot drift apart.

A cut is not the same thing as an incident -- one incident can span several shots, and a shot
can contain nothing at all -- so anything built on this has to be checked by eye before its
boundaries are treated as ground truth.
"""
import os

import cv2
import numpy as np

SHOT_CUT_DIFF = float(os.environ.get('SHOT_CUT_DIFF', 45.0))
# A fixed threshold misses cuts in dark footage, and it did: in Test/9 a cut between a porch
# camera and a doorbell fisheye measured 37.1 while every neighbouring frame pair sat at 1.0.
# A 37x outlier, unmistakable next to its neighbours, and invisible to a number tuned on
# brighter material -- so two unrelated scenes were merged into one "incident".
#
# A cut is therefore also anything far above the LOCAL noise floor. The two rules are a union,
# never an intersection, because the two failures are not equally bad: a spurious cut splits one
# incident into two, which a review notices and shrugs at, while a missed cut merges two
# incidents into one and quietly poisons whatever is measured on it.
REL_FLOOR = float(os.environ.get('SHOT_CUT_REL_FLOOR', 12.0))   # below this, never a cut
REL_K = float(os.environ.get('SHOT_CUT_REL_K', 8.0))            # times the local median
LOCAL_WINDOW = 30
THUMB = (64, 36)


def thumbnail(frame):
    return cv2.cvtColor(cv2.resize(frame, THUMB), cv2.COLOR_BGR2GRAY).astype(np.int16)


def frame_distance(a, b):
    """Mean absolute difference between two frames, on the thumbnail. 0 = identical."""
    return float(np.mean(np.abs(thumbnail(a) - thumbnail(b))))


def is_shot_cut(a, b, threshold=None):
    """Absolute rule only. Kept for callers that compare two frames with no context;
    shots() below adds the local-outlier rule, which needs the surrounding distances."""
    return frame_distance(a, b) > (SHOT_CUT_DIFF if threshold is None else threshold)


def shots(video_path, threshold=None, min_frames=1):
    """-> list of (start_frame, end_frame_exclusive, seconds) for each shot in the video.

    min_frames drops shots shorter than that outright: a one- or two-frame "shot" is a
    dissolve or a compression artefact, not something anybody filmed.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    absolute = SHOT_CUT_DIFF if threshold is None else threshold
    dists, prev = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if prev is not None:
            dists.append(frame_distance(prev, frame))
        prev = frame
    cap.release()

    bounds = [0]
    for i, d in enumerate(dists):
        if d > absolute:
            bounds.append(i + 1)
            continue
        # The local floor is taken from the frames around this one, excluding itself, so a
        # genuine cut cannot raise the bar it has to clear.
        lo = max(0, i - LOCAL_WINDOW)
        window = dists[lo:i] + dists[i + 1:i + 1 + LOCAL_WINDOW]
        if not window:
            continue
        local = sorted(window)[len(window) // 2]
        if d > REL_FLOOR and d > REL_K * local:
            bounds.append(i + 1)
    bounds.append(len(dists) + 1)
    bounds = sorted(set(bounds))
    out = []
    for start, end in zip(bounds, bounds[1:]):
        if end - start >= min_frames:
            out.append((start, end, (end - start) / fps))
    return out, fps
