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
THUMB = (64, 36)


def thumbnail(frame):
    return cv2.cvtColor(cv2.resize(frame, THUMB), cv2.COLOR_BGR2GRAY).astype(np.int16)


def frame_distance(a, b):
    """Mean absolute difference between two frames, on the thumbnail. 0 = identical."""
    return float(np.mean(np.abs(thumbnail(a) - thumbnail(b))))


def is_shot_cut(a, b, threshold=None):
    return frame_distance(a, b) > (SHOT_CUT_DIFF if threshold is None else threshold)


def shots(video_path, threshold=None, min_frames=1):
    """-> list of (start_frame, end_frame_exclusive, seconds) for each shot in the video.

    min_frames drops shots shorter than that outright: a one- or two-frame "shot" is a
    dissolve or a compression artefact, not something anybody filmed.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    bounds, prev, index = [0], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if prev is not None and is_shot_cut(prev, frame, threshold):
            bounds.append(index)
        prev, index = frame, index + 1
    cap.release()
    bounds.append(index)
    out = []
    for start, end in zip(bounds, bounds[1:]):
        if end - start >= min_frames:
            out.append((start, end, (end - start) / fps))
    return out, fps
