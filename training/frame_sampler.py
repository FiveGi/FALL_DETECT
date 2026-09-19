"""Feed a clip to the detector at the frame rate the deployment actually runs at.

Every evaluation script here used to read every frame of a 30fps file. That measures a
detector nobody deploys: the classifier's window is a fixed number of FRAMES, so at 30 fps it
spans 0.5s and at 15 fps it spans 1.0s, and the same model scores 43/60 URFD falls at one rate
and 17/60 at another (SS50). The live camera loop is pinned to `V3_TARGET_FPS`, so that is the
rate a measurement has to use to mean anything.

Selection is integer slot arithmetic rather than a float clock. A "next due time" comparison
was tried first and silently dropped ~9% of frames *even at target == source*, because the
equality landed on the wrong side of the rounding; it showed up as a known 43/60 reading 39/60.

Usage:

    from frame_sampler import sampled_frames, TARGET_FPS

    for frame in sampled_frames(path, rgb_half=True):
        ...

Set TARGET_FPS (or V3_TARGET_FPS) in the environment to measure another rate; pass
`target_fps=0` to read every frame, which is what the old scripts did.
"""
import os

import cv2

# Defaults to the rate docker-compose.gpu.yml pins the camera loop to.
TARGET_FPS = float(os.environ.get('TARGET_FPS', os.environ.get('V3_TARGET_FPS', 15)))


def sampled_frames(path, rgb_half=False, target_fps=None):
    """Yield the frames a camera loop running at `target_fps` would have seen.

    rgb_half: URFD's mp4s place a depth map and the colour image side by side, so only the
    right half is usable.
    """
    fps_target = TARGET_FPS if target_fps is None else target_fps
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, last_slot = 0, -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fps_target:
                slot = int(i * fps_target / src_fps)
                i += 1
                if slot == last_slot:
                    continue
                last_slot = slot
            else:
                i += 1
            yield frame[:, frame.shape[1] // 2:] if rgb_half else frame
    finally:
        cap.release()


def effective_fps(path, target_fps=None):
    """The rate frames are handed out at -- the target, or the clip's own rate if it is
    slower than the target, since sampling can only drop frames and never invent them."""
    fps_target = TARGET_FPS if target_fps is None else target_fps
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return min(src_fps, fps_target) if fps_target else src_fps
