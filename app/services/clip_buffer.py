"""Keeps the last ~60 seconds of each camera in memory so an alert can ship the footage
leading UP TO the fall, not just the single frame it fired on.

A still image says "something happened"; the minute before it is what lets a caregiver tell
a real fall from someone lying down, and see whether the person moved afterwards. By the
time the detector fires, that minute is already in the past -- so it has to have been
retained all along, which is what this ring buffer does.

Memory is the reason frames are stored JPEG-encoded and downscaled rather than raw: one
minute of raw 1024x576 BGR at 17 fps is ~1.8 GB per camera, while the same minute as
640-wide JPEGs is roughly 30-60 MB. The buffer is capped by frame count and drops the oldest
frame on every append, so a camera that runs for days uses the same memory as one that just
started.
"""
import os
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np

# Seconds of history to keep, and the rate the clip is written back out at. Capturing every
# frame would make the buffer (and the resulting file) larger for no real benefit -- the
# point is for a person to watch what happened, and 10 fps is plenty for that.
CLIP_SECONDS = int(os.environ.get("CLIP_SECONDS", 60))
CLIP_FPS = int(os.environ.get("CLIP_FPS", 10))
CLIP_WIDTH = int(os.environ.get("CLIP_WIDTH", 640))
JPEG_QUALITY = int(os.environ.get("CLIP_JPEG_QUALITY", 70))

_buffers = {}
_lock = threading.Lock()


def _buffer_for(camera_id):
    with _lock:
        buf = _buffers.get(camera_id)
        if buf is None:
            buf = deque(maxlen=CLIP_SECONDS * CLIP_FPS)
            _buffers[camera_id] = buf
        return buf


def add_frame(camera_id, frame):
    """Called from the detection loop for every frame it processes.

    Cheap by design -- a resize plus a JPEG encode -- because it runs inline in the loop
    whose frame rate decides whether the detector keeps up with the camera at all. Frames
    are sampled down to CLIP_FPS using wall-clock time rather than a frame counter, so the
    clip plays back at real speed whether the loop is running at 5 fps or 25.
    """
    buf = _buffer_for(camera_id)
    now = time.time()
    if buf and (now - buf[-1][0]) < (1.0 / CLIP_FPS):
        return

    h, w = frame.shape[:2]
    if w > CLIP_WIDTH:
        scale = CLIP_WIDTH / float(w)
        frame = cv2.resize(frame, (CLIP_WIDTH, int(h * scale)))
    ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if ok:
        buf.append((now, encoded))


def _write_h264(path, frames, w, h):
    """Encode the buffered JPEGs to H.264 via ffmpeg. -> True on success.

    The frames are already JPEG, so they are piped straight in as an image2 stream instead
    of being decoded to raw arrays and re-encoded -- less CPU, and this runs on the same
    worker that has to keep up with a camera. yuv420p + faststart are what make the result
    play on phones and start before it has fully downloaded.
    """
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False

    cmd = [exe, '-y', '-loglevel', 'error',
           '-f', 'image2pipe', '-vcodec', 'mjpeg', '-framerate', str(CLIP_FPS), '-i', '-',
           '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', path]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        for _, encoded in frames:
            proc.stdin.write(encoded.tobytes())
        # close() then read stderr and wait -- NOT communicate(), which tries to flush the
        # stdin this just closed and raises "flush of closed file".
        proc.stdin.close()
        err = proc.stderr.read()
        proc.wait(timeout=120)
        if proc.returncode != 0:
            print(f'[clip] ffmpeg failed: {err.decode(errors="replace")[:300]}')
            return False
    except Exception as e:
        print(f'[clip] ffmpeg error: {e}')
        return False
    return os.path.isfile(path) and os.path.getsize(path) > 0


def save_clip(camera_id, out_dir, filename=None):
    """Write everything currently buffered for this camera to an mp4. -> path or None.

    Returns None rather than raising when there is nothing buffered (camera just started,
    or the loop never called add_frame): an alert that cannot carry a clip must still be
    delivered, so the caller treats the clip as optional.
    """
    buf = _buffer_for(camera_id)
    with _lock:
        frames = list(buf)
    if not frames:
        return None

    first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        return None
    h, w = first.shape[:2]

    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = f"fall_clip_{camera_id}_{int(time.time())}.mp4"
    path = os.path.join(out_dir, filename)

    if not _write_h264(path, frames, w, h):
        # Fall back to whatever cv2 can encode. The file will be mp4v, which many phones
        # refuse to play, so this is a last resort that keeps the clip available in the web
        # UI rather than losing it entirely.
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), CLIP_FPS, (w, h))
        if not writer.isOpened():
            return None
        for _, encoded in frames:
            img = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
        writer.release()

    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    return path


def clear(camera_id):
    """Drop a stopped camera's history so it isn't held for the life of the worker."""
    with _lock:
        _buffers.pop(camera_id, None)
