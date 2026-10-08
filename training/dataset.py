"""Windowed dataset built from per-video pose keypoint sequences, pooled from two
sources with different keypoint sets:
  - extract_poses.py: MediaPipe's 33-point BlazePose landmarks (GMDCSA24 videos)
  - parse_fallvision.py: pre-extracted COCO-17 keypoints (FallVision dataset)

Everything is unified onto the shared COCO-17 keypoint set (MediaPipe's 33 points
is a superset that includes all 17 COCO body points, so GMDCSA24 sequences are
downsampled to match rather than the other way around) so both datasets can be
windowed and trained on together.

Each video becomes several overlapping fixed-length windows. Keypoints are
normalized per-frame relative to the hip center and torso size so the model
is robust to where the person stands in the frame and how far from the
camera they are.
"""
import json
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

COCO17_ORDER = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]
NUM_KEYPOINTS = len(COCO17_ORDER)  # 17

# MediaPipe BlazePose (33 points) index for each COCO17_ORDER entry, in order.
MEDIAPIPE33_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12

# 15, matching the deployed model and app/detection/v3_fall_detection.WINDOW_SIZE. With
# TEMPORAL_STRIDE=2 below, a window covers 1.0s of real time at the 15 fps the camera
# loop is pinned to. Window size, temporal stride, runtime window and V3_TARGET_FPS are
# one decision; tools/check_config_coherence.py fails if they drift apart.
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 15))
# Window step. 10 was set when clips were 30 fps; at RESAMPLE_FPS=8 a 3 s clip is 24 frames, and
# a step of 10 leaves one or two windows per clip. WINDOW_STEP overrides it.
STRIDE = int(os.environ.get("WINDOW_STEP", 10))

# TEMPORAL_STRIDE: keep every k-th frame, to match the frame rate the live pipeline can
# actually sustain (see this module's git history / SKILL.md -- offline eval sees 30fps,
# a real camera loop sees ~1-5fps). 1 = original behaviour, every frame.
TEMPORAL_STRIDE = int(os.environ.get("TEMPORAL_STRIDE", 2))   # train on every 2nd frame
                                                             # of 30fps source = 15 fps

# RESAMPLE_FPS: resample every file to this rate BY ITS OWN TIMESTAMPS before velocity is
# computed, instead of TEMPORAL_STRIDE's blanket "keep every k-th frame" (which replaces it).
# A blanket stride assumes every source is 30 fps; measured 2026-10-01 they are not: CAUCAFall
# is 20 fps, and FallVision -- 58% of the files -- mixes 15/24/30/60/120 fps, so at stride 2 a
# 120 fps clip was trained as 60 fps (its 15-frame window spanned 0.25 s). 8 = the CPU camera
# loop. Each file's native rate comes from an `fps` key in the file or from
# data/source_fps.json (build_source_fps.py); a file with neither is an error, not a guess.
RESAMPLE_FPS = float(os.environ.get("RESAMPLE_FPS", 0))
_SOURCE_FPS = None


_EXCLUDED = None


def source_fps(pose_dir, name, data):
    """Native rate, or None for a file build_source_fps.py excluded (no usable time base)."""
    global _SOURCE_FPS, _EXCLUDED
    if "fps" in data:
        return float(data["fps"])
    if _SOURCE_FPS is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "source_fps.json")
        j = json.load(open(p)) if os.path.exists(p) else {}
        _SOURCE_FPS, _EXCLUDED = j.get("fps", {}), set(j.get("excluded", []))
    key = "%s/%s" % (os.path.basename(os.path.normpath(pose_dir)), name)
    if key in _EXCLUDED:
        return None
    if RESAMPLE_FPS <= 0 and key not in _SOURCE_FPS:
        return 0.0   # rate unknown but not needed without resampling; only exclusion matters here
    if key not in _SOURCE_FPS:
        raise KeyError("RESAMPLE_FPS is set but the native rate of %s is unknown "
                       "(run build_source_fps.py, or store an `fps` key)" % key)
    return float(_SOURCE_FPS[key])


# RUNTIME_MISSES=1: replay frames with nobody detected the way the camera loop does
# (v3_fall_detection._step_person), not as zeros. At runtime a miss repeats the last real pose
# for up to MAX_HELD_RUN frames, then adds nothing (the window freezes); only before anyone has
# been seen are zeros used. Every extractor here writes zeros for a miss, so training saw a
# real->zero->real jump the runtime never produces (Codex review, 2026-10-01, P1). Applied after
# resampling, because the runtime counts misses in frames it actually sampled.
RUNTIME_MISSES = os.environ.get("RUNTIME_MISSES", "0") == "1"
_MAX_HELD_RUN = None


def max_held_run():
    """MAX_HELD_RUN read from the runtime module's source, so the two cannot drift apart."""
    global _MAX_HELD_RUN
    if _MAX_HELD_RUN is None:
        import re
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "app", "detection", "v3_fall_detection.py")
        m = re.search(r'MAX_HELD_RUN = int\(os\.environ\.get\("V3_MAX_HELD_RUN", (\d+)\)\)', open(p).read())
        if not m:
            raise RuntimeError("MAX_HELD_RUN not found in %s -- update dataset.max_held_run()" % p)
        _MAX_HELD_RUN = int(os.environ.get("V3_MAX_HELD_RUN", m.group(1)))
    return _MAX_HELD_RUN


def runtime_miss_indices(raw):
    """raw (T,17,3) with all-zero frames for misses -> (frames, source_index) as the runtime
    buffer would hold them: held copies of the last real pose, frozen past MAX_HELD_RUN."""
    found = raw[:, :, 2].max(axis=1) > 0
    out, src, last, held = [], [], None, 0
    for i in range(len(raw)):
        if found[i]:
            last, held = i, 0
            out.append(raw[i]); src.append(i)
        else:
            held += 1
            if last is None:
                out.append(raw[i]); src.append(i)          # nothing real yet: zeros
            elif held <= max_held_run():
                out.append(raw[last]); src.append(i)       # held copy, labelled as this frame
            # else: frozen -- the frame contributes nothing
    if not out:
        return raw, np.arange(len(raw))
    return np.stack(out), np.asarray(src)


def resample_indices(n_frames, native_fps, target_fps, offset=0.0):
    """Frame indices a camera loop sampling at target_fps would see, by timestamp."""
    duration = n_frames / native_fps
    t = np.arange(offset / target_fps, duration, 1.0 / target_fps)
    return np.clip(np.round(t * native_fps).astype(int), 0, n_frames - 1)

# USE_HIP_MOTION: add the hip centre's own frame-to-frame displacement as two extra input
# channels, scaled by torso size so it stays camera-distance invariant.
#
# Why this is worth trying: normalize_sequence() pins the hip to the origin every frame and
# add_velocity() then differences the *normalized* coordinates, so the model never sees the
# body travelling through the frame -- only limbs moving relative to the hip. A fall and a
# deliberate lie-down produce nearly the same limb-relative shape change; what separates them
# is how fast and far the whole body drops, which is exactly what the normalization removes.
# compute_motion_energy() below already notes this, but its output is only used to pick the
# labelling peak, never fed to the classifier.
#
# SS33 tested hip drop and velocity spikes as standalone decision RULES and found the
# distributions overlap. This is a different experiment: as an input channel the classifier
# can combine it with pose shape ("torso horizontal AND the body dropped fast") rather than
# having to separate the classes on the signal alone.
#
# Channels are duplicated across all 17 joints so every existing (T, K, feat_dim) reshape --
# the flip and occlusion augmentations especially -- keeps working unchanged.
# "1" = raw per-frame displacement in x and y. "dy" = vertical only, averaged over a short
# window: most training clips come from hand-held or panning footage, where per-frame hip
# displacement largely measures the camera rather than the person, and panning is mostly
# horizontal. A fall is a drop, so the vertical component over ~0.3s is the part with a
# chance of surviving that noise.
HIP_MOTION = os.environ.get("USE_HIP_MOTION", "0")
USE_HIP_MOTION = HIP_MOTION in ("1", "dy")

# USE_FRAME_POSITION: append where the person IS in the frame -- the hip centre's height and
# the apparent torso size -- as two more channels.
#
# Why: normalize_sequence() subtracts the hip centre and divides by torso size every frame, so
# the classifier cannot tell lying on a bed from lying on the floor. Both are a horizontal
# body, and once centred they are the same picture. What separates them is height in the frame
# (a bed is raised) and apparent scale (how far away the person is), and normalisation is
# exactly what throws both away. Six of the seven GMDCSA24 val false alarms are beds, and they
# have survived every decision rule tried on top of the classifier.
#
# Neither channel changes under a horizontal flip, so the flip augmentation needs no new case
# for them -- the hip's *horizontal* position was deliberately left out, both for that reason
# and because it carries no physical meaning, only room layout.
#
# THE RISK IS OVERFITTING, NOT SPEED. Absolute position lets the model memorise the four rooms
# GMDCSA24 was filmed in rather than learn that beds are higher than floors. Train three seeds
# and let URFD -- filmed somewhere else entirely -- decide: if URFD does not improve, it
# learned the rooms.
USE_FRAME_POSITION = os.environ.get("USE_FRAME_POSITION", "0") == "1"

# Channel order is [x, y, conf, vx, vy] + [hip dx, hip dy] + [hip height, torso size], each
# optional block appended only when its flag is on, so an index means the same thing whichever
# combination is selected.
FEAT_DIM = 5 + (2 if USE_HIP_MOTION else 0) + (2 if USE_FRAME_POSITION else 0)
HIP_MOTION_AT = 5 if USE_HIP_MOTION else None
FRAME_POSITION_AT = (7 if USE_HIP_MOTION else 5) if USE_FRAME_POSITION else None

# Index to swap with for a left-right mirror flip (self-pairs for Nose, which has no
# left/right counterpart). Used by flip_horizontal_window() below.
FLIP_PAIRS = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


# COORD_SCALE: the coordinate space the stored keypoints are in, by pose directory.
#
# Everything here except FallVision stores x and y already normalised to [0,1] of the frame,
# because extract_poses.py and yolopose_extractor.py divide by the frame size on the way out.
# FallVision ships pre-extracted keypoint CSVs in PIXELS and parse_fallvision.py copies them
# through unchanged -- and it is 5845 of the 10102 training videos, 58% of the set.
#
# **This never mattered before.** normalize_sequence() divides by a torso size measured in the
# same units, so any uniform scale cancels exactly and all five original channels are identical
# either way. USE_FRAME_POSITION is the first thing that reads a raw coordinate for its own
# sake, and unscaled it would hand the classifier a hip height around 200 for FallVision and
# around 0.5 for everything else: a dataset-identity flag with four hundred times the magnitude
# of the signal it is supposed to carry, and a value the runtime can never produce.
#
# 640 is measured, not taken from the dataset's documentation, which does not state it:
#   - over 2.4M detected keypoints, 99.6% of x and 99.99% of y land inside [0, 640], and what
#     lies outside is the small negative and over-range excursion a pose model extrapolates.
#   - the space is square rather than letterboxed: 35% of y values fall outside the [140, 500]
#     band that a 16:9 image would occupy inside a 640x640 letterbox.
#   - dividing by it puts FallVision's torso size on top of every other directory's -- median
#     0.111 against 0.113-0.156 elsewhere -- and torso size as a fraction of frame width is a
#     physical quantity that should agree across datasets framed similarly.
#
# The control that keeps this honest: train one arm with FallVision dropped entirely
# (`SKIP_POSE_DIRS=poses_fallvision`). If the scale were wrong, the two would disagree.
COORD_SCALE = {"poses_fallvision": 640.0}


def to_coco17(raw_seq):
    """raw_seq: (T, 33, 3) MediaPipe or (T, 17, 3) already-COCO17 -> (T, 17, 3) COCO17."""
    if raw_seq.shape[1] == NUM_KEYPOINTS:
        return raw_seq
    if raw_seq.shape[1] == 33:
        return raw_seq[:, MEDIAPIPE33_TO_COCO17, :]
    raise ValueError(f"Unexpected keypoint count: {raw_seq.shape[1]}")


def normalize_sequence(seq):
    """seq: (T, 17, 3) -> (T, 17, 3) normalized (x, y relative to torso, confidence untouched)."""
    xy = seq[:, :, :2]
    vis = seq[:, :, 2:3]

    hip_center = (xy[:, LEFT_HIP] + xy[:, RIGHT_HIP]) / 2.0  # (T, 2)
    shoulder_center = (xy[:, LEFT_SHOULDER] + xy[:, RIGHT_SHOULDER]) / 2.0  # (T, 2)
    torso_size = np.linalg.norm(shoulder_center - hip_center, axis=1)  # (T,)
    torso_size = np.clip(torso_size, 1e-3, None)

    xy_norm = (xy - hip_center[:, None, :]) / torso_size[:, None, None]
    return np.concatenate([xy_norm, vis], axis=-1)


def hip_motion(raw_seq):
    """raw_seq: (T, 17, 3) RAW keypoints -> (T, 2) hip-centre displacement per frame,
    divided by torso size so a person far from the camera and a person near it give the same
    number for the same real movement."""
    xy = raw_seq[:, :, :2]
    hip_center = (xy[:, LEFT_HIP] + xy[:, RIGHT_HIP]) / 2.0
    shoulder_center = (xy[:, LEFT_SHOULDER] + xy[:, RIGHT_SHOULDER]) / 2.0
    torso_size = np.clip(np.linalg.norm(shoulder_center - hip_center, axis=1), 1e-3, None)
    delta = np.diff(hip_center, axis=0, prepend=hip_center[:1]) / torso_size[:, None]
    if HIP_MOTION == "dy":
        dy = delta[:, 1]
        # ~0.3s at 30fps, shrunk to fit a short sequence: np.convolve(mode="same") returns
        # max(len(a), len(kernel)) samples, so a kernel longer than the clip changes its length.
        width = min(9, len(dy) if len(dy) % 2 else len(dy) - 1)
        if width >= 3:
            dy = np.convolve(dy, np.ones(width) / width, mode="same")
        delta = np.stack([np.zeros_like(dy), dy], axis=1)
    return delta


def frame_position(raw_seq):
    """raw_seq: (T, 17, 3) raw keypoints normalised to the frame -> (T, 2).

    Column 0 is the hip centre's height in the frame (0 at the top, 1 at the bottom) and
    column 1 is the apparent torso size as a fraction of the frame. Both are read from the RAW
    sequence, before normalize_sequence() removes them.
    """
    xy = raw_seq[:, :, :2]
    hip_center = (xy[:, LEFT_HIP] + xy[:, RIGHT_HIP]) / 2.0
    shoulder_center = (xy[:, LEFT_SHOULDER] + xy[:, RIGHT_SHOULDER]) / 2.0
    torso = np.clip(np.linalg.norm(shoulder_center - hip_center, axis=1), 1e-3, None)
    return np.stack([hip_center[:, 1], torso], axis=1)


def add_velocity(norm_seq, raw_seq=None):
    """norm_seq: (T, 17, 3) torso-normalized [x, y, confidence].
    Returns (T, 17, 5): [x, y, confidence, vx, vy] where velocity is the frame-to-frame
    change in normalized position. Giving the model velocity directly (rather than making
    it infer motion from a stack of raw positions) makes the fall-vs-calm-movement signal
    explicit instead of implicit.

    With USE_HIP_MOTION, returns (T, 17, 7) with the hip centre's own displacement appended,
    repeated across joints -- see the flag's comment above.
    """
    xy = norm_seq[:, :, :2]
    vel = np.diff(xy, axis=0, prepend=xy[:1])  # (T, 17, 2), first frame velocity = 0
    out = np.concatenate([norm_seq, vel], axis=-1)
    if USE_HIP_MOTION:
        if raw_seq is None:
            raise ValueError("USE_HIP_MOTION needs the raw sequence to measure hip movement")
        hm = hip_motion(raw_seq)[:, None, :]              # (T, 1, 2)
        out = np.concatenate([out, np.repeat(hm, out.shape[1], axis=1)], axis=-1)
    if USE_FRAME_POSITION:
        if raw_seq is None:
            raise ValueError("USE_FRAME_POSITION needs the raw sequence for frame coordinates")
        fp = frame_position(raw_seq)[:, None, :]          # (T, 1, 2)
        out = np.concatenate([out, np.repeat(fp, out.shape[1], axis=1)], axis=-1)
    return out


def compute_motion_energy(raw_seq, smooth=5):
    """raw_seq: (T, 17, 3) RAW (un-normalized) keypoints, image-relative coords.
    Returns (T,) motion energy per frame = mean frame-to-frame landmark displacement,
    smoothed with a short moving average to avoid single-frame detection jitter.
    Must be computed BEFORE torso-relative normalization, since normalization
    pins the hip to the origin every frame and erases real movement in the frame.
    """
    xy = raw_seq[:, :, :2]
    diffs = np.linalg.norm(np.diff(xy, axis=0), axis=2).mean(axis=1)  # (T-1,)
    energy = np.concatenate([[0.0], diffs])  # (T,)
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        energy = np.convolve(energy, kernel, mode="same")
    return energy


def load_all_videos(pose_dirs):
    """pose_dirs: a directory path or list of directory paths containing .npz files
    (either MediaPipe-33 or COCO-17 format -- auto-detected per file).
    Returns list of dicts: {keypoints (T,17,5) [x,y,conf,vx,vy], motion (T,), label, subject, name}.
    """
    if isinstance(pose_dirs, str):
        pose_dirs = [pose_dirs]

    videos = []
    n_excluded = 0
    n_slow = 0
    for pose_dir in pose_dirs:
        scale = COORD_SCALE.get(os.path.basename(os.path.normpath(pose_dir)))
        for path in sorted(glob.glob(os.path.join(pose_dir, "*.npz"))):
            data = np.load(path, allow_pickle=True)
            raw = to_coco17(data["keypoints"].astype(np.float32))
            if scale:
                # Into the same [0,1]-of-the-frame space every other directory is already in,
                # and the space the runtime feeds. See COORD_SCALE above.
                raw = raw.copy()
                raw[:, :, :2] /= scale
            idx = None
            if RESAMPLE_FPS <= 0 and os.environ.get("EXCLUDE_NO_TIMEBASE") == "1"                     and source_fps(pose_dir, os.path.basename(path), data) is None:
                # Ablation A (Codex): stride-based training on the SAME clip membership as the
                # resampled runs, so cadence is not confounded with the 495 excluded files.
                n_excluded += 1
                continue
            if RESAMPLE_FPS > 0:
                native = source_fps(pose_dir, os.path.basename(path), data)
                if native is None:
                    n_excluded += 1
                    continue
                idx = resample_indices(len(raw), native, RESAMPLE_FPS)
                raw = raw[idx]
            elif TEMPORAL_STRIDE > 1 and "fps" in data:
                # A file that states its own rate (new extractions: Le2i, OF-Syn) is not a 30 fps
                # recording, so a blanket stride would halve an already-reduced rate (8 -> 4 fps).
                # Bring it to the stride's nominal rate (30 / stride) by timestamp, never upward.
                # Tolerance 0.1 fps absorbs rounding of a stated rate (15.0 vs 14.99) only; anything
                # faster is resampled down (Codex review: 0.5 let 15.25-15.5 fps through unchanged).
                nominal = 30.0 / TEMPORAL_STRIDE
                if float(data["fps"]) > nominal + 0.1:
                    idx = resample_indices(len(raw), float(data["fps"]), nominal)
                    raw = raw[idx]
                else:
                    n_slow += 1
            elif TEMPORAL_STRIDE > 1:
                # Before velocity: vx/vy must be the delta between two frames the runtime
                # actually sees in sequence, not between two 30fps neighbours.
                raw = raw[::TEMPORAL_STRIDE]
            keep = None
            if RUNTIME_MISSES:
                raw, keep = runtime_miss_indices(raw)
            motion = compute_motion_energy(raw)
            seq = add_velocity(normalize_sequence(raw), raw_seq=raw)
            video = {
                "keypoints": seq,
                "motion": motion,
                "label": int(data["label"]),
                "subject": str(data["subject"]),
                "name": os.path.basename(path),
                "source": os.path.basename(os.path.normpath(pose_dir)),
            }
            if "frame_labels" in data:
                # Real per-frame ground truth (CAUCAFall) -- used instead of the
                # peak-motion heuristic when available, since it's not a guess.
                # Strided exactly like the keypoints above. Until 2026-10-01 they were not
                # (Codex, DATA-DESIGN #9): at TEMPORAL_STRIDE=2 a window centred on keypoint
                # frame c read the label of ORIGINAL frame c, i.e. from the first half of the
                # clip's timeline. Measured on the deployed recipe: 17% of CAUCAFall and 18.5%
                # of GMDCSA24 windows mislabelled, 9/50 and 6/79 fall clips with no fall
                # window at all; OF-ItW unaffected (its clips carry one label throughout).
                fl = data["frame_labels"].astype(np.int64)
                # Exactly the selection applied to the keypoints above: resampled indices, a blanket
                # stride, or nothing (a stated-rate file already at or below the nominal rate).
                if idx is not None:
                    fl = fl[idx]
                elif TEMPORAL_STRIDE > 1 and "fps" not in data:
                    fl = fl[::TEMPORAL_STRIDE]
                if keep is not None:
                    fl = fl[keep]
                video["frame_labels"] = fl
            videos.append(video)
    if n_slow:
        print(f"TEMPORAL_STRIDE: {n_slow} files with a stated rate at or below {30.0 / TEMPORAL_STRIDE:.1f} fps used as they are")
    if n_excluded:
        print(f"RESAMPLE_FPS: skipped {n_excluded} files with no usable time base (source_fps.json 'excluded')")
    return videos


def make_windows(videos, window_size=WINDOW_SIZE, stride=STRIDE, onset_buffer=None):
    """Slice each video into overlapping windows. Short videos are padded by repeating the last frame.

    Per-window labels come from one of two sources:
      - Real per-frame ground truth (CAUCAFall's "frame_labels"), when present: the
        window is labeled 1 if its center frame is annotated as a fall frame. This is
        exact, not a guess.
      - Otherwise (GMDCSA24, FallVision -- only a whole-clip label exists), windows from
        just-before the frame of peak motion energy (the fall event) onward are labeled
        1, and windows well before that (normal standing/walking) are labeled 0. Without
        this, every window in a Fall clip would get labeled "fall" even though a large
        part of the clip shows normal standing beforehand.
    """
    if onset_buffer is None:
        onset_buffer = window_size // 2

    samples = []  # (features (window_size, F), label)
    for v in videos:
        seq = v["keypoints"]  # (T, 17, 5)
        T = seq.shape[0]
        flat = seq.reshape(T, -1)  # (T, F)
        motion = v["motion"]
        frame_labels = v.get("frame_labels")

        peak_frame = int(np.argmax(motion)) if (frame_labels is None and v["label"] == 1) else None

        if T < window_size:
            pad_amount = window_size - T
            flat = np.concatenate([flat, np.repeat(flat[-1:], pad_amount, axis=0)], axis=0)
            motion = np.concatenate([motion, np.repeat(motion[-1:], pad_amount, axis=0)])
            if frame_labels is not None:
                frame_labels = np.concatenate([frame_labels, np.repeat(frame_labels[-1:], pad_amount, axis=0)])
            T = window_size

        starts = list(range(0, T - window_size + 1, stride))
        if not starts:
            starts = [0]
        for s in starts:
            center = s + window_size // 2
            if frame_labels is not None:
                label = int(frame_labels[min(center, T - 1)])
            elif v["label"] == 1:
                label = 1 if center >= (peak_frame - onset_buffer) else 0
            else:
                label = 0
            samples.append((flat[s:s + window_size], label, v["name"]))
    return samples


def flip_horizontal_window(feat, num_keypoints=NUM_KEYPOINTS, feat_dim=None):
    """feat: (window_size, num_keypoints*feat_dim) flat, [x,y,conf,vx,vy] per joint,
    torso-relative normalized (hip centered at x=0) -> left-right mirrored window.
    A fall is not inherently left- or right-handed, so this doubles effective training
    data for free -- standard augmentation for pose classification, not yet tried in
    any of this session's training runs (SS17-34)."""
    feat_dim = FEAT_DIM if feat_dim is None else feat_dim
    T = feat.shape[0]
    seq = feat.reshape(T, num_keypoints, feat_dim).copy()
    seq = seq[:, FLIP_PAIRS, :]
    seq[:, :, 0] = -seq[:, :, 0]   # x
    seq[:, :, 3] = -seq[:, :, 3]   # vx
    # Keyed off the flag rather than off feat_dim: with USE_FRAME_POSITION alone the feature
    # dimension is also 7, but channel 5 is then the hip's HEIGHT, which a horizontal flip
    # must leave alone. Negating it would have taught the model that beds are upside down.
    if HIP_MOTION_AT is not None:
        seq[:, :, HIP_MOTION_AT] = -seq[:, :, HIP_MOTION_AT]   # hip dx travels the other way
    return seq.reshape(T, -1)


# TRUNC_AUG (TRUNC-v1, Codex design 2026-10-02 + P4 decision 2026-10-04; agreed by the team, Gemini's
# full-cap variant measured as TRUNC_FULL=1): near-camera falls lose joints to the frame edge or to an
# occluder, often progressively as the body slides out of view. With probability 0.25 per training window:
#   80% frame edge (bottom/top/left/right 0.5/0.1/0.2/0.2): joints ranked by their temporal-median
#        coordinate toward that edge (ties by COCO index), cap 4 or 6 equiprobably;
#   20% occlusion of one limb group {5,7,9} {6,8,10} {11,13,15} {12,14,16}, cap 3, ascending index;
#   then 50% static (the capped joints hidden in all frames) or 50% progressive (onset uniform 0..10,
#   one joint at onset, one more each frame through frame 14, up to the cap).
# Hidden joints get x, y, conf, vx, vy = 0; the global hip/frame channels (5+) are kept. Whole-person
# loss is NOT simulated here: RUNTIME_MISSES already trains the runtime's hold/freeze. Own RNG, so the
# control arm's sampling is unchanged.
TRUNC_AUG = os.environ.get("TRUNC_AUG", "0") == "1"
TRUNC_FULL = os.environ.get("TRUNC_FULL", "0") == "1"
_TRUNC_RNG = np.random.default_rng(int(os.environ.get("TRAIN_SEED", 42)) + 7919)
_LIMB_GROUPS = ([5, 7, 9], [6, 8, 10], [11, 13, 15], [12, 14, 16])


def trunc_window(feat, num_keypoints=NUM_KEYPOINTS, feat_dim=None, rng=None):
    rng = _TRUNC_RNG if rng is None else rng
    if rng.random() >= 0.25:
        return feat
    feat_dim = FEAT_DIM if feat_dim is None else feat_dim
    T = feat.shape[0]
    seq = feat.reshape(T, num_keypoints, feat_dim).copy()
    if rng.random() < 0.8:
        edge = rng.choice(4, p=[0.5, 0.1, 0.2, 0.2])            # bottom, top, left, right
        axis, sign = {0: (1, 1), 1: (1, -1), 2: (0, -1), 3: (0, 1)}[int(edge)]
        med = np.median(seq[:, :, axis], axis=0) * sign
        order = sorted(range(num_keypoints), key=lambda j: (-med[j], j))
        cap = int(rng.choice([4, 6]))
    else:
        order = list(_LIMB_GROUPS[int(rng.integers(4))])
        cap = 3
    progressive = rng.random() < 0.5
    full = progressive and TRUNC_FULL
    if full:
        # Gemini's variant (Codex review 2026-10-04 P2): the whole skeleton must be reachable, so the
        # order covers all 17 joints (limb group first, then the rest by index) and the growth rate is
        # set so the last frame hides all of them, whatever the onset.
        order = list(order) + [j for j in range(num_keypoints) if j not in order]
        cap = num_keypoints
    hidden = np.zeros((T, num_keypoints), dtype=bool)
    if progressive:
        onset = int(rng.integers(0, 11))
        for t in range(onset, T):
            n = t - onset + 1
            if full:
                n = int(np.ceil(n * num_keypoints / (T - onset)))
            hidden[t, order[:min(cap, n)]] = True
    else:
        hidden[:, order[:cap]] = True
    seq[:, :, :5][hidden] = 0.0
    return seq.reshape(T, -1)


def occlude_window(feat, num_keypoints=NUM_KEYPOINTS, feat_dim=None, prob=0.3, max_joints=3, max_span=8):
    """feat: (window_size, num_keypoints*feat_dim) flat array.
    With probability `prob`, simulates a brief per-joint tracking dropout (a common real
    failure mode -- e.g. an elbow/wrist occluded by the torso during a twisting fall,
    the exact pattern SS34 diagnosed as clip14's likely weak spot) by freezing a few
    joints' position for a short run of frames and zeroing their velocity there, mirroring
    how _step_person holds the last known state rather than zero-filling on a dropout."""
    if np.random.rand() > prob:
        return feat
    feat_dim = FEAT_DIM if feat_dim is None else feat_dim
    T = feat.shape[0]
    seq = feat.reshape(T, num_keypoints, feat_dim).copy()
    n_joints = np.random.randint(1, max_joints + 1)
    joints = np.random.choice(num_keypoints, size=n_joints, replace=False)
    span = min(np.random.randint(2, max_span + 1), T)
    start = np.random.randint(0, max(1, T - span + 1))
    freeze_xy = seq[max(0, start - 1), joints, :2].copy()
    for j_idx, j in enumerate(joints):
        seq[start:start + span, j, 0] = freeze_xy[j_idx, 0]
        seq[start:start + span, j, 1] = freeze_xy[j_idx, 1]
        seq[start:start + span, j, 3] = 0.0  # vx
        seq[start:start + span, j, 4] = 0.0  # vy
    return seq.reshape(T, -1)


# USE_IR_AUG: replay REAL night-vision skeleton errors onto training windows.
#
# On simulated CCTV night footage the deployed detector catches 25-33% of URFD falls against 75%
# by day, and the pose model still finds the person in most frames -- so the classifier is being
# handed a skeleton that is present but wrong. training/measure/ir_keypoint_degradation.py
# measured how: the person is lost in ~12% of frames, and joints including the hips jump by up
# to ~0.29 torso lengths at p90 with the pose model's confidence unchanged. The hip jumps matter
# most here, because every feature is measured relative to the hip, so one misplaced hip moves
# the whole normalised skeleton and reads as motion.
#
# The errors are not modelled, they are replayed: training/data/ir_residual_bank.npz holds the
# measured per-frame night-minus-day displacement of every joint (build_ir_residual_bank.py),
# from GMDCSA24 training-side clips only, so URFD -- the held-out set that judges this -- never
# contributed an error. Off by default, so the deployed recipe is unchanged.
USE_IR_AUG = os.environ.get("USE_IR_AUG", "0") == "1"
IR_AUG_PROB = float(os.environ.get("IR_AUG_PROB", 0.5))
IR_BANK_PATH = os.environ.get(
    "IR_BANK_PATH", os.path.join(os.path.dirname(__file__), "data", "ir_residual_bank.npz"))
_ir_bank = None


def ir_degrade_window(feat, num_keypoints=NUM_KEYPOINTS, feat_dim=None):
    """feat: (T, num_keypoints*5) -> the same window as a night-vision camera would have yielded.

    Each frame draws one real residual frame from the bank. A displacement of r_j torso lengths
    at joint j and r_hip at the hip centre moves the hip-relative, torso-normalised coordinate
    by (r_j - r_hip), which is what is added. Frames the bank marks as person-lost repeat the
    previous frame, which is what _step_person does with a missed detection. Velocity is then
    recomputed from the new positions, since a jump IS a velocity to the classifier.
    """
    global _ir_bank
    feat_dim = FEAT_DIM if feat_dim is None else feat_dim
    if feat_dim != 5:
        # Hip-motion and frame-position channels come from RAW coordinates that no longer
        # exist at this point, so they could not be degraded consistently.
        raise ValueError("USE_IR_AUG supports the 5-channel features only (x, y, conf, vx, vy)")
    if _ir_bank is None:
        with np.load(IR_BANK_PATH) as b:
            _ir_bank = (b["residuals"].astype(np.float32), b["lost"].astype(bool))
    residuals, lost = _ir_bank
    T = feat.shape[0]
    seq = feat.reshape(T, num_keypoints, feat_dim).copy()
    pick = np.random.randint(0, len(residuals), size=T)
    r = residuals[pick]                                          # (T, 17, 2)
    r_hip = (r[:, LEFT_HIP] + r[:, RIGHT_HIP]) / 2.0             # (T, 2)
    xy = seq[:, :, :2] + (r - r_hip[:, None, :])
    for t in range(1, T):
        if lost[pick[t]]:
            xy[t] = xy[t - 1]
    seq[:, :, :2] = xy
    seq[:, :, 3:5] = np.diff(xy, axis=0, prepend=xy[:1])
    return seq.reshape(T, -1)


class FallWindowDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat, label, _name = self.samples[idx]
        feat = feat.astype(np.float32)
        if self.augment:
            # First, so the flip and occlusion below act on the degraded skeleton the way they
            # would act on a real night frame.
            if USE_IR_AUG and np.random.rand() < IR_AUG_PROB:
                feat = ir_degrade_window(feat)
            if np.random.rand() < 0.5:
                feat = flip_horizontal_window(feat)
            feat = occlude_window(feat)
            if TRUNC_AUG:
                feat = trunc_window(feat)
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.float32)


def group_key(v):
    """The unit that must stay on one side of the split (Codex DATA-DESIGN #2, #4): everything
    that shares a person or a recording. OF-ItW segments of one source video share the video;
    FallVision's `subject` is its archive batch (no person id exists), so a batch is one group;
    OF-Syn's two 8 fps offsets of one clip share the clip."""
    src, name, subj = v.get("source", ""), v["name"], v["subject"]
    if "ofitw" in src or "omnifall_adl" in src:
        return "%s:%s" % (src, name.rsplit("_", 2)[0])
    if subj not in ("-1", "", "None"):
        return "%s:%s" % (src, subj)
    return "%s:%s" % (src, name)


def split_by_group(videos, val_pct=20):
    """Deterministic split by a hash of group_key: a group's side never depends on which other
    sources are loaded, so adding data cannot silently move existing videos between train and
    validation (the old random split reshuffled everything whenever a source was added)."""
    import hashlib
    # GMDCSA24_TRAIN_SUBJECTS / GMDCSA24_VAL_SUBJECTS (e.g. "2,3,4" / "1"): GMDCSA24 has only four
    # people, so the hash put three in validation and left one indoor-ADL subject to train on
    # (Codex ablation C, 2026-10-01). Listed subjects are forced; unlisted ones keep the hash.
    tr_s = {x.strip().lstrip("s") for x in os.environ.get("GMDCSA24_TRAIN_SUBJECTS", "").split(",") if x.strip()}
    va_s = {x.strip().lstrip("s") for x in os.environ.get("GMDCSA24_VAL_SUBJECTS", "").split(",") if x.strip()}
    if tr_s & va_s:
        raise ValueError("a GMDCSA24 subject cannot be both train and val: %s" % sorted(tr_s & va_s))
    train, val = [], []
    for v in videos:
        if v.get("source") in ("poses_yolopose", "poses_gmdcsa24_v2", "poses_gmdcsa24"):
            subj = str(v["subject"]).lstrip("s")
            if subj in tr_s:
                train.append(v)
                continue
            if subj in va_s:
                val.append(v)
                continue
        h = int(hashlib.md5(group_key(v).encode()).hexdigest(), 16) % 100
        (val if h < val_pct else train).append(v)
    return train, val


def split_videos(videos, val_ratio=0.2, seed=42):
    """Stratified split at the video level (never splits a single video's windows across sets)."""
    rng = np.random.RandomState(seed)
    by_label = {0: [], 1: []}
    for v in videos:
        by_label[v["label"]].append(v)

    train, val = [], []
    for label, vids in by_label.items():
        idx = rng.permutation(len(vids))
        n_val = max(1, int(len(vids) * val_ratio))
        val_idx = set(idx[:n_val].tolist())
        for i, v in enumerate(vids):
            (val if i in val_idx else train).append(v)
    return train, val
