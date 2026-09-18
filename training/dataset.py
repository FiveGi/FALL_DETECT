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

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 30))
STRIDE = 10

# TEMPORAL_STRIDE: keep every k-th frame, to match the frame rate the live pipeline can
# actually sustain (see this module's git history / SKILL.md -- offline eval sees 30fps,
# a real camera loop sees ~1-5fps). 1 = original behaviour, every frame.
TEMPORAL_STRIDE = int(os.environ.get("TEMPORAL_STRIDE", 1))

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
FEAT_DIM = 7 if USE_HIP_MOTION else 5

# Index to swap with for a left-right mirror flip (self-pairs for Nose, which has no
# left/right counterpart). Used by flip_horizontal_window() below.
FLIP_PAIRS = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


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
    for pose_dir in pose_dirs:
        for path in sorted(glob.glob(os.path.join(pose_dir, "*.npz"))):
            data = np.load(path, allow_pickle=True)
            raw = to_coco17(data["keypoints"].astype(np.float32))
            if TEMPORAL_STRIDE > 1:
                # Before velocity: vx/vy must be the delta between two frames the runtime
                # actually sees in sequence, not between two 30fps neighbours.
                raw = raw[::TEMPORAL_STRIDE]
            motion = compute_motion_energy(raw)
            seq = add_velocity(normalize_sequence(raw), raw_seq=raw)
            video = {
                "keypoints": seq,
                "motion": motion,
                "label": int(data["label"]),
                "subject": str(data["subject"]),
                "name": os.path.basename(path),
            }
            if "frame_labels" in data:
                # Real per-frame ground truth (CAUCAFall) -- used instead of the
                # peak-motion heuristic when available, since it's not a guess.
                video["frame_labels"] = data["frame_labels"].astype(np.int64)
            videos.append(video)
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
    if feat_dim >= 7:
        seq[:, :, 5] = -seq[:, :, 5]   # hip dx -- the body now travels the other way too
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
            if np.random.rand() < 0.5:
                feat = flip_horizontal_window(feat)
            feat = occlude_window(feat)
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.float32)


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
