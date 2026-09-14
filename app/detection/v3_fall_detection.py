"""Pose-only fall classifier (v3): windowed COCO-17 keypoint sequences -> temporal CNN.

Replaces the v2 DeepSVDD pipeline (RGB ResNet50 + optical flow + a pose feature
branch that always returned zeros -- see FeatureExtractorONNX.extract_pose_features
in v2_fall_detection_onnx.py) with a model trained purely on pose keypoints, which
is both simpler and actually uses the skeleton signal it's supposed to. See
training/model.py and training/train.py for how it was trained (~10,000 windowed
clips pooled from GMDCSA24 + FallVision + CAUCAFall + OmniFall OF-ItW/OOPS).

Pose backend is YOLO-pose (yolo26s-pose), not MediaPipe -- switched SS34/35 after
validating end to end (better raw detection rate on hard "person down" frames,
faster on CPU, and -- with flip + synthetic-occlusion training augmentation added
specifically to close a gap found on a real cane-assisted fall clip -- matching or
exceeding the prior MediaPipe-trained model's numbers on both GMDCSA24 held-out
test sets). Its output is already COCO-17 (index-identical to LEFT_SHOULDER=5/
RIGHT_SHOULDER=6/LEFT_HIP=11/RIGHT_HIP=12 below), unlike MediaPipe's 33-point
BlazePose output which needed the MEDIAPIPE33_TO_COCO17 remap (kept below, no
longer used by this file, but training/extract_poses.py's original MediaPipe-based
extraction scripts still reference it).

Preprocessing here must match training/dataset.py exactly: pose -> COCO-17 subset ->
torso-relative normalization -> per-frame velocity -> 30-frame window.

Operating point (see training/tune_threshold.py): sigmoid threshold 0.5, plus
requiring SMOOTH_NEED-of-last-SMOOTH_OF windows to agree before raising an alert
(not strictly consecutive -- a 50-clip end-to-end batch test found several real
falls where confidence spiked above threshold but dipped for a single window in
between, which a strict "N in a row" rule threw away). This still cuts false
alarms for a modest recall cost. This is not accurate enough to alert
autonomously; treat detections as a prompt for staff to check the camera, not a
confirmed event.
"""
import os
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

NUM_KEYPOINTS = 17
# Overridable so the low-frame-rate experiment (a model trained on frames subsampled to the
# rate a live camera actually achieves -- see training/eval_v3_frame_drop.py) can be evaluated
# through this exact production code path instead of a parallel copy of it.
WINDOW_SIZE = int(os.environ.get("V3_WINDOW_SIZE", 30))
STRIDE = 10
THRESHOLD = 0.5
# Env-overridable alongside V3_WINDOW_SIZE: at a live camera'''s real frame rate each window
# advances by a whole 1/5 s, so "2 positive windows out of 3" is a much longer wait than it
# was at 30fps -- worth measuring rather than assuming (training/eval_v3_frame_drop.py).
SMOOTH_NEED = int(os.environ.get("V3_SMOOTH_NEED", 2))   # need this many...
SMOOTH_OF = 3      # ...positive windows out of the last this many (not strictly consecutive)
NUM_POSES = 4      # max people tracked per camera at once -- see detect_v3_fall_multi

MEDIAPIPE33_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12


SMOOTH_KERNEL = 3
# MediaPipe's PoseLandmarker runs in IMAGE mode (each frame estimated independently,
# no temporal tracking -- see extract_poses.py), which is fine for a static photo but
# means keypoints can jitter frame-to-frame even when the person hasn't moved: on a
# real test clip, a standing-still person's shoulder-hip tilt angle was measured
# swinging 5deg -> 87deg -> 12deg -> 83deg across consecutive frames (SKILL.md SS18).
# Since velocity (frame-to-frame keypoint displacement) is a direct input feature,
# that jitter reads as fast motion and produced 3/3 confirmed false "fall" alerts on
# real test footage, independently confirmed by Gemini reviewing the same frames.
# A short trailing/centered moving average on x,y (not visibility) damps single-frame
# jitter while a genuine fall -- large, multi-frame, sustained displacement -- still
# comes through. Applied only here (inference), not in training/dataset.py's
# preprocessing, which is a real train/inference mismatch; validated empirically
# instead (SS18) rather than assumed safe -- see there before changing this further.
def _smooth_keypoints(raw_window, kernel=SMOOTH_KERNEL):
    """raw_window: (T, 17, 3) -> same shape, x,y smoothed with a centered moving
    average (visibility left untouched)."""
    T = raw_window.shape[0]
    half = kernel // 2
    smoothed = raw_window.copy()
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        smoothed[t, :, :2] = raw_window[lo:hi, :, :2].mean(axis=0)
    return smoothed


def _normalize_and_velocity(raw_window):
    """raw_window: (WINDOW_SIZE, 17, 3) raw [x, y, visibility] -> (WINDOW_SIZE, 17, 5)
    [x, y, confidence, vx, vy], torso-relative and scale-normalized per frame."""
    raw_window = _smooth_keypoints(raw_window)
    xy = raw_window[:, :, :2]
    vis = raw_window[:, :, 2:3]

    hip_center = (xy[:, LEFT_HIP] + xy[:, RIGHT_HIP]) / 2.0
    shoulder_center = (xy[:, LEFT_SHOULDER] + xy[:, RIGHT_SHOULDER]) / 2.0
    torso_size = np.clip(np.linalg.norm(shoulder_center - hip_center, axis=1), 1e-3, None)

    xy_norm = (xy - hip_center[:, None, :]) / torso_size[:, None, None]
    norm_seq = np.concatenate([xy_norm, vis], axis=-1)

    vel = np.diff(xy_norm, axis=0, prepend=xy_norm[:1])
    return np.concatenate([norm_seq, vel], axis=-1)


# Input resolution handed to YOLO-pose. Ultralytics' default is 640, which downsamples a
# 1080p camera frame ~3x and is measured to be where recall is lost: a fall composited into a
# 2x-wide frame (same person, half the pixels) drops from 15/15 to 10/15 with nobody else in
# shot (training/eval_multiperson_composite.py), matching SS6's finding that errors cluster
# where the person is far from the camera. Raising it costs GPU time, which the GPU now has
# (53 fps standalone versus the ~25 fps a camera needs).
# 960, not ultralytics' default 640. Measured across every ground-truth surface this project
# uses, 960 is better or equal on all of them and worse on none -- val 15/15 falls with
# ADL-clean 9/16 -> 10/16, train50 unchanged at 22/25 and 22/25, and on the two-person
# composites the fall is caught 13/15 instead of 12/15 (14/15 vs 10/15 in the blank-half
# control that isolates resolution from the second person). 1280 trades differently: far
# fewer false alarms (val ADL-clean 14/16) but it starts losing falls (14/15 val, 20/25
# train50), so it is not taken. Costs 18.8 -> 21.8 ms/frame on the GPU, which still leaves
# roughly twice the throughput a 25 fps camera needs.
IMGSZ = int(os.environ.get("V3_IMGSZ", 960))

# Which tracker assigns a person their identity across frames. "hip" is the original
# nearest-hip-centre matcher in PersonTracker below; "bytetrack" uses ultralytics' own
# tracker (Kalman motion prediction + IoU), which exists because the hip matcher loses
# people constantly: measured on Test/1,12,16,17, between 68% and 91% of the track ids it
# creates are re-detections of someone it already had, not new people. Each of those resets
# that person's 30-frame window, so the classifier keeps scoring half-filled windows.
TRACKER = os.environ.get("V3_TRACKER", "hip")

# Confidence YOLO-pose must have in a person before their pose is used. Ultralytics' own
# default is 0.25; 0.5 was chosen here before any of it was measured. It decides detection
# CONTINUITY, which is what actually breaks multi-person tracking: on Test/1 the detector
# averages 0.83 people per frame, so tracks keep expiring and restarting their 30-frame
# window regardless of which tracker is used.
# 0.3, measured. Raising detection continuity is what actually helps multi-person accuracy:
# at 0.5 the held-out train50 set catches 22/25 falls, at 0.3 it catches 24/25, with ADL-clean
# unchanged on both sets (val 10/16, train50 22/25) and the two-person composites identical
# (13/15, 4/4 clean). 0.25 measures the same as 0.3 but keeps more marginal detections, so 0.3
# is taken as the smaller change from the previous 0.5.
POSE_CONF = float(os.environ.get("V3_POSE_CONF", 0.3))


def _autodetect_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class V3PoseFallDetector:
    """Loads the pose extractor + ONNX classifier. One instance shared across all cameras.

    Pose backend is YOLO-pose (yolo26s-pose, native COCO-17 keypoint output -- index-
    identical to LEFT_SHOULDER/RIGHT_SHOULDER/LEFT_HIP/RIGHT_HIP below, no 33->17
    remapping needed), not MediaPipe -- see SKILL.md SS34/35. Replaced MediaPipe after
    validating end to end: better raw detection rate on hard "person down" frames
    (81% vs 72%), ~40% faster on CPU, and -- after adding flip + synthetic-occlusion
    training augmentation (SS35) specifically to close a gap found on a real clip
    involving a mobility cane and torso rotation -- matching or exceeding the prior
    MediaPipe-trained model's numbers on both GMDCSA24 held-out test sets."""

    def __init__(self, model_dir, device=None):
        """device: "cuda", "cpu", or None to auto-detect.

        Pose extraction is the whole cost of this pipeline and it runs on every frame with
        no stride, so the device choice decides whether the loop keeps up with a camera at
        all: measured ~2 fps on 2 CPU cores versus ~26 fps on this machine's GPU
        (training/bench_deployed_v3.py). At ~2 fps a live camera's frames are dropped faster
        than the 30-frame window can span a fall and recall collapses -- see
        training/eval_v3_frame_drop.py. Auto-detect rather than defaulting to CPU so a host
        that has a GPU actually uses it; V3_DEVICE overrides when that is not wanted."""
        onnx_path = os.path.join(model_dir, "fall_classifier_v3.onnx")
        yolopose_path = os.path.join(model_dir, "yolo26s-pose.pt")

        if device is None:
            device = os.environ.get("V3_DEVICE") or _autodetect_device()
        self.device = device

        providers = ["CPUExecutionProvider"]
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            # The classifier is tiny, so this is a nice-to-have; the CPU-only onnxruntime
            # build has no CUDA provider and asking for it anyway is a hard error.
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.pose_model = YOLO(yolopose_path)
        print(f"[V3] pose backend on {device}")

    def extract_keypoints(self, frame_bgr):
        """-> ((17, 3) COCO17 [x, y, confidence], person_found: bool) for the FIRST
        detected person only. Zeros + False if nobody detected. Single-person callers
        (training/eval scripts, all validated against this exact signature -- see
        SKILL.md) keep using this; camera_manager.py uses extract_all_keypoints /
        detect_v3_fall_multi below for multi-person tracking instead."""
        people = self.extract_all_keypoints(frame_bgr)
        if not people:
            return np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32), False
        return people[0][0], True

    def extract_all_keypoints(self, frame_bgr):
        """-> list of (kpts17 (17,3) [x, y, confidence], hip_center (2,)) for every
        person detected this frame, up to NUM_POSES, sorted by detection confidence
        (highest first). Empty list if nobody detected."""
        h, w = frame_bgr.shape[:2]
        result = self.pose_model.predict(frame_bgr, verbose=False, conf=POSE_CONF, classes=[0],
                                         device=self.device, imgsz=IMGSZ)[0]
        people = []
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return people
        box_confs = result.boxes.conf.cpu().numpy()
        order = np.argsort(-box_confs)[:NUM_POSES]
        for i in order:
            kxy = result.keypoints.xy[i].cpu().numpy()
            kconf = (result.keypoints.conf[i].cpu().numpy()
                     if result.keypoints.conf is not None else np.ones(NUM_KEYPOINTS, dtype=np.float32))
            kpts17 = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
            kpts17[:, 0] = kxy[:, 0] / w
            kpts17[:, 1] = kxy[:, 1] / h
            kpts17[:, 2] = kconf
            hip_center = (kpts17[LEFT_HIP, :2] + kpts17[RIGHT_HIP, :2]) / 2.0
            people.append((kpts17, hip_center))
        return people

    def extract_tracked_keypoints(self, frame_bgr):
        """-> list of (track_id, kpts17, hip_center), ids assigned by ultralytics' tracker.

        persist=True keeps the tracker's state between calls, which means this detector
        instance is tied to ONE video stream -- model_manager hands out a detector per camera
        for exactly that reason. Detections without an id (ByteTrack emits those on the first
        frames of a new track) are skipped rather than given a synthetic id, so a person only
        enters a window once their identity is stable.
        """
        h, w = frame_bgr.shape[:2]
        result = self.pose_model.track(
            frame_bgr, persist=True, tracker="bytetrack.yaml", verbose=False,
            conf=POSE_CONF, classes=[0], device=self.device, imgsz=IMGSZ)[0]
        people = []
        if result.keypoints is None or result.boxes is None or result.boxes.id is None:
            return people
        ids = result.boxes.id.int().cpu().numpy()
        for i, track_id in enumerate(ids[:NUM_POSES]):
            kxy = result.keypoints.xy[i].cpu().numpy()
            kconf = (result.keypoints.conf[i].cpu().numpy()
                     if result.keypoints.conf is not None else np.ones(NUM_KEYPOINTS, dtype=np.float32))
            kpts17 = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
            kpts17[:, 0] = kxy[:, 0] / w
            kpts17[:, 1] = kxy[:, 1] / h
            kpts17[:, 2] = kconf
            hip = (kpts17[LEFT_HIP, :2] + kpts17[RIGHT_HIP, :2]) / 2.0
            people.append((int(track_id), kpts17, hip))
        return people

    def predict_window(self, raw_window):
        """raw_window: (WINDOW_SIZE, 17, 3) -> fall probability in [0, 1]."""
        feat = _normalize_and_velocity(raw_window).reshape(1, WINDOW_SIZE, -1).astype(np.float32)
        logit = self.session.run(["logit"], {"input": feat})[0]
        return float(1.0 / (1.0 + np.exp(-logit.reshape(-1)[0])))


# Env-overridable so the multi-person false-positive sweep can A/B it (see
# training/diagnose_multi_person.py: most multi-person alerts fire from windows that are
# mostly held copies rather than genuinely observed frames).
MIN_PERSON_FRACTION = float(os.environ.get("V3_MIN_PERSON_FRACTION", 0.2))
# Fraction of frames in a window that must have a detected person before trusting the
# classifier's output. Deliberately low: MediaPipe's per-frame pose detection is much
# less reliable once someone is on the ground (prone/occluded bodies aren't what it was
# mostly trained on) -- a real CAUCAFall forward-fall clip only had a person detected in
# 40% of frames, concentrated in bursts, not a steady rate. 0.7 was tuned only against an
# empty-room case (which sits at ~0% detected) and ended up suppressing genuine falls too.
RESET_PERSON_FRACTION = 0.05
# Below this, treat it as a genuinely empty scene and fully reset state. Between this and
# MIN_PERSON_FRACTION, hold the last known state instead of resetting -- see below.
COLLAPSE_CONFIDENCE = 0.6
# Whether the collapse rule fires at all. It was added for MediaPipe, which lost people
# entirely once they were prone; YOLO-pose detects prone frames far more reliably (81.2% vs
# 71.7%, SS34), so the rule may now cost more than it earns -- it reports probability 1.0 on
# a person VANISHING, which is also what a night scene or a spurious detection looks like.
# Env-overridable so that is measured rather than argued.
COLLAPSE_ENABLED = os.environ.get("V3_COLLAPSE_ENABLED", "1") != "0"
# If the model was this confident right before person-detection collapsed to ~zero,
# treat the collapse itself as a fall signal (see detect_v3_fall).
# 0.2 still rejects the empty-room case while letting spotty-but-real detections through.
# Env-overridable so its effect can be measured by A/B rather than assumed (set very high
# to reproduce the pre-guard behaviour of holding the last pose indefinitely).
MAX_HELD_RUN = int(os.environ.get("V3_MAX_HELD_RUN", 5))
# How many CONSECUTIVE undetected frames may be back-filled with the last real keypoints
# before this stops writing to the window entirely. Holding is what SS18 validated as the
# fix for short (1-2 frame) dropouts, but SS24 traced two high-confidence false alerts
# (0.92-0.96 with person_found=False) to the opposite extreme: person_found flickering for
# ~1-2s fills the 30-frame window with mostly repeated copies of one earlier pose, and the
# classifier scores that frozen-then-jump pattern as fall-like. A plain person_fraction
# gate can't separate those cases -- a genuine CAUCAFall forward fall sits at 40% detected,
# inside the same band -- so this caps the RUN LENGTH instead of the total fraction. Past
# the cap the window is frozen (nothing appended) rather than padded further, so the
# classifier keeps re-scoring the last genuinely observed frames instead of a synthetic
# pattern, which also preserves a real fall's pre-dropout signal (the case
# MIN_PERSON_FRACTION exists to protect). person_flags still records every miss, so the
# fraction gates above keep responding normally.
#
# MEASURED EFFECT ON CURRENT DATA: none, on any clip available here. GMDCSA24 held-out is
# bit-identical with and without it (15/15 falls, 9/16 ADL clean, eval_v3_on_gmdcsa24_val.py),
# and on the Test/ clips with the worst dropouts -- 10/11/6.mp4, where check_held_run.py
# measures runs of 122-359 consecutive misses -- both settings fire zero alerts, because the
# person_fraction gates already suppress everything there (ab_held_run_on_test.py). SS24's
# mechanism was observed under MediaPipe; YOLO-pose detects the hard prone frames far more
# reliably (81.2% vs 71.7%, SS34), so the flicker pattern that produced it no longer appears
# in this data. Kept as a cheap guard against that documented failure mode returning, NOT
# because it was shown to fix anything -- do not cite it as an accuracy improvement.


class V3FallDetectionState:
    """Per-camera state: rolling keypoint buffer + smoothing history."""

    def __init__(self):
        self.raw_buffer = deque(maxlen=WINDOW_SIZE)
        self.person_flags = deque(maxlen=WINDOW_SIZE)
        self.frames_since_infer = 0
        self.recent_flags = deque(maxlen=SMOOTH_OF)
        self.has_run_once = False
        self.last_probability = 0.0
        self.last_detected = False
        self.collapse_fired = False
        self.last_good_kpts = None
        self.held_run = 0

    def is_ready(self):
        return len(self.raw_buffer) == WINDOW_SIZE


def _step_person(kpts, person_found, state: V3FallDetectionState,
                  fall_detector: V3PoseFallDetector, threshold):
    """One person's rolling-window update + classification for this frame -- the
    exact state machine detect_v3_fall validated (SS9-SS18), factored out so
    detect_v3_fall (single person) and detect_v3_fall_multi (N people, one of these
    states per tracked person) share identical logic instead of two copies drifting
    apart. Returns (detected, probability, label)."""
    if person_found:
        state.last_good_kpts = kpts
        state.held_run = 0
        state.raw_buffer.append(kpts)
    else:
        # A momentary tracking dropout (1-2 frames, common mid-fall/near occlusion --
        # see MIN_PERSON_FRACTION above) makes extract_keypoints return an all-zero
        # vector. Feeding that raw into the window creates a real->zero->real jump
        # that reads as a huge velocity spike -- confirmed as part of the false-alarm
        # mechanism in SKILL.md SS18 (alongside plain frame-to-frame jitter, which
        # _smooth_keypoints handles separately). Hold the last real detection instead
        # of zero-filling; person_flags still records the true miss for MIN_PERSON_FRACTION.
        state.held_run += 1
        if state.last_good_kpts is None:
            state.raw_buffer.append(kpts)  # nothing real seen yet -- zeros are all there is
        elif state.held_run <= MAX_HELD_RUN:
            state.raw_buffer.append(state.last_good_kpts)
        # Past MAX_HELD_RUN: append nothing, freezing the window on the last genuinely
        # observed frames instead of packing it with more copies of one pose (SS24).
    state.person_flags.append(person_found)
    state.frames_since_infer += 1

    if not state.is_ready():
        return False, 0.0, "Analyzing..."

    person_fraction = sum(state.person_flags) / len(state.person_flags)
    if person_fraction < RESET_PERSON_FRACTION:
        # Essentially nobody detected across the whole window -- normally a genuinely
        # empty room/person off-camera, safe to fully reset. BUT: if confidence was high
        # right before detection collapsed to ~zero, that transition itself (visible and
        # apparently falling -> suddenly untrackable) is consistent with a real collapse,
        # not someone calmly walking off -- confirmed via batch testing, where two real
        # falls peaked at 0.73/0.69 then MediaPipe lost the person entirely for the rest
        # of the clip. Fire one alert on that transition instead of silently discarding it.
        was_collapse = (COLLAPSE_ENABLED and state.last_probability > COLLAPSE_CONFIDENCE
                        and not state.collapse_fired)
        state.recent_flags.clear()
        state.last_probability = 0.0
        state.last_detected = False
        if was_collapse:
            state.collapse_fired = True
            return True, 1.0, "fall"
        return False, 0.0, "no_person"

    if person_fraction < MIN_PERSON_FRACTION:
        # Some detections, but too few to trust a fresh prediction from this window --
        # don't run the classifier on effectively-degraded input. Importantly, do NOT
        # clear prior state here: a person who just fell is lying down, which MediaPipe
        # often fails to track for a stretch right after the fall -- clearing on every
        # low-detection window was wiping out the fall signal at exactly the moment it
        # mattered (confirmed via batch testing: probability climbed to 0.73 right
        # before the person went down, then got erased by this gate). Hold the last
        # known state instead of erasing it.
        label = "fall" if state.last_detected else "no_person"
        return state.last_detected, state.last_probability, label

    # Only re-run the classifier every STRIDE frames -- matches the window stride the
    # model was validated on, and keeps this affordable at real camera frame rates.
    if state.has_run_once and state.frames_since_infer < STRIDE:
        label = "fall" if state.last_detected else "no_fall"
        return state.last_detected, state.last_probability, label

    state.frames_since_infer = 0
    state.has_run_once = True
    state.collapse_fired = False  # person is reliably visible again -- a future collapse is a new event
    raw_window = np.stack(state.raw_buffer, axis=0)
    probability = fall_detector.predict_window(raw_window)
    state.recent_flags.append(probability > threshold)
    state.last_probability = probability
    state.last_detected = sum(state.recent_flags) >= SMOOTH_NEED

    label = "fall" if state.last_detected else "no_fall"
    return state.last_detected, probability, label


def detect_v3_fall(frame, state: V3FallDetectionState, fall_detector: V3PoseFallDetector,
                    config, camera=None, threshold=None):
    """Single-person entry point -- same signature/return shape as the old
    detect_v2_fall_only_onnx, so it's a drop-in replacement: returns
    (detected, probability, label, frame). Used by all the training/eval scripts;
    camera_manager.py uses detect_v3_fall_multi instead."""
    threshold = threshold if threshold is not None else THRESHOLD
    kpts, person_found = fall_detector.extract_keypoints(frame)
    detected, probability, label = _step_person(kpts, person_found, state, fall_detector, threshold)
    return detected, probability, label, frame


MAX_TRACK_DISTANCE = 0.15
# Max normalized hip-center movement (as a fraction of frame width/height) between
# consecutive frames for a detection to count as "the same person" -- chosen as a
# generous-but-not-unlimited gate: a person walking normally moves much less than
# this between frames at real camera fps, but it's loose enough to survive MediaPipe's
# own per-frame jitter (SS18) without needing a real motion model.
MAX_MISSED_FRAMES = WINDOW_SIZE
# How many consecutive frames a track can go undetected (occluded, briefly off-camera)
# before being dropped -- one full window's worth, so a track surviving a gap this
# long still has stale-but-recent history rather than restarting cold.


class PersonTracker:
    """Nearest-hip-center tracker so each person's rolling window doesn't get
    contaminated by a different person's keypoints frame to frame. Not a real
    multi-object tracker -- no motion model, no re-identification after a track is
    dropped. Fine for the same-room, few-people, mostly-static-camera case this is
    built for; people crossing paths closely enough to swap positions within one
    MAX_TRACK_DISTANCE step could swap track IDs. That's a state-continuity glitch,
    not a missed detection -- both people are still tracked and classified."""

    def __init__(self):
        self.next_id = 0
        self.tracks = {}  # track_id -> {"centroid": (x, y), "missed": int}

    def update(self, detections):
        """detections: list of (kpts, hip_center) from extract_all_keypoints.
        Returns list of (track_id, kpts_or_None, seen: bool) for every currently
        active track, including ones not matched this frame (kpts=None, seen=False)."""
        unmatched = list(range(len(detections)))
        matched = {}

        for track_id, t in sorted(self.tracks.items()):
            if not unmatched:
                break
            dists = sorted(
                ((float(np.linalg.norm(t["centroid"] - detections[i][1])), i) for i in unmatched),
                key=lambda x: x[0],
            )
            best_dist, best_i = dists[0]
            if best_dist < MAX_TRACK_DISTANCE:
                matched[track_id] = best_i
                unmatched.remove(best_i)

        results = []
        for track_id, t in self.tracks.items():
            if track_id in matched:
                kpts, centroid = detections[matched[track_id]]
                t["centroid"] = centroid
                t["missed"] = 0
                results.append((track_id, kpts, True))
            else:
                t["missed"] += 1
                results.append((track_id, None, False))

        for i in unmatched:
            kpts, centroid = detections[i]
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {"centroid": centroid, "missed": 0}
            results.append((track_id, kpts, True))

        self.tracks = {tid: t for tid, t in self.tracks.items() if t["missed"] <= MAX_MISSED_FRAMES}
        return [r for r in results if r[0] in self.tracks]


class V3MultiPersonFallState:
    """Per-camera state for multi-person detection: a PersonTracker plus one
    V3FallDetectionState per tracked person, so each person's rolling window/alert
    smoothing is independent of every other person in frame."""

    def __init__(self):
        self.tracker = PersonTracker()
        self.person_states = {}  # track_id -> V3FallDetectionState
        # Used only by the bytetrack path: ByteTrack keeps its own identities, so this
        # side only has to remember where each id was last seen and for how long it has
        # been missing.
        self.missed = {}
        self.last_centroid = {}
        # People actually detected in the most recent frame, as opposed to tracks being held
        # through a dropout -- the number to report to a human.
        self.seen_count = 0


def detect_v3_fall_multi(frame, multi_state: V3MultiPersonFallState,
                          fall_detector: V3PoseFallDetector, config, camera=None, threshold=None):
    """Multi-person entry point. Returns a list of
    (track_id, detected, probability, label, hip_center) -- one entry per person
    currently tracked in this camera's frame (including ones not seen this exact
    frame but still within MAX_MISSED_FRAMES, matching single-person's tolerance for
    momentary tracking dropouts)."""
    threshold = threshold if threshold is not None else THRESHOLD

    if TRACKER == "bytetrack":
        # ByteTrack owns the identities, so there is no separate matching step to fail. It
        # reports only people it can see this frame; tracks it is holding through an
        # occlusion are stepped as "not seen" below, the same way the hip tracker's misses
        # are, so _step_person's hold/freeze behaviour is unchanged.
        seen_people = fall_detector.extract_tracked_keypoints(frame)
        seen_now = {tid: (kpts, hip) for tid, kpts, hip in seen_people}
        for tid, (_, hip) in seen_now.items():
            multi_state.last_centroid[tid] = hip
            multi_state.missed[tid] = 0
        for tid in list(multi_state.person_states):
            if tid not in seen_now:
                multi_state.missed[tid] = multi_state.missed.get(tid, 0) + 1
        tracked = [(tid, seen_now[tid][0], True) for tid in seen_now]
        tracked += [(tid, None, False) for tid in multi_state.person_states
                    if tid not in seen_now
                    and multi_state.missed.get(tid, 0) <= MAX_MISSED_FRAMES]
    else:
        detections = fall_detector.extract_all_keypoints(frame)
        tracked = multi_state.tracker.update(detections)

    multi_state.seen_count = sum(1 for _, _, seen in tracked if seen)

    results = []
    for track_id, kpts, seen in tracked:
        state = multi_state.person_states.setdefault(track_id, V3FallDetectionState())
        step_kpts = kpts if seen else np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        detected, probability, label = _step_person(step_kpts, seen, state, fall_detector, threshold)
        if TRACKER == "bytetrack":
            centroid = multi_state.last_centroid.get(track_id, np.zeros(2, dtype=np.float32))
        else:
            centroid = multi_state.tracker.tracks[track_id]["centroid"]
        results.append((track_id, detected, probability, label, centroid))

    if TRACKER == "bytetrack":
        for track_id in list(multi_state.person_states):
            if multi_state.missed.get(track_id, 0) > MAX_MISSED_FRAMES:
                multi_state.person_states.pop(track_id, None)
                multi_state.missed.pop(track_id, None)
                multi_state.last_centroid.pop(track_id, None)
        return results

    # Drop state for any track the tracker has expired, so memory doesn't grow
    # unbounded over a long-running camera session.
    for track_id in list(multi_state.person_states):
        if track_id not in multi_state.tracker.tracks:
            del multi_state.person_states[track_id]

    return results
