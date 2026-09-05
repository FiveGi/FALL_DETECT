"""
Feasibility check: could a YOLO-pose model (native COCO-17 keypoint output --
exact same index convention v3_fall_detection.py already uses, LEFT_SHOULDER=5,
RIGHT_SHOULDER=6, LEFT_HIP=11, RIGHT_HIP=12 -- zero remapping needed unlike
MediaPipe's 33->17) replace MediaPipe as the pose extractor?

Focus on the case that matters most: person-detection RATE on frames where
someone is down/lying on the ground -- v3_fall_detection.py's own comments note
MediaPipe drops to ~40% detection rate on a real CAUCAFall forward-fall clip
(prone/occluded bodies aren't what MediaPipe was mostly trained on). Uses the
last 2s of each GMDCSA24 val Fall clip (person down after falling) plus two
Test/ clips (11 = IR bunk bed, 15 = hospital corridor fall) as cross-checks.

Tests yolo26n/s/m-pose (speed/quality tradeoff) against the deployed MediaPipe
pipeline, throttled to 4 threads to match the production server.
"""
import os
import time
import cv2
import numpy as np
import torch
import importlib.util

torch.set_num_threads(4)

from ultralytics import YOLO

_mod_path = os.path.join(os.path.dirname(__file__), "..", "app", "detection", "v3_fall_detection.py")
_spec = importlib.util.spec_from_file_location("v3_fall_detection", _mod_path)
_v3mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3mod)
V3PoseFallDetector = _v3mod.V3PoseFallDetector

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(ROOT, "models")
mp_detector = V3PoseFallDetector(MODEL_DIR)

YOLO_POSE_MODELS = ["yolo26n-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt"]

VAL_FALL = ["s1_Fall_02", "s1_Fall_06", "s1_Fall_16", "s2_Fall_02", "s2_Fall_04",
            "s2_Fall_09", "s2_Fall_14", "s2_Fall_20", "s3_Fall_02", "s3_Fall_09",
            "s3_Fall_12", "s3_Fall_13", "s3_Fall_16", "s4_Fall_06", "s4_Fall_17"]
FALL_DIR = os.path.join(ROOT, "training", "data", "gmdcsa24_fall_raw")

TEST_DIR = os.environ.get("TEST_DIR", r"D:\project\PROJECT\Test")
CROSS_CHECK_CLIPS = ["11.mp4", "15.mp4"]


def gather_frames():
    frames = []  # (source_label, frame_bgr)
    for clip in VAL_FALL:
        path = os.path.join(FALL_DIR, f"{clip}.mp4")
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = n / fps
        t = max(0.0, dur - 2.0)
        while t < dur:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if ok:
                frames.append((f"{clip}@{t:.1f}s(down)", frame))
            t += 0.3
        cap.release()
    for clip in CROSS_CHECK_CLIPS:
        path = os.path.join(TEST_DIR, clip)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = n / fps
        t = 0.0
        while t < dur:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if ok:
                frames.append((f"{clip}@{t:.1f}s", frame))
            t += 1.0
        cap.release()
    return frames


frames = gather_frames()
print(f"Gathered {len(frames)} frames ({sum(1 for l,_ in frames if '(down)' in l)} 'person-down' GMDCSA24 frames "
      f"+ {sum(1 for l,_ in frames if '(down)' not in l)} Test/ cross-check frames)\n")

# --- MediaPipe baseline ---
mp_found = 0
mp_times = []
for i, (label, frame) in enumerate(frames):
    t0 = time.perf_counter()
    kpts, found = mp_detector.extract_keypoints(frame)
    dt = time.perf_counter() - t0
    if i >= 5:
        mp_times.append(dt)
    if found:
        mp_found += 1
print(f"MediaPipe: {mp_found}/{len(frames)} frames with a person detected "
      f"({mp_found/len(frames):.1%}), avg {sum(mp_times)/len(mp_times):.4f}s/frame\n")

# --- YOLO-pose variants ---
for model_name in YOLO_POSE_MODELS:
    model = YOLO(model_name)
    found = 0
    times = []
    for i, (label, frame) in enumerate(frames):
        t0 = time.perf_counter()
        r = model.predict(frame, verbose=False, conf=0.5)[0]
        dt = time.perf_counter() - t0
        if i >= 5:
            times.append(dt)
        if r.boxes is not None and len(r.boxes) > 0:
            found += 1
    avg_t = sum(times) / len(times) if times else 0.0
    print(f"{model_name}: {found}/{len(frames)} frames with a person detected "
          f"({found/len(frames):.1%}), avg {avg_t:.4f}s/frame")
