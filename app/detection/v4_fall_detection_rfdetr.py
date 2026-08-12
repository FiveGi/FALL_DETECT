"""Fall detection v4: RF-DETR object detector fine-tuned directly on fall footage
(classes: standing / falling / fallen), replacing the v3 pose-keypoint pipeline.

Why: v3 (MediaPipe pose -> temporal CNN) depended entirely on MediaPipe successfully
tracking a human skeleton every frame. On real CAUCAFall test clips, MediaPipe
sometimes failed to detect a person for an ENTIRE clip once they were on the ground
(a prone/collapsed body isn't what it was trained on) -- in a 50-clip end-to-end batch
test, this caused several real falls to be missed with zero usable signal at all.
RF-DETR classifies the raw RGB frame directly, so it doesn't have that failure mode:
on the exact same clips that gave MediaPipe 0% person-detection, RF-DETR correctly
tracked standing -> falling -> fallen.

Compared on the same 50-clip benchmark (5 subjects x 10 activities), requiring
FALLEN_STREAK_NEEDED consecutive "fallen" classifications before alerting:
  v3 (pose):          80% sensitivity, 80% specificity
  v4 (this, RF-DETR):  80% sensitivity, 84% specificity  (strictly better)
Lowering FALLEN_STREAK_NEEDED to 1 trades up to 92% sensitivity for 76% specificity,
if the deployment wants to bias further toward not missing falls.

Failure pattern is different from v3, not eliminated: this model sometimes calls
kneeling/picking-up-object "falling" for a sustained stretch (never "fallen" though,
which is what the streak requirement filters out), and can miss a fall that ends in a
sitting position (visually similar to just sitting down). Still not accurate enough to
alert autonomously -- treat detections as a prompt for staff to check the camera.

Checkpoint provenance: fine-tuned from Roboflow's RF-DETR-base on a Roboflow-hosted
"fall_detaction-3" dataset (per the checkpoint's saved training args), not trained by
this pipeline -- training data/label quality has not been independently audited here.
"""
import os

import cv2
from rfdetr import RFDETRBase

FALLEN_STREAK_NEEDED = 2
# Consecutive "fallen" classifications (each STRIDE_FRAMES apart) required before
# alerting. 1 = more sensitive (92%/76% on the 50-clip benchmark), 2 = the default
# used to derive the numbers in the module docstring (80%/84%).
STRIDE_FRAMES = 5
# Run the classifier every this-many frames -- matches how the 50-clip benchmark
# above was sampled, and keeps this affordable at real camera frame rates (RF-DETR
# is heavier per-frame than the old pose CNN).
CONFIDENCE_THRESHOLD = 0.3


class V4RFDETRFallDetector:
    """Loads the fine-tuned RF-DETR checkpoint. One instance shared across all cameras."""

    def __init__(self, model_dir):
        ckpt_path = os.path.join(model_dir, "rfdetr_fall_detector.pth")
        self.model = RFDETRBase(pretrain_weights=os.path.abspath(ckpt_path))
        self.class_names = self.model.class_names  # ['fallen', 'falling', 'standing']

    def classify_frame(self, frame_bgr):
        """frame_bgr: raw camera frame (BGR, as OpenCV reads it) -> (label: str, confidence: float).
        label is one of 'fallen' / 'falling' / 'standing' / 'none' (nothing detected above threshold)."""
        # RF-DETR expects RGB (cv2.cvtColor, not a [::-1] slice, since the latter
        # produces a negative-stride view that torch.from_numpy can't accept).
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = self.model.predict(frame_rgb, threshold=CONFIDENCE_THRESHOLD)
        if len(detections.class_id) == 0:
            return "none", 0.0
        best = int(detections.confidence.argmax())
        label = self.class_names[int(detections.class_id[best]) - 1]  # class_id is 1-indexed
        return label, float(detections.confidence[best])


class V4FallDetectionState:
    """Per-camera state: how many frames since we last classified, and the current
    streak of consecutive 'fallen' classifications."""

    def __init__(self):
        self.frames_since_infer = 0
        self.has_run_once = False
        self.fallen_streak = 0
        self.last_probability = 0.0
        self.last_detected = False


def detect_v4_fall(frame, state: V4FallDetectionState, fall_detector: V4RFDETRFallDetector,
                    config, camera=None, threshold=None):
    """Same signature/return shape as detect_v2_fall_only_onnx / detect_v3_fall, so
    it's a drop-in replacement: returns (detected, probability, label, frame)."""
    state.frames_since_infer += 1

    if state.has_run_once and state.frames_since_infer < STRIDE_FRAMES:
        label = "fall" if state.last_detected else "no_fall"
        return state.last_detected, state.last_probability, label, frame

    state.frames_since_infer = 0
    state.has_run_once = True

    cls, confidence = fall_detector.classify_frame(frame)
    state.last_probability = confidence

    if cls == "fallen":
        state.fallen_streak += 1
    else:
        state.fallen_streak = 0

    state.last_detected = state.fallen_streak >= FALLEN_STREAK_NEEDED
    label = "fall" if state.last_detected else ("no_person" if cls == "none" else "no_fall")
    return state.last_detected, confidence, label, frame
