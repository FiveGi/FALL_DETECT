"""
Same methodology/clip lists as eval_v3_on_gmdcsa24_val.py + eval_v3_on_gmdcsa24_train50.py,
but with keypoints extracted via YOLO-pose (yolopose_extractor.py) instead of MediaPipe,
so a model TRAINED on YOLO-pose features is evaluated on features from the same
distribution -- testing against MediaPipe-extracted keypoints here would be an invalid
train/inference mismatch. Reuses v3_fall_detection.py's exact _step_person state machine
(pose-extraction-agnostic -- takes raw (17,3) keypoints, doesn't care how they were made)
so the alert logic/thresholds/smoothing are identical to production, only the pose
backend and the ONNX model differ.
"""
import os
import sys
import importlib.util
import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location(
    "v3_fall_detection", os.path.join(ROOT, "app", "detection", "v3_fall_detection.py")
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
V3FallDetectionState = v3.V3FallDetectionState
_step_person = v3._step_person
WINDOW_SIZE = v3.WINDOW_SIZE
THRESHOLD = v3.THRESHOLD

from yolopose_extractor import YoloPoseExtractor

ONNX_PATH = os.environ.get("EVAL_ONNX_PATH", "")
YOLOPOSE_MODEL = os.environ.get("YOLOPOSE_MODEL", "yolo26s-pose.pt")
EVAL_THRESHOLD = float(os.environ.get("EVAL_THRESHOLD", THRESHOLD))


class OnnxFallClassifier:
    """Minimal stand-in for V3PoseFallDetector's .predict_window(), pointed at an
    arbitrary ONNX checkpoint instead of the fixed models/fall_classifier_v3.onnx."""
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def predict_window(self, raw_window):
        feat = v3._normalize_and_velocity(raw_window).reshape(1, WINDOW_SIZE, -1).astype(np.float32)
        logit = self.session.run(["logit"], {"input": feat})[0]
        return float(1.0 / (1.0 + np.exp(-logit.reshape(-1)[0])))


class EnsembleOnnxFallClassifier:
    """Averages predict_window() probability across multiple seeds' ONNX checkpoints --
    a cheap (no retraining) way to smooth out the seed-to-seed variance found when
    testing individual YOLO-pose checkpoints against Test/14-16 (SKILL.md SS34)."""
    def __init__(self, onnx_paths):
        self.classifiers = [OnnxFallClassifier(p) for p in onnx_paths]

    def predict_window(self, raw_window):
        probs = [c.predict_window(raw_window) for c in self.classifiers]
        return sum(probs) / len(probs)


def make_classifier(onnx_path_str):
    paths = [p.strip() for p in onnx_path_str.split(",") if p.strip()]
    if len(paths) == 1:
        return OnnxFallClassifier(paths[0])
    return EnsembleOnnxFallClassifier(paths)


VAL_FALL = ["s1_Fall_02", "s1_Fall_06", "s1_Fall_16", "s2_Fall_02", "s2_Fall_04",
            "s2_Fall_09", "s2_Fall_14", "s2_Fall_20", "s3_Fall_02", "s3_Fall_09",
            "s3_Fall_12", "s3_Fall_13", "s3_Fall_16", "s4_Fall_06", "s4_Fall_17"]
VAL_ADL = ["s1_ADL_01", "s1_ADL_05", "s1_ADL_11", "s1_ADL_13", "s2_ADL_03", "s2_ADL_07",
           "s2_ADL_13", "s2_ADL_15", "s2_ADL_16", "s2_ADL_18", "s2_ADL_20", "s3_ADL_07",
           "s3_ADL_11", "s4_ADL_07", "s4_ADL_08", "s4_ADL_10"]
TRAIN_FALL = ['s4_Fall_13', 's3_Fall_04', 's2_Fall_21', 's1_Fall_07', 's4_Fall_08',
              's3_Fall_20', 's2_Fall_17', 's2_Fall_08', 's3_Fall_14', 's1_Fall_05',
              's2_Fall_10', 's1_Fall_10', 's2_Fall_19', 's2_Fall_11', 's4_Fall_11',
              's4_Fall_02', 's1_Fall_09', 's3_Fall_05', 's1_Fall_01', 's2_Fall_07',
              's4_Fall_09', 's1_Fall_03', 's4_Fall_01', 's1_Fall_13', 's4_Fall_10']
TRAIN_ADL = ['s4_ADL_16', 's4_ADL_15', 's2_ADL_08', 's3_ADL_12', 's3_ADL_22',
             's3_ADL_16', 's3_ADL_20', 's1_ADL_16', 's4_ADL_17', 's2_ADL_05',
             's3_ADL_10', 's2_ADL_09', 's2_ADL_04', 's1_ADL_09', 's4_ADL_18',
             's4_ADL_04', 's4_ADL_03', 's3_ADL_09', 's3_ADL_13', 's2_ADL_11',
             's3_ADL_04', 's4_ADL_01', 's4_ADL_11', 's3_ADL_05', 's3_ADL_02']

FALL_DIR = os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_fall_raw")
VAL_ADL_DIR = os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_adl_raw_val")
TRAIN_ADL_DIR = os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_adl_raw_train50")


def run_clip(extractor, classifier, path, threshold=THRESHOLD):
    state = V3FallDetectionState()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    alerts = []
    last_label = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        kpts, found = extractor.extract_keypoints(frame)
        detected, probability, label = _step_person(kpts, found, state, classifier, threshold)
        if label != last_label:
            if label == "fall":
                alerts.append((t, round(probability, 2)))
            last_label = label
        frame_idx += 1
    cap.release()
    return alerts


def run_set(extractor, classifier, name, fall_list, adl_list, fall_dir, adl_dir, threshold=None):
    threshold = threshold if threshold is not None else EVAL_THRESHOLD
    print(f"\n########## {name} (threshold={threshold}) ##########")
    print("=== Fall clips (want >=1 alert each) ===")
    fall_caught = 0
    for cname in fall_list:
        path = os.path.join(fall_dir, cname + ".mp4")
        if not os.path.exists(path):
            print(f"  {cname}: MISSING FILE")
            continue
        alerts = run_clip(extractor, classifier, path, threshold=threshold)
        caught = len(alerts) > 0
        fall_caught += caught
        marker = "OK" if caught else "MISSED"
        print(f"  {cname}: {marker}  alerts={alerts}")

    print("=== ADL clips (want 0 alerts each) ===")
    adl_clean = 0
    for cname in adl_list:
        path = os.path.join(adl_dir, cname + ".mp4")
        if not os.path.exists(path):
            print(f"  {cname}: MISSING FILE")
            continue
        alerts = run_clip(extractor, classifier, path, threshold=threshold)
        clean = len(alerts) == 0
        adl_clean += clean
        marker = "OK" if clean else "FALSE ALARM"
        print(f"  {cname}: {marker}  alerts={alerts}")

    print(f"\n{name} Fall clips caught: {fall_caught}/{len(fall_list)} ({fall_caught/len(fall_list):.1%})")
    print(f"{name} ADL clips clean: {adl_clean}/{len(adl_list)} ({adl_clean/len(adl_list):.1%})")


def main():
    extractor = YoloPoseExtractor(YOLOPOSE_MODEL)
    classifier = make_classifier(ONNX_PATH)

    run_set(extractor, classifier, "VAL", VAL_FALL, VAL_ADL, FALL_DIR, VAL_ADL_DIR)
    run_set(extractor, classifier, "TRAIN50", TRAIN_FALL, TRAIN_ADL, FALL_DIR, TRAIN_ADL_DIR)


if __name__ == "__main__":
    main()
