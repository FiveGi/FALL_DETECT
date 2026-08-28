import os
import sys
import importlib.util
import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location(
    "v3_fall_detection", os.path.join(ROOT, "app", "detection", "v3_fall_detection.py")
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
V3FallDetectionState = v3.V3FallDetectionState
_step_person = v3._step_person
THRESHOLD = v3.THRESHOLD

from yolopose_extractor import YoloPoseExtractor
from eval_yolopose_on_gmdcsa24 import OnnxFallClassifier

import sys
clip = sys.argv[1] if len(sys.argv) > 1 else "14.mp4"
onnx_path = sys.argv[2] if len(sys.argv) > 2 else "data/yolopose_seed42.onnx"

extractor = YoloPoseExtractor("yolo26s-pose.pt")
classifier = OnnxFallClassifier(onnx_path)

path = os.path.join(r"D:\project\PROJECT\Test", clip)
state = V3FallDetectionState()
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = frame_idx / fps
    kpts, found = extractor.extract_keypoints(frame)
    detected, probability, label = _step_person(kpts, found, state, classifier, THRESHOLD)
    print(f"t={t:.2f}s found={found} prob={probability:.3f} label={label}")
    frame_idx += 1
cap.release()
