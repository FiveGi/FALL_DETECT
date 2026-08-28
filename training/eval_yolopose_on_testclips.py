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
from eval_yolopose_on_gmdcsa24 import make_classifier, run_clip

ONNX_PATH = os.environ["EVAL_ONNX_PATH"]
YOLOPOSE_MODEL = os.environ.get("YOLOPOSE_MODEL", "yolo26s-pose.pt")
EVAL_THRESHOLD = float(os.environ.get("EVAL_THRESHOLD", THRESHOLD))
TEST_DIR = r"D:\project\PROJECT\Test"

CLIPS = [f"{i}.mp4" for i in range(1, 18)]


def main():
    extractor = YoloPoseExtractor(YOLOPOSE_MODEL)
    classifier = make_classifier(ONNX_PATH)

    for clip in CLIPS:
        path = os.path.join(TEST_DIR, clip)
        if not os.path.exists(path):
            print(f"{clip}: MISSING")
            continue
        alerts = run_clip(extractor, classifier, path, threshold=EVAL_THRESHOLD)
        marker = "ALERT" if alerts else "no alert"
        print(f"{clip}: {marker}  alerts={alerts}")


if __name__ == "__main__":
    main()
