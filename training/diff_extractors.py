import os
import importlib.util
import numpy as np
import cv2

spec = importlib.util.spec_from_file_location(
    "v3_fall_detection", os.path.join("..", "app", "detection", "v3_fall_detection.py")
)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

from yolopose_extractor import YoloPoseExtractor

prod_detector = v3.V3PoseFallDetector(model_dir=os.path.join("..", "models"))
standalone_extractor = YoloPoseExtractor(model_name=os.path.join("..", "models", "yolo26s-pose.pt"))

cap = cv2.VideoCapture("data/gmdcsa24_adl_raw_val/s2_ADL_15.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_idx = 0
max_diff = 0.0
n_diff_found = 0
while frame_idx < 100:
    ok, frame = cap.read()
    if not ok:
        break
    kpts_prod, found_prod = prod_detector.extract_keypoints(frame)
    kpts_standalone, found_standalone = standalone_extractor.extract_keypoints(frame)
    if found_prod != found_standalone:
        print(f"frame {frame_idx}: found mismatch! prod={found_prod} standalone={found_standalone}")
        n_diff_found += 1
    else:
        d = np.abs(kpts_prod - kpts_standalone).max()
        max_diff = max(max_diff, d)
        if d > 1e-4:
            print(f"frame {frame_idx}: kpts differ, max_abs_diff={d:.6f}")
    frame_idx += 1
cap.release()
print(f"\nmax keypoint diff over {frame_idx} frames: {max_diff:.6f}")
print(f"found-mismatches: {n_diff_found}")
