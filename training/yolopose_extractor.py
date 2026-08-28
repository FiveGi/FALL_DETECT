"""
Shared YOLO-pose keypoint extractor, used as a drop-in alternative to MediaPipe
in the dataset extraction scripts (extract_poses.py, extract_caucafall_poses.py,
extract_ofitw_poses.py) -- same (17,3) [x,y,confidence] per-frame output shape,
but ALREADY in COCO-17 order (yolo26's pose head uses the standard COCO-17
convention, index-identical to app/detection/v3_fall_detection.py's
LEFT_SHOULDER=5/RIGHT_SHOULDER=6/LEFT_HIP=11/RIGHT_HIP=12), so no 33->17 mapping
step is needed the way extract_poses.py needs for MediaPipe's 33-point output.
x/y are saved normalized to [0,1] (image-relative), matching MediaPipe's own
convention, so downstream code (dataset.py's to_coco17/normalize) needs no changes.
"""
import numpy as np
from ultralytics import YOLO

NUM_KEYPOINTS = 17


class YoloPoseExtractor:
    def __init__(self, model_name="yolo26s-pose.pt", conf=0.5):
        self.model = YOLO(model_name)
        self.conf = conf

    def extract_keypoints(self, frame_bgr):
        """-> ((17,3) [x,y,confidence] normalized to [0,1], found: bool) for the
        highest-confidence detected person. Zeros + False if nobody detected."""
        h, w = frame_bgr.shape[:2]
        r = self.model.predict(frame_bgr, verbose=False, conf=self.conf, classes=[0])[0]
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            return np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32), False

        box_confs = r.boxes.conf.cpu().numpy()
        best = int(box_confs.argmax())
        kxy = r.keypoints.xy[best].cpu().numpy()  # (17,2) pixel coords
        kconf = r.keypoints.conf[best].cpu().numpy() if r.keypoints.conf is not None else np.ones(NUM_KEYPOINTS)

        kpts = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        kpts[:, 0] = kxy[:, 0] / w
        kpts[:, 1] = kxy[:, 1] / h
        kpts[:, 2] = kconf
        return kpts, True
