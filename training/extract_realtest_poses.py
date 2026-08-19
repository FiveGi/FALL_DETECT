"""Turn the Gemini+Claude-verified alerts from Test/4.mp4-11.mp4 (SS21) into training
segments: real falls become positive examples, and -- the useful part -- the false
positives we caught (bed-lying, dancing with raised arms, standing near objects) become
explicit hard negatives, teaching the model "this exact pose is NOT a fall" instead of
hoping more generic ADL data covers it.

Each verified point becomes one short segment (+-1.5s around the timestamp, clamped to
clip bounds), pose-extracted and saved with a constant frame_labels array -- same
format/reasoning as extract_ofitw_poses.py's real annotated segments, not a heuristic.
"""
import os
import re
import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

TEST_DIR = r"D:\project\PROJECT\Test"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "poses_realtest_v1")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models_cache", "pose_landmarker_lite.task")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "data", "gemini_screen_new_clips.json")
NUM_LANDMARKS = 33
HALF_WINDOW_S = 1.5

# Degenerate cases with no real person in frame -- not useful pose training examples,
# would just be a segment of all-zero keypoints regardless of label.
EXCLUDE = {
    ("10", "t=0015.4s_p=0.92.jpg"),  # TikTok loading screen, no person
    ("9", "t=0048.9s_p=0.78.jpg"),   # "no person visible in the image"
}

# The 3 that hit a transient 503 on the first pass and were retried separately
# (see conversation) -- not in the saved JSON, added by hand.
MANUAL_VERDICTS = {
    ("5", "t=0008.3s_p=0.95.jpg"): "FALL",
    ("8", "t=0033.1s_p=0.95.jpg"): "NOT_A_FALL",
    ("8", "t=0045.4s_p=0.58.jpg"): "FALL",
}


def parse_verdict(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))["verdict"]
    except Exception:
        return None


def make_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_segment_keypoints(video_path, start_s, end_s, landmarker):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = max(0, int(start_s * fps))
    end_frame = max(start_frame + 1, int(end_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    sequence = []
    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            kpts = np.array([[p.x, p.y, p.visibility] for p in lm], dtype=np.float32)
        else:
            kpts = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        sequence.append(kpts)
        frame_idx += 1
    cap.release()
    if not sequence:
        return None
    return np.stack(sequence, axis=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RESULTS_PATH) as f:
        raw_results = json.load(f)

    points = []  # (clip, timestamp_s, label)
    for key, text in raw_results.items():
        clip, fname = key.split("/", 1)
        if (clip, fname) in EXCLUDE:
            continue
        verdict = parse_verdict(text)
        if verdict is None:
            print(f"  skip unparseable: {key} -> {text!r}")
            continue
        m = re.search(r"t=(\d+\.\d+)s", fname)
        t = float(m.group(1))
        points.append((clip, t, 1 if verdict == "FALL" else 0))

    for (clip, fname), verdict in MANUAL_VERDICTS.items():
        m = re.search(r"t=(\d+\.\d+)s", fname)
        t = float(m.group(1))
        points.append((clip, t, 1 if verdict == "FALL" else 0))

    print(f"{len(points)} verified points -> segments "
          f"({sum(1 for _,_,l in points if l==1)} fall, {sum(1 for _,_,l in points if l==0)} not-fall)")

    landmarker = make_landmarker()
    total = 0
    for clip, t, label in sorted(points):
        video_path = os.path.join(TEST_DIR, f"{clip}.mp4")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = n_frames / fps if fps else 0
        cap.release()

        start_s = max(0.0, t - HALF_WINDOW_S)
        end_s = min(duration, t + HALF_WINDOW_S)

        out_name = f"realtest_{clip}_{t:.1f}s_label{label}.npz"
        out_path = os.path.join(OUT_DIR, out_name)
        seq = extract_segment_keypoints(video_path, start_s, end_s, landmarker)
        if seq is None or len(seq) < 5:
            print(f"  SKIP (too short): clip {clip} t={t}")
            continue

        frame_labels = np.full(len(seq), label, dtype=np.int64)
        np.savez_compressed(
            out_path, keypoints=seq, label=label, subject=-1, frame_labels=frame_labels
        )
        total += 1

    landmarker.close()
    print(f"\nDone. Extracted {total} segments to {OUT_DIR}")


if __name__ == "__main__":
    main()
