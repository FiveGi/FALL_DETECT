"""Sample diverse frames from Test/ clips for a ground-truth person-count accuracy
test (person-detector comparison, requested by user: measure real accuracy, not just
agreement with the currently-deployed model)."""
import os
import cv2

TEST_DIR = os.environ.get("TEST_DIR", r"D:\project\PROJECT\Test")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "gt_frames")
os.makedirs(OUT_DIR, exist_ok=True)

# Mix of multi-scene compilation clips (varying 0-3 people) and clean single-scene
# real elderly-fall clips (1-2 people) for a realistic spread of person counts.
CLIPS = [1, 2, 4, 5, 7, 8, 9, 12, 14, 15, 16]
FRAMES_PER_CLIP = 7

saved = []
for clip_id in CLIPS:
    path = os.path.join(TEST_DIR, f"{clip_id}.mp4")
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if n <= 0:
        cap.release()
        continue
    step = max(1, n // FRAMES_PER_CLIP)
    idx = 0
    count = 0
    while idx < n and count < FRAMES_PER_CLIP:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            t = idx / fps
            out_path = os.path.join(OUT_DIR, f"{clip_id}_{t:.1f}.jpg")
            cv2.imwrite(out_path, frame)
            saved.append(out_path)
            count += 1
        idx += step
    cap.release()

print(f"Saved {len(saved)} frames to {OUT_DIR}")
