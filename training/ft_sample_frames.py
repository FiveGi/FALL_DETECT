"""Sample a larger, denser set of frames from Test/ clips for YOLO fine-tuning
(pseudo-label self-training), explicitly excluding any timestamp too close to
the existing Gemini ground-truth test frames (data/gt_frames/) to avoid
train/test leakage."""
import os
import re
import glob
import cv2

TEST_DIR = r"D:\project\PROJECT\Test"
GT_FRAMES_DIR = os.path.join(os.path.dirname(__file__), "data", "gt_frames")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "ft_frames")
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDE_MARGIN_S = 1.0  # skip any sample within 1s of a held-out GT frame, same clip

# Parse held-out GT frame timestamps per clip so we can exclude them.
gt_times_by_clip = {}
for fp in glob.glob(os.path.join(GT_FRAMES_DIR, "*.jpg")):
    name = os.path.basename(fp)
    m = re.match(r"(\d+)_([\d.]+)\.jpg", name)
    if m:
        clip_id, t = m.group(1), float(m.group(2))
        gt_times_by_clip.setdefault(clip_id, []).append(t)

CLIPS = list(range(1, 18))
SAMPLE_INTERVAL_S = 1.5

saved = 0
for clip_id in CLIPS:
    cid = str(clip_id)
    path = os.path.join(TEST_DIR, f"{clip_id}.mp4")
    if not os.path.exists(path):
        continue
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n / fps
    excl = gt_times_by_clip.get(cid, [])

    t = 0.0
    while t < duration:
        if not any(abs(t - gt) < EXCLUDE_MARGIN_S for gt in excl):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, frame = cap.read()
            if ok:
                out_path = os.path.join(OUT_DIR, f"{clip_id}_{t:.1f}.jpg")
                cv2.imwrite(out_path, frame)
                saved += 1
        t += SAMPLE_INTERVAL_S
    cap.release()

print(f"Saved {saved} fine-tuning frames to {OUT_DIR} (excluded frames within {EXCLUDE_MARGIN_S}s of held-out GT set)")
