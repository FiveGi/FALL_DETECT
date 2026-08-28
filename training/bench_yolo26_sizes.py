"""
Compare yolov10x (currently deployed, alone-detection person counter) against
yolo26 in n/s/m/l/x sizes: speed (matching production's exact .track() call,
bytetrack, conf=0.6, classes=[0]) + person-count consistency across real Test/
clips. Throttled to 4 threads to match the production server's actual 4 vCPUs
(see memory: 103.114.203.22 has only 4 online vCPUs).
"""
import os
import time
import cv2
import torch

torch.set_num_threads(4)

from ultralytics import YOLO

TEST_DIR = r"D:\project\PROJECT\Test"
MODELS = ["yolov10x.pt", "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]

CLIPS = ["1.mp4", "5.mp4", "9.mp4", "13.mp4"]
FRAMES_PER_CLIP = 15
CONF = 0.6

def sample_frames():
    frames = []
    for clip in CLIPS:
        path = os.path.join(TEST_DIR, clip)
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            cap.release()
            continue
        step = max(1, n // FRAMES_PER_CLIP)
        idx = 0
        while idx < n and len(frames) < len(CLIPS) * FRAMES_PER_CLIP:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append((clip, idx, frame))
            idx += step
        cap.release()
    return frames

frames = sample_frames()
print(f"Sampled {len(frames)} frames from {CLIPS}\n")

results = {}
for model_name in MODELS:
    print(f"=== {model_name} ===")
    model = YOLO(model_name)
    counts = []
    times = []
    for i, (clip, idx, frame) in enumerate(frames):
        t0 = time.perf_counter()
        r = model.track(source=[frame], stream=False, tracker="bytetrack.yaml",
                         conf=CONF, classes=[0], verbose=False, persist=False)
        dt = time.perf_counter() - t0
        if i >= 3:  # skip first 3 as warmup
            times.append(dt)
        n_persons = len(r[0].boxes) if r and r[0].boxes is not None else 0
        counts.append((clip, idx, n_persons))
    avg_time = sum(times) / len(times) if times else 0.0
    results[model_name] = {"avg_time": avg_time, "counts": counts}
    print(f"  avg inference time: {avg_time:.4f}s/frame ({1/avg_time:.1f} fps)" if avg_time else "  no timing data")
    print()

print("=== SUMMARY: speed (4-thread CPU) ===")
baseline = results["yolov10x.pt"]["avg_time"]
for m in MODELS:
    t = results[m]["avg_time"]
    rel = (t / baseline - 1) * 100 if baseline else 0
    sign = "+" if rel >= 0 else ""
    print(f"  {m}: {t:.4f}s/frame  ({sign}{rel:.0f}% vs yolov10x)")

print("\n=== SUMMARY: per-frame person-count agreement vs yolov10x ===")
ref = {(c, i): n for c, i, n in results["yolov10x.pt"]["counts"]}
for m in MODELS:
    if m == "yolov10x.pt":
        continue
    diffs = 0
    total = 0
    for c, i, n in results[m]["counts"]:
        total += 1
        if ref.get((c, i)) != n:
            diffs += 1
    print(f"  {m}: {total - diffs}/{total} frames match yolov10x's person count exactly")
