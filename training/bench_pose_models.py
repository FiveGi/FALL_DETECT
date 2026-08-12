"""Benchmarks 3 candidate person/pose models against each other, matching each
model's EXACT production call pattern (verified by reading camera_manager.py and
v3_fall_detection.py directly, not assumed). Written to answer: given v4 (RF-DETR)
is confirmed non-viable on the production server (SKILL.md SS9), is there a CPU-friendly
alternative? First run (on the real server, SS13) found MediaPipe crashes outright there
(SIGILL, no AVX) and the currently-live yolov10x.pt alone-detection model is far too
slow (3.3s/frame) -- yolo11n-pose.pt was the only one of the three that worked.

  - yolov10x.pt   : the model actually in use for alone-detection today. Called via
                    .track(source=[frame], tracker="bytetrack.yaml", conf=0.6,
                    classes=[0], verbose=False) -- see AlonePersonDetector.detect_persons
                    in app/detection/fall_detection.py. Runs on EVERY frame in
                    production (process_alone_detection has no stride/skip at all).
  - MediaPipe PoseLandmarker (v3): vision.PoseLandmarker, IMAGE mode,
                    min_pose_detection_confidence=0.5 -- see
                    V3PoseFallDetector.extract_keypoints in
                    app/detection/v3_fall_detection.py. ALSO runs on every frame in
                    production (only the downstream temporal classifier is strided,
                    not pose extraction itself) -- this matters: MediaPipe's raw
                    per-frame latency directly gates the real-time loop with zero
                    stride cushion, unlike RF-DETR (STRIDE_FRAMES=5).
  - yolo11n-pose.pt: new candidate (nano pose variant), auto-downloaded via
                    ultralytics on first use.

Benchmarks under TWO thread configurations (docker-compose.yml pins
OMP_NUM_THREADS=1 for celery_worker; the open question was whether raising that
specifically for this task would help without oversubscribing against the
container's --cpus=2.0 limit -- tested on the real server: it doesn't matter, <=1%
either way, ruling out thread oversubscription as an explanation for anything):
  - OMP_NUM_THREADS=1 (current production setting)
  - OMP_NUM_THREADS=4

Usage:
    python training/bench_pose_models.py --model yolov10x --threads 1

Runs continuously for RUN_SECONDS (env var, default 90s, not a fixed iteration
count) to catch shared-vCPU throttling drift, looping over CAUCAFall test frames.
Reports p50/p95/p99/max latency, first-half vs second-half mean (throttling
indicator), and peak RSS via /proc/self/status (Linux only -- reports NaN on
Windows dev machines, latency numbers are still valid there).
"""
import argparse
import glob
import os
import statistics
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAUCAFALL_ROOT = os.environ.get(
    "CAUCAFALL_ROOT", "d:/project/PROJECT/Dataset CAUCAFall/CAUCAFall"
)
RUN_SECONDS = int(os.environ.get("RUN_SECONDS", "90"))


def get_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return float("nan")


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[idx]


def summarize(name, latencies_ms, rss_mb):
    latencies_ms.sort()
    n = len(latencies_ms)
    half = n // 2
    first_half_mean = statistics.mean(latencies_ms[:half]) if half else float("nan")
    second_half_mean = statistics.mean(latencies_ms[half:]) if n - half else float("nan")
    drift_pct = 100 * (second_half_mean - first_half_mean) / first_half_mean if first_half_mean else 0

    print(f"\n=== {name} ===")
    print(f"  n={n} inferences over {RUN_SECONDS}s")
    print(f"  p50={percentile(latencies_ms, 0.50):.0f}ms  p95={percentile(latencies_ms, 0.95):.0f}ms  "
          f"p99={percentile(latencies_ms, 0.99):.0f}ms  max={max(latencies_ms):.0f}ms  min={min(latencies_ms):.0f}ms")
    print(f"  first-half mean={first_half_mean:.0f}ms  second-half mean={second_half_mean:.0f}ms  "
          f"drift={drift_pct:+.1f}% ({'THROTTLING SUSPECTED' if drift_pct > 15 else 'stable'})")
    print(f"  peak RSS={rss_mb:.0f}MB")


def bench_yolov10x(frames, run_seconds):
    from ultralytics import YOLO
    model = YOLO(os.path.join(REPO_ROOT, "models", "yolov10x.pt"))
    _ = model.track(source=[frames[0]], stream=False, tracker="bytetrack.yaml", conf=0.6, classes=[0], verbose=False)  # warmup

    latencies, peak_rss = [], 0.0
    t_end = time.time() + run_seconds
    i = 0
    while time.time() < t_end:
        frame = frames[i % len(frames)]
        t0 = time.time()
        model.track(source=[frame], stream=False, tracker="bytetrack.yaml", conf=0.6, classes=[0], verbose=False)
        latencies.append((time.time() - t0) * 1000)
        peak_rss = max(peak_rss, get_rss_mb())
        i += 1
    return latencies, peak_rss


def bench_mediapipe(frames, run_seconds):
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(REPO_ROOT, "models", "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    import cv2
    def to_mp_image(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    _ = landmarker.detect(to_mp_image(frames[0]))  # warmup

    latencies, peak_rss = [], 0.0
    t_end = time.time() + run_seconds
    i = 0
    while time.time() < t_end:
        frame = frames[i % len(frames)]
        mp_img = to_mp_image(frame)
        t0 = time.time()
        landmarker.detect(mp_img)
        latencies.append((time.time() - t0) * 1000)
        peak_rss = max(peak_rss, get_rss_mb())
        i += 1
    return latencies, peak_rss


def bench_yolo_pose_nano(frames, run_seconds):
    from ultralytics import YOLO
    model = YOLO("yolo11n-pose.pt")  # auto-downloads on first use
    _ = model.predict(frames[0], verbose=False)  # warmup

    latencies, peak_rss = [], 0.0
    t_end = time.time() + run_seconds
    i = 0
    while time.time() < t_end:
        frame = frames[i % len(frames)]
        t0 = time.time()
        model.predict(frame, verbose=False)
        latencies.append((time.time() - t0) * 1000)
        peak_rss = max(peak_rss, get_rss_mb())
        i += 1
    return latencies, peak_rss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yolov10x", "mediapipe", "yolo11n-pose"])
    ap.add_argument("--threads", type=int, required=True)
    args = ap.parse_args()

    import cv2
    frame_paths = sorted(glob.glob(f"{CAUCAFALL_ROOT}/Subject.2/Fall backwards/*.png"))
    frames = [cv2.imread(f) for f in frame_paths]
    print(f"Loaded {len(frames)} test frames. OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')}, "
          f"requested torch threads={args.threads}")

    try:
        import torch
        torch.set_num_threads(args.threads)
        print(f"torch.get_num_threads() = {torch.get_num_threads()}")
    except ImportError:
        pass

    fn = {"yolov10x": bench_yolov10x, "mediapipe": bench_mediapipe, "yolo11n-pose": bench_yolo_pose_nano}[args.model]
    t0 = time.time()
    latencies, peak_rss = fn(frames, RUN_SECONDS)
    elapsed = time.time() - t0

    summarize(f"{args.model} (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')})", latencies, peak_rss)
    print(f"\n(wall time for this run: {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
