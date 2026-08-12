"""Measures false-positives-per-hour for the v4 (RF-DETR) fall detector on long-form
video known to contain NO genuine falls (e.g. general ADL / activities-of-daily-living
footage such as Toyota Smarthome, NTU RGB+D's RGB stream, or any home-video source
you're confident has zero real fall events).

Why this metric, and why it's cheap: because the footage is known fall-free, every
alert the model fires is a false positive by construction -- no ground-truth labeling
needed. That makes it available as a baseline *before* any new dataset access is
granted and *before* server-side latency numbers exist (this measures accuracy, not
speed, so it runs fine on a dev machine). See SKILL.md SS6/SS7.3: the ~52.6% false-positive
rate found on bending/kneeling/floor-sitting stock photos is the problem this measures
in a more realistic, continuous-footage form.

Runs the literal production detect_v4_fall() frame-by-frame (respecting
STRIDE_FRAMES / FALLEN_STREAK_NEEDED exactly as camera_manager.py's Celery task would),
and groups consecutive alert frames into a single "event" rather than counting every
flagged frame separately -- one false alarm that lasts 4 seconds should count once, not
once per sampled frame within it (see the reviewer's dedupe-by-event point: sampling
every STRIDE_FRAMES frames of a person kneeling for 30s would otherwise yield dozens of
near-duplicate "false positive" frames from a single event).

Usage:
    python training/measure_fp_per_hour.py --video-dir /path/to/videos --out-dir out/

Outputs (out-dir):
  - raw_results.json: EVERY sampled frame's (video, timestamp_s, label, confidence,
    detected) -- not just the ones that fired an alert. Kept so future decision-rule
    changes (threshold, streak, stride) can be re-scored offline without re-running
    inference, matching the capture-then-sweep pattern already used elsewhere in this
    project (SKILL.md SS4/SS8, rfdetr_raw_results.json).
  - events.json: one entry per detected event (video, start_s, end_s, peak_confidence,
    candidate_frame_path).
  - candidates/: the peak-confidence frame image from each event, saved as a
    hard-negative fine-tuning candidate (this is the raw material for actually fixing
    SS7.3, if these turn out to be bending/kneeling/floor poses as expected).
  - summary printed to stdout: total hours processed, total events, FP/hour.
"""
import argparse
import glob
import importlib.util
import json
import os
import time

import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

spec = importlib.util.spec_from_file_location(
    "v4_fall_detection_rfdetr",
    os.path.join(REPO_ROOT, "app", "detection", "v4_fall_detection_rfdetr.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def process_video(path, detector):
    """Runs detect_v4_fall() over every frame of one video. Returns
    (raw_frame_results, events, duration_s)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  !! could not open {path}")
        return [], [], 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0.0

    state = mod.V4FallDetectionState()
    raw = []
    events = []
    current_event = None  # {"start_s", "peak_conf", "peak_frame_idx", "peak_frame"}
    frame_idx = 0
    last_infer_frame = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_s = frame_idx / fps

        detected, prob, label, _ = mod.detect_v4_fall(frame, state, detector, config={})

        # Only log a raw entry when a *new* inference actually happened (state.frames_since_infer
        # resets to 0 right after a real inference call), so raw_results.json reflects the real
        # sampled signal rather than STRIDE_FRAMES-1 duplicate copies of the cached value.
        if state.frames_since_infer == 0:
            raw.append({"video": os.path.basename(path), "t_s": round(t_s, 2),
                        "label": label, "confidence": round(prob, 4), "detected": bool(detected)})

            if detected:
                if current_event is None:
                    current_event = {"start_s": t_s, "peak_conf": prob, "peak_frame": frame.copy()}
                elif prob > current_event["peak_conf"]:
                    current_event["peak_conf"] = prob
                    current_event["peak_frame"] = frame.copy()
            else:
                if current_event is not None:
                    current_event["end_s"] = t_s
                    events.append(current_event)
                    current_event = None

        frame_idx += 1

    if current_event is not None:
        current_event["end_s"] = duration_s
        events.append(current_event)

    cap.release()
    return raw, events, duration_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True, help="Folder of known-fall-free videos")
    ap.add_argument("--out-dir", default="fp_per_hour_out", help="Where to write results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    candidates_dir = os.path.join(args.out_dir, "candidates")
    os.makedirs(candidates_dir, exist_ok=True)

    detector = mod.V4RFDETRFallDetector(model_dir=os.path.join(REPO_ROOT, "models"))

    videos = sorted(
        f for f in glob.glob(os.path.join(args.video_dir, "**", "*"), recursive=True)
        if f.lower().endswith(VIDEO_EXTS)
    )
    if not videos:
        print(f"No videos found under {args.video_dir}")
        return

    print(f"Found {len(videos)} videos.")

    all_raw = []
    all_events = []
    total_duration_s = 0.0
    t0 = time.time()

    for i, path in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {os.path.basename(path)} ...", flush=True)
        raw, events, duration_s = process_video(path, detector)
        total_duration_s += duration_s

        for j, ev in enumerate(events):
            frame_name = f"{os.path.splitext(os.path.basename(path))[0]}_event{j:03d}_t{ev['start_s']:.0f}s.jpg"
            frame_path = os.path.join(candidates_dir, frame_name)
            cv2.imwrite(frame_path, ev["peak_frame"])
            all_events.append({
                "video": os.path.basename(path),
                "start_s": round(ev["start_s"], 2),
                "end_s": round(ev["end_s"], 2),
                "peak_confidence": round(ev["peak_conf"], 4),
                "candidate_frame": frame_name,
            })

        all_raw.extend(raw)
        print(f"    duration={duration_s/60:.1f}min, events={len(events)}")

    elapsed = time.time() - t0
    total_hours = total_duration_s / 3600.0
    fp_per_hour = len(all_events) / total_hours if total_hours > 0 else float("nan")

    with open(os.path.join(args.out_dir, "raw_results.json"), "w") as f:
        json.dump(all_raw, f, indent=1)
    with open(os.path.join(args.out_dir, "events.json"), "w") as f:
        json.dump(all_events, f, indent=1)

    print(f"\n=== SUMMARY ({elapsed:.0f}s to process) ===")
    print(f"Videos processed: {len(videos)}")
    print(f"Total footage: {total_hours:.2f} hours (known fall-free)")
    print(f"Total false-positive events: {len(all_events)}")
    print(f"FP/hour: {fp_per_hour:.2f}")
    print(f"Candidate hard-negative frames saved to: {candidates_dir}")


if __name__ == "__main__":
    main()
