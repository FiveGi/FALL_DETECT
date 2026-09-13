"""Render what the multi-person tracker actually sees, so a human (or an agent with vision)
can check it rather than trusting the counts.

`diagnose_multi_person.py` reports things like "7 people" on a clip. That number is the
detector's own claim, and this project has already been bitten once by trusting a model's
self-report (SS27: furniture and cars counted as people). This draws every tracked person's
skeleton, box and track id onto sampled frames and writes them out, plus a per-clip summary
of the things that would make multi-person tracking untrustworthy even when the counts look
right:

  id_switches     a track whose hip centre jumps further than MAX_TRACK_DISTANCE between
                  consecutive seen frames -- i.e. the id probably moved to a different body.
  short_tracks    tracks that existed for fewer than WINDOW_SIZE frames. These can never
                  produce a real classification (the window never fills), so a scene made of
                  them is churn, not tracking.
  id_churn        total ids created divided by the peak simultaneous person count. 1.0 means
                  every person got exactly one id for the whole clip; 5.0 means ids are being
                  recycled constantly.
  fall_track      for frames where an alert fires, which track id it belonged to and how long
                  that track had existed -- so a fall can be checked against the person who
                  actually fell.

Usage:
    python training/audit_multiperson_tracking.py 12 16 1     # clips to audit
    python training/audit_multiperson_tracking.py --frames 6  # frames saved per clip
"""
import argparse
import importlib.util
import json
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    'v3_fall_detection', os.path.join(ROOT, 'app', 'detection', 'v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

TEST_DIR = os.environ.get('TEST_DIR', os.path.join(ROOT, 'Test'))
MODEL_DIR = os.environ.get('TEST_MODEL_DIR', os.path.join(ROOT, 'models'))
OUT_DIR = os.path.join(ROOT, 'training', 'data', 'multiperson_audit')

# COCO-17 skeleton, for drawing only.
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255), (255, 128, 0),
          (128, 0, 255), (0, 0, 255), (255, 255, 0)]


def draw(frame, results, fps_t):
    out = frame.copy()
    h, w = out.shape[:2]
    for track_id, detected, prob, label, centroid in results:
        color = COLORS[track_id % len(COLORS)]
        if label == 'fall' or detected:
            color = (0, 0, 255)
        cx, cy = int(centroid[0] * w), int(centroid[1] * h)
        cv2.circle(out, (cx, cy), 6, color, -1)
        cv2.putText(out, f'#{track_id} {prob:.2f}' + (' FALL' if detected else ''),
                    (max(cx - 40, 2), max(cy - 12, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2, cv2.LINE_AA)
    cv2.putText(out, f't={fps_t:.1f}s  tracked={len(results)}', (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_poses(frame, people):
    """people: list of (kpts17, hip) straight from the detector -- drawn separately from the
    tracker's view so a detection with no track, or a track with no detection, is visible."""
    out = frame
    h, w = out.shape[:2]
    for i, (kp, _) in enumerate(people):
        color = COLORS[i % len(COLORS)]
        pts = [(int(x * w), int(y * h)) if c > 0.3 else None for x, y, c in kp]
        for a, b in EDGES:
            if pts[a] and pts[b]:
                cv2.line(out, pts[a], pts[b], color, 2, cv2.LINE_AA)
        for p in pts:
            if p:
                cv2.circle(out, p, 3, color, -1)
        xs = [p[0] for p in pts if p]
        ys = [p[1] for p in pts if p]
        if xs and ys:
            cv2.rectangle(out, (min(xs) - 6, min(ys) - 6), (max(xs) + 6, max(ys) + 6), color, 1)
    return out


def audit(detector, clip, save_frames):
    path = os.path.join(TEST_DIR, f'{clip}.mp4')
    if not os.path.exists(path):
        return None
    state = v3.V3MultiPersonFallState()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    born, last_seen_centroid, switches = {}, {}, 0
    peak, idx, saved = 0, 0, 0
    alerts = []
    # Save frames spread across the clip, biased to frames with the most people, so the saved
    # images actually show the multi-person case instead of whatever frame came first.
    candidates = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        people = detector.extract_all_keypoints(frame)
        results = v3.detect_v3_fall_multi(frame, state, detector, config=None)
        peak = max(peak, len(results))

        for track_id, _, _, _, centroid in results:
            born.setdefault(track_id, idx)
            prev = last_seen_centroid.get(track_id)
            if prev is not None:
                if float(np.linalg.norm(np.array(centroid) - np.array(prev))) > v3.MAX_TRACK_DISTANCE:
                    switches += 1
            last_seen_centroid[track_id] = centroid

        firing = [r for r in results if r[1]]
        if firing:
            for track_id, _, prob, _, _ in firing:
                alerts.append({'t': round(idx / fps, 1), 'track': track_id,
                               'prob': round(float(prob), 2),
                               'track_age': idx - born.get(track_id, idx),
                               'tracked_now': len(results)})
        candidates.append((len(results), bool(firing), idx, frame, people, results))
        if len(candidates) > 400:            # keep memory bounded on long clips
            candidates.sort(key=lambda c: (c[1], c[0]), reverse=True)
            candidates = candidates[:80]
        idx += 1
    cap.release()

    candidates.sort(key=lambda c: (c[1], c[0]), reverse=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    picked = []
    for n_people, fired, i, frame, people, results in candidates:
        if saved >= save_frames:
            break
        if any(abs(i - j) < total / (save_frames * 2) for j in picked):
            continue
        img = draw_poses(frame, people)
        img = draw(img, results, i / fps)
        name = f'clip{clip}_f{i:05d}_{n_people}people{"_FALL" if fired else ""}.jpg'
        cv2.imwrite(os.path.join(OUT_DIR, name), img)
        picked.append(i)
        saved += 1

    ages = {t: idx - b for t, b in born.items()}
    short = sum(1 for a in ages.values() if a < v3.WINDOW_SIZE)
    return {
        'clip': clip, 'frames': idx, 'peak_tracked': peak, 'ids_created': len(born),
        'id_churn': round(len(born) / max(peak, 1), 2),
        'short_tracks': short, 'id_switches': switches, 'alerts': alerts,
        'saved_frames': saved,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clips', nargs='*', default=None)
    ap.add_argument('--frames', type=int, default=4)
    args = ap.parse_args()
    clips = args.clips or ['1', '4', '8', '12', '16', '17']

    detector = v3.V3PoseFallDetector(model_dir=MODEL_DIR)
    report = []
    print(f"{'clip':>5} {'peak':>5} {'ids':>5} {'churn':>6} {'short':>6} {'switch':>7} {'alerts':>7}")
    for c in clips:
        r = audit(detector, c, args.frames)
        if not r:
            continue
        report.append(r)
        print(f"{r['clip']:>5} {r['peak_tracked']:>5} {r['ids_created']:>5} "
              f"{r['id_churn']:>6} {r['short_tracks']:>6} {r['id_switches']:>7} "
              f"{len(r['alerts']):>7}", flush=True)

    dest = os.path.join(ROOT, 'training', 'data', 'multiperson_audit.json')
    with open(dest, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    print('\nframes written to', OUT_DIR)
    print('report written to', dest)


if __name__ == '__main__':
    main()
