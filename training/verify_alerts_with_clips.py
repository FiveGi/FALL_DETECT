"""Verify alerts as CLIPS, not stills -- because a fall is a motion, not a pose.

Every verification pass in this project so far (SS36, SS40's Gemini round) judged a single
frame per alert. That is the wrong instrument for this task and it cuts both ways:

  - a person lying on the ground looks identical in one frame whether they fell or lay down
    deliberately, so a still can call a false alarm "FALL";
  - a genuine fall caught mid-motion can look like ordinary walking in one frame, so a still
    can call a real catch "NOT_A_FALL".

Only the seconds around the moment can separate those. This cuts a window from PRE seconds
before the alert to POST seconds after, sends the video to Gemini, and asks whether a fall
actually happens in it -- plus, for the same window, whether the person was already on the
ground before the alert, which is the specific confusion a still cannot resolve.

It also writes a contact sheet per alert (a grid of frames across the window) so the motion
can be read by eye in one image.

Usage:
    ALERTS_FILE=prod_multi_alerts_new.json VERIFY_TAG=new \\
        python training/verify_alerts_with_clips.py
"""
import json
import os
import re
import sys
import time

import cv2
import numpy as np
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, '.env'))
from google import genai  # noqa: E402

TEST_DIR = os.environ.get('TEST_DIR', os.path.join(ROOT, 'Test'))
DATA_DIR = os.path.join(ROOT, 'training', 'data')
TAG = os.environ.get('VERIFY_TAG', 'new')
ALERTS_FILE = os.environ.get('ALERTS_FILE', f'prod_multi_alerts_{TAG}.json')
CLIP_DIR = os.path.join(DATA_DIR, f'verify_clips_{TAG}')
SHEET_DIR = os.path.join(DATA_DIR, f'verify_sheets_{TAG}')

# Seconds either side of the alert. 4s before is enough to show the person upright and moving
# before a real fall; 3s after shows whether they stay down.
PRE = float(os.environ.get('CLIP_PRE', 4.0))
POST = float(os.environ.get('CLIP_POST', 3.0))
SHEET_COLS = 5
SHEET_ROWS = 3

PROMPT = (
    "This is a few seconds of video around the moment a fall-detection system raised an "
    "alert. The alert moment is roughly in the middle of the clip.\n"
    "Judge the MOTION over the clip, not any single frame.\n"
    "- FALL means a person actually falls, collapses or loses balance to the ground during "
    "this clip.\n"
    "- ALREADY_DOWN means someone is on the ground the whole time without falling in the "
    "clip (for example lying down, sitting on the floor, or already fallen before it "
    "starts).\n"
    "- NOT_A_FALL means nobody falls: walking, standing, sitting down deliberately, bending "
    "over, exercising, or no person visible at all.\n"
    'Answer ONLY compact JSON: {"verdict": "FALL or ALREADY_DOWN or NOT_A_FALL", '
    '"reason": "short phrase"}'
)


SHOT_CUT_DIFF = float(os.environ.get('SHOT_CUT_DIFF', 45.0))


def _is_shot_cut(a, b):
    """Hard cut detector: mean absolute difference between consecutive frames, on a small
    grayscale thumbnail so ordinary motion and noise stay well under the threshold while a
    change of scene goes far above it."""
    ga = cv2.cvtColor(cv2.resize(a, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.int16)
    gb = cv2.cvtColor(cv2.resize(b, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.int16)
    return float(np.mean(np.abs(ga - gb))) > SHOT_CUT_DIFF


def cut_clip(video_path, t, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    start = max(0, int((t - PRE) * fps))
    end = int((t + POST) * fps)
    if total:
        end = min(end, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    # Read the whole window first, then keep only the run of frames belonging to the same
    # shot as the alert -- writing as we go would emit frames from the previous scene.
    window, alert_pos = [], int(min(PRE, t) * fps)
    for _ in range(max(0, end - start)):
        ok, frame = cap.read()
        if not ok:
            break
        window.append(frame)
    cap.release()
    if not window:
        return None, []

    alert_pos = min(alert_pos, len(window) - 1)
    lo = 0
    for i in range(alert_pos, 0, -1):
        if _is_shot_cut(window[i - 1], window[i]):
            lo = i
            break
    hi = len(window)
    for i in range(alert_pos + 1, len(window)):
        if _is_shot_cut(window[i - 1], window[i]):
            hi = i
            break
    frames = window[lo:hi]
    if len(frames) < int(fps):  # under a second of usable shot -- too short to judge motion
        return None, frames

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        return None, frames
    for f in frames:
        writer.write(f)
    writer.release()
    return out_path, frames


def contact_sheet(frames, out_path, marker_index):
    """A grid of frames across the window, so the motion is readable in one image. The frame
    nearest the alert is outlined, so it is obvious what the system reacted to."""
    if not frames:
        return None
    want = SHEET_COLS * SHEET_ROWS
    idxs = [int(i * (len(frames) - 1) / max(1, want - 1)) for i in range(want)]
    tile_w = 320
    tiles = []
    for n, i in enumerate(idxs):
        f = frames[i]
        scale = tile_w / f.shape[1]
        t = cv2.resize(f, (tile_w, int(f.shape[0] * scale)))
        if abs(i - marker_index) <= (len(frames) / want) / 2:
            cv2.rectangle(t, (2, 2), (t.shape[1] - 3, t.shape[0] - 3), (0, 0, 255), 4)
        cv2.putText(t, f'{n + 1}', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(t)
    h = min(t.shape[0] for t in tiles)
    tiles = [t[:h] for t in tiles]
    rows = [cv2.hconcat(tiles[r * SHEET_COLS:(r + 1) * SHEET_COLS]) for r in range(SHEET_ROWS)]
    cv2.imwrite(out_path, cv2.vconcat(rows))
    return out_path


def main():
    os.makedirs(CLIP_DIR, exist_ok=True)
    os.makedirs(SHEET_DIR, exist_ok=True)
    alerts_by_clip = json.load(open(os.path.join(DATA_DIR, ALERTS_FILE), encoding='utf-8'))
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

    results = {}
    for clip_id, alerts in sorted(alerts_by_clip.items(), key=lambda x: int(x[0])):
        video = os.path.join(TEST_DIR, f'{clip_id}.mp4')
        if not alerts or not os.path.exists(video):
            continue
        for t, p in alerts:
            key = f'{clip_id}/t={t:.1f}s_p={p:.2f}'
            base = f'{TAG}_{clip_id}_{t:.1f}'
            clip_path, frames = cut_clip(video, t, os.path.join(CLIP_DIR, base + '.mp4'))
            if not clip_path:
                # Usually means the alert sits within a second of a scene change, so there
                # is not enough continuous footage to judge a motion. Recorded rather than
                # skipped silently, since it is a property of the source material.
                results[key] = '{"verdict": "TOO_SHORT", "reason": "shot shorter than 1s around the alert"}'
                print(key, '-> TOO_SHORT (shot boundary)', flush=True)
                continue
            contact_sheet(frames, os.path.join(SHEET_DIR, base + '.jpg'),
                          marker_index=len(frames) // 2)

            verdict = 'ERROR'
            for attempt in range(4):
                try:
                    f = client.files.upload(file=clip_path)
                    # Video uploads are processed asynchronously; generating against a file
                    # still in PROCESSING fails, so wait for it to go ACTIVE.
                    for _ in range(30):
                        info = client.files.get(name=f.name)
                        if str(info.state) .endswith('ACTIVE'):
                            break
                        time.sleep(2)
                    resp = client.models.generate_content(
                        model='gemini-flash-lite-latest', contents=[f, PROMPT])
                    verdict = resp.text
                    break
                except Exception as e:
                    print(key, f'attempt {attempt} failed:', str(e)[:90], flush=True)
                    time.sleep(6)
            results[key] = verdict
            short = re.sub(r'\s+', ' ', verdict)[:110]
            print(key, '->', short, flush=True)

    out = os.path.join(DATA_DIR, f'prod_{TAG}_clipverify_results.json')
    json.dump(results, open(out, 'w', encoding='utf-8'), indent=2)
    print('\nwrote', out)
    print('clips in', CLIP_DIR)
    print('contact sheets in', SHEET_DIR)


if __name__ == '__main__':
    sys.exit(main())
