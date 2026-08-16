"""Use Gemini to find the real fall-onset timestamp in each GMDCSA24 Fall clip,
replacing dataset.py's motion-energy-peak heuristic with an actual visual judgment.

ADL clips need no relabeling (frame_labels is already exactly all-zero, not a
guess -- there's no fall in them). Only the 79 Fall clips are ambiguous: the
whole-clip label is correct ("this clip contains a fall") but *which frames*
count as the fall was previously guessed from the frame with the largest
frame-to-frame keypoint displacement, which in practice tends to land near
impact/landing rather than the true onset of losing balance (verified on
s1_Fall_01: heuristic peak = 5.68s, Gemini onset = 3.1s -- a 2.5s gap of real
falling motion the old heuristic was labeling as "not fall").

Writes one JSON result per clip to data/gemini_relabel_results.json (resumable --
already-labeled clips are skipped on rerun) and does not touch data/poses/ itself;
build_relabeled_dataset.py consumes the results to produce a new directory.
"""
import os
import re
import json
import time
import glob

import cv2
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "gmdcsa24_fall_raw")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "data", "gemini_relabel_results.json")
MODEL = "gemini-flash-lite-latest"  # gemini-2.5-flash's free tier is 20 req/day and was
# exhausted after ~9 clips; flash-lite-latest draws from a separate quota pool
MAX_RETRIES = 5

PROMPT = """You are labeling a fall-detection training clip. Watch the whole video.
Identify the exact timestamp (in seconds, one decimal place) when the person BEGINS
to fall -- the moment they first lose balance / start the falling motion. Not the
moment they land, not before -- the onset of the fall itself.
If for some reason no fall occurs in this clip, say so.
Respond ONLY with compact JSON: {"fall_onset_s": <number>, "confidence": "high|medium|low", "note": "<short reason>"}"""


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def video_duration_s(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames / fps if fps else None


def parse_response(text, duration):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    onset = data.get("fall_onset_s")
    if not isinstance(onset, (int, float)):
        return None
    if duration is not None and not (0 <= onset <= duration):
        return None
    return {
        "fall_onset_s": float(onset),
        "confidence": data.get("confidence", "unknown"),
        "note": data.get("note", ""),
    }


def label_one(client, path, duration):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            f = client.files.upload(file=path)
            while f.state.name == "PROCESSING":
                time.sleep(2)
                f = client.files.get(name=f.name)
            if f.state.name != "ACTIVE":
                raise RuntimeError(f"upload ended in state {f.state.name}")

            resp = client.models.generate_content(model=MODEL, contents=[f, PROMPT])
            parsed = parse_response(resp.text or "", duration)
            try:
                client.files.delete(name=f.name)
            except Exception:
                pass
            if parsed is None:
                raise ValueError(f"could not parse/validate response: {resp.text!r}")
            return parsed
        except genai_errors.ClientError as e:
            last_err = e
            if getattr(e, "code", None) == 429:
                wait = min(60, 5 * attempt)
                print(f"    rate limited, waiting {wait}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                print(f"    client error (attempt {attempt}/{MAX_RETRIES}): {e}")
                time.sleep(3)
        except Exception as e:
            last_err = e
            print(f"    error (attempt {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(3)
    return {"error": str(last_err)}


def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    results = load_results()

    clips = sorted(glob.glob(os.path.join(RAW_DIR, "*.mp4")))
    print(f"Found {len(clips)} Fall clips, {len(results)} already labeled")

    for i, path in enumerate(clips, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in results and "error" not in results[name]:
            continue
        duration = video_duration_s(path)
        print(f"[{i}/{len(clips)}] {name} (duration {duration:.1f}s)")
        result = label_one(client, path, duration)
        result["duration_s"] = duration
        results[name] = result
        save_results(results)
        if "error" in result:
            print(f"    FAILED: {result['error']}")
        else:
            print(f"    onset={result['fall_onset_s']}s confidence={result['confidence']}")

    ok = sum(1 for r in results.values() if "error" not in r)
    failed = [n for n, r in results.items() if "error" in r]
    print(f"\nDone. {ok}/{len(clips)} labeled successfully.")
    if failed:
        print(f"Failed ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
