"""Have Gemini independently count the actual number of people in each sampled
frame -- the ground truth to score each YOLO variant's person-count against."""
import os
import glob
import json
import time
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from google import genai

FRAMES_DIR = os.path.join(os.path.dirname(__file__), "data", "gt_frames")
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "gt_gemini_counts.json")

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
prompt = (
    "Count the number of distinct real human people visible in this image (partially "
    "visible/cut-off people count if clearly a person; do not count photos, posters, "
    "reflections, or statues). Answer ONLY with a single integer, nothing else."
)

frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
results = {}
for fp in frame_paths:
    name = os.path.basename(fp)
    for attempt in range(4):
        try:
            f = client.files.upload(file=fp)
            resp = client.models.generate_content(model="gemini-flash-lite-latest", contents=[f, prompt])
            text = resp.text.strip()
            digits = "".join(c for c in text if c.isdigit())
            count = int(digits) if digits else -1
            results[name] = count
            print(name, "->", count, f"(raw: {text!r})" if count == -1 else "")
            break
        except Exception as e:
            print(name, f"attempt {attempt} failed:", str(e)[:80])
            time.sleep(6)
    else:
        results[name] = -1

json.dump(results, open(OUT_PATH, "w"), indent=2)
print(f"\nDone. Saved to {OUT_PATH}")
