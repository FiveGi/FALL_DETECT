import os
import glob
import json
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

RESULTS_ROOT = os.environ.get("TEST_RESULTS_ROOT", r"D:\project\PROJECT\Test_Results")
CLIPS = ["4", "5", "6", "7", "8", "9", "10", "11"]

prompt = (
    "This still is from a fall-detection system that just flagged 'fall' at this "
    "moment (ignore the red border / text overlay, look at the body pose). Is the "
    "person actually falling or fallen, or upright/normal? Answer ONLY compact JSON: "
    '{"verdict": "FALL or NOT_A_FALL", "reason": "short phrase"}'
)

results = {}
for clip in CLIPS:
    frame_dir = os.path.join(RESULTS_ROOT, clip, "alert_frames")
    if not os.path.isdir(frame_dir):
        continue
    for path in sorted(glob.glob(os.path.join(frame_dir, "*.jpg"))):
        key = f"{clip}/{os.path.basename(path)}"
        try:
            f = client.files.upload(file=path)
            resp = client.models.generate_content(model="gemini-flash-lite-latest", contents=[f, prompt])
            results[key] = resp.text
        except Exception as e:
            results[key] = f"ERROR: {e}"
        print(key, "->", results[key].replace("\n", " "))

with open(os.path.join(os.path.dirname(__file__), "data", "gemini_screen_new_clips.json"), "w") as f:
    json.dump(results, f, indent=2)
