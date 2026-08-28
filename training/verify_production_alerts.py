"""Gemini-verify every alert from the newly-deployed YOLO-pose+augmentation production
model, both single- and multi-person paths, across all Test/ clips -- mirrors
verify_current_model_alerts.py's methodology (re-extracts a fresh frame per alert
straight from source, doesn't trust stale alert_frames/)."""
import os
import sys
import json
import time
import cv2
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from google import genai

TEST_DIR = r"D:\project\PROJECT\Test"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FRAMES_DIR = os.path.join(DATA_DIR, "verify_frames_prod")
os.makedirs(FRAMES_DIR, exist_ok=True)

which = sys.argv[1] if len(sys.argv) > 1 else "single"
alerts_by_clip = json.load(open(os.path.join(DATA_DIR, f"prod_{which}_alerts.json")))

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
prompt = (
    "This still is from a fall-detection system that just flagged 'fall' at this "
    "moment (raw frame, no overlay). Is the person actually falling or fallen, or "
    "upright/normal/doing something else? Answer ONLY compact JSON: "
    '{"verdict": "FALL or NOT_A_FALL", "reason": "short phrase"}'
)

results = {}
for clip_id, alerts in sorted(alerts_by_clip.items(), key=lambda x: int(x[0])):
    if not alerts:
        continue
    video_path = os.path.join(TEST_DIR, f"{clip_id}.mp4")
    cap = cv2.VideoCapture(video_path)
    for t, p in alerts:
        key = f"{clip_id}/t={t:.1f}s_p={p:.2f}"
        frame_path = os.path.join(FRAMES_DIR, f"{which}_{clip_id}_{t:.1f}.jpg")
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            print(key, "-> FRAME READ FAILED")
            continue
        cv2.imwrite(frame_path, frame)

        for attempt in range(4):
            try:
                f = client.files.upload(file=frame_path)
                resp = client.models.generate_content(model="gemini-flash-lite-latest", contents=[f, prompt])
                results[key] = resp.text
                print(key, "->", resp.text.replace("\n", " "))
                break
            except Exception as e:
                print(key, f"attempt {attempt} failed:", str(e)[:80])
                time.sleep(6)
        else:
            results[key] = "ERROR: all retries failed"
    cap.release()

json.dump(results, open(os.path.join(DATA_DIR, f"prod_{which}_verify_results.json"), "w"), indent=2)
print(f"\nDone. {len(results)} checked ({which}).")
