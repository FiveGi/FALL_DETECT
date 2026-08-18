import os
import json
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

pool = json.load(open("data/oops_screen_pool.json"))
results = {}
prompt = (
    "This still is from a video used to test a fall-detection system meant for "
    "monitoring ELDERLY people INDOORS (home/care facility). Rate how relevant this "
    "scene is to that use case. Answer ONLY compact JSON: "
    '{"setting": "indoor or outdoor", "activity": "short phrase", '
    '"elderly_relevant": true or false, "reason": "one short phrase"} '
    "elderly_relevant should be true only for calm, everyday, home/domestic-like "
    "activity (walking, standing, sitting, bending, household tasks) -- false for "
    "sports, stunts, kids playing rough, vehicles, water activities, or anything "
    "clearly not home-monitoring-like."
)

for i, c in enumerate(pool):
    img_path = f"data/oops_screen_frames/{i:02d}.jpg"
    if not os.path.exists(img_path):
        continue
    f = client.files.upload(file=img_path)
    try:
        resp = client.models.generate_content(model="gemini-flash-lite-latest", contents=[f, prompt])
        results[str(i)] = resp.text
    except Exception as e:
        results[str(i)] = f"ERROR: {e}"
    print(i, c["stem"][:40], "->", results[str(i)].replace("\n", " "))

json.dump(results, open("data/oops_screen_results.json", "w"), indent=2)
