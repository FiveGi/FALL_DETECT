import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

times = [61.0, 62.5, 40.0, 45.0, 44.0, 52.5, 54.5, 60.5, 42.5, 20.0]
prompt = (
    "A pose-detection system found a human pose somewhere in this frame with LOW "
    "confidence. Is there an actual real person visible anywhere in this frame, or "
    "is this frame showing objects/scenery with no real person (which would mean "
    "the detection was a false hallucination)? Answer ONLY compact JSON: "
    '{"real_person_present": true or false, "what_is_in_frame": "short phrase"}'
)

for t in times:
    path = f"data/lowvis_{t}.jpg"
    if not os.path.exists(path):
        print(t, "MISSING FILE")
        continue
    f = client.files.upload(file=path)
    resp = client.models.generate_content(model="gemini-flash-lite-latest", contents=[f, prompt])
    print(f"t={t} ->", resp.text.replace("\n", " "))
