import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

frames = [
    ("data/gmdcsa24_fp_s1_ADL_01.jpg", "s1_ADL_01 t=3.7s"),
    ("data/gmdcsa24_fp_s4_ADL_08.jpg", "s4_ADL_08 t=1.3s"),
    ("data/gmdcsa24_fp_s2_ADL_15.jpg", "s2_ADL_15 t=7.9s"),
]

prompt = (
    "A fall-detection system just flagged 'fall' at this moment (no overlay on this "
    "image, just the raw frame). Look at the person's pose and surroundings. Is this "
    "person actually falling/fallen, or are they on a bed doing something else "
    "(lying down, sitting up, adjusting position)? Answer in 1-2 sentences: verdict "
    "(FALL or NOT A FALL) and what they're actually doing."
)

for path, label in frames:
    f = client.files.upload(file=path)
    resp = client.models.generate_content(model="gemini-flash-lite-latest", contents=[f, prompt])
    print(f"--- {label} ---")
    print(resp.text)
    print()
