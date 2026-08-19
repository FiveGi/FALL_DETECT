import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

frames = [
    ("data/train50_fp_s3_ADL_02_a.jpg", "s3_ADL_02 t=1.3s"),
    ("data/train50_fp_s3_ADL_02_b.jpg", "s3_ADL_02 t=5.3s"),
    ("data/train50_fp_s4_ADL_16.jpg", "s4_ADL_16 t=5.3s"),
    ("data/train50_fp_s4_ADL_15.jpg", "s4_ADL_15 t=3.7s"),
    ("data/train50_fp_s3_ADL_09.jpg", "s3_ADL_09 t=1.6s"),
    ("data/train50_fp_s3_ADL_04.jpg", "s3_ADL_04 t=6.0s"),
]

prompt = (
    "A fall-detection system just flagged 'fall' at this moment (raw frame, no "
    "overlay). Is the person actually falling/fallen, or doing something else "
    "(e.g. lying/sitting on a bed, bending down, standing)? Answer in 1 sentence: "
    "verdict (FALL or NOT_A_FALL) and what they're doing."
)

for path, label in frames:
    f = client.files.upload(file=path)
    resp = client.models.generate_content(model="gemini-flash-lite-latest", contents=[f, prompt])
    print(f"--- {label} ---")
    print(resp.text)
    print()
