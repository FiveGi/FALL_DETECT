import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

prompt = """Advising on an elderly fall-detection system (CPU-only). We just tested swapping the pose
keypoint extractor from MediaPipe BlazePose to YOLO-pose (yolo26s-pose), re-extracting keypoints for our
~4,250 video-derived training clips (GMDCSA24, CAUCAFall, OOPS/OF-ItW) and retraining the same small
temporal-CNN fall classifier from scratch, 5 random seeds, identical hyperparameters to our currently
deployed MediaPipe-based model.

Results: on our standard held-out test set (GMDCSA24, 31+50 clips), YOLO-pose essentially MATCHES the
deployed model (100% recall / ~60% ADL-clean vs deployed's 100%/62.5%). Raw pose-detection rate on
hard "person lying on ground" frames is even BETTER than MediaPipe (81% vs 72%) and ~40% faster on CPU.

But on 3 real elderly-fall video clips we independently collected (not in training data) that the deployed
MediaPipe model catches solidly (confidence 0.85, 0.64, 0.93):
  - clip A: caught by only 1 of 5 seeds, and barely (0.50 vs the 0.5 threshold)
  - clip B: caught by 2 of 5 seeds (0.60-0.68)
  - clip C: caught by all 5 seeds solidly (0.71-0.96)

We tried averaging all 5 seeds' output probabilities (ensembling) - made it WORSE, not better (clip A/B
missed even more). We tried lowering the alert threshold from 0.5 to 0.4 - barely moved the numbers, still
missed clip A. We confirmed clip A isn't a pose-detection failure (YOLO-pose finds the person in 100% of
its frames) - the classifier itself just never builds confidence, peaking at 0.44 during an unambiguous,
textbook fall (elderly woman collapsed on floor, mobility cane fallen beside her, visible distress).

Our working theory: clip A's specific fall pattern (using a cane, more torso rotation during the fall,
brief self-occlusion mid-collapse) is under-represented in our ~4,250 training clips, and this is a
training-data-coverage gap rather than something ensembling/threshold-tuning can fix.

Given this evidence, do you agree with that diagnosis, or is there a cheaper experiment we're missing before
concluding we need new targeted training data (e.g. clips of falls involving canes/walkers/mobility aids)?
Also: is it plausible that YOLO-pose's keypoint values are subtly systematically different from MediaPipe's
in a way that would specifically hurt on rotation/self-occlusion (not just detection presence, which we've
ruled out), and if so is there a quick diagnostic for that we haven't tried? Be concrete and skeptical.
200 words max.
"""

resp = client.models.generate_content(model="gemini-flash-lite-latest", contents=[prompt])
print(resp.text)
