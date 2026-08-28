import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

prompt = """You are advising on an elderly fall-detection system (Flask/Celery backend,
runs on CPU only, no GPU). Pipeline: MediaPipe BlazePose extracts 17 keypoints per
frame -> torso-relative normalize + per-frame velocity -> 30-frame (1s @ 30fps) window
-> small temporal CNN classifier outputs fall probability.

PROBLEM: the classifier confuses "person lying still on a bed" (should NOT alert) with
"person who fell and is now lying still on the floor" (SHOULD alert), because pose
keypoints alone carry no information about what surface is under the person - both look
geometrically identical once the person stops moving. This causes false alarms on 5-6 of
16 held-out ADL (activities of daily living) clips in our GMDCSA24 test set, all of them
"lying/sitting on bed" scenes.

We already ruled out (via 9 separate retraining runs across 3 seeds): more training data,
different loss class-weighting, and bigger model capacity - none change this pattern,
because it's a representational limitation, not a data/capacity problem.

We then tried adding scene *context* using an already-deployed YOLO object detector
(yolov10x, which recognizes bed/couch/chair COCO classes) as an external signal, tested
against our actual val-set clips with real bounding boxes:

  1. "Is a bed anywhere in frame" -> FAILS: 14 of 15 genuine GMDCSA24 fall clips also have
     a bed visible in the room throughout (bedroom setting), even though the person falls
     on the floor away from it. This would suppress ~93% of genuine falls.

  2. "Does the person's bounding box overlap a bed box" (fraction of person-box area
     inside bed-box) -> PARTIALLY helps (would fix 5/6 false positives, overlap 0.23-0.83)
     but 9 of 15 genuine fall clips ALSO show high person-bed overlap at the end (person
     falls onto/near/against the bed, which is realistic - e.g. falling while getting out
     of bed), with overlap 0.61-1.00. The two distributions overlap heavily (0.61-0.83
     zone contains both classes) - no clean threshold separates them.

  3. "Was there a recent high keypoint-velocity spike before the still moment" (idea:
     genuine falls have a sudden impact, calm bed-lying doesn't) -> ALSO FAILS: measured
     max velocity magnitude (same feature already in the classifier) in the ADL false
     positives ranges 0.06-2.20, and in the "near-bed" genuine falls ranges 0.16-7.54.
     Heavy overlap again - e.g. one false-positive ADL clip peaks at 2.20 (higher than
     5 of the 10 genuine near-bed falls), because ordinary bed-settling motion (adjusting
     position, sitting down) and MediaPipe's own per-frame jitter produce velocity
     magnitudes comparable to some real (but not dramatically forceful) falls in this
     dataset.

All three geometric/kinematic signals we could extract failed to cleanly separate the two
classes. Given this evidence, what would you actually try next? Consider: is there a
different, cheap signal we haven't tried; is a small amount of targeted manual review
more practical than more automation here; is there a lightweight way to bring in actual
pixel/appearance information (not just keypoints) without requiring GPU/heavy compute;
or is accepting this false-positive rate (with alerting still biased toward high recall)
the right tradeoff for a CPU-only elderly-care deployment. Be concrete and skeptical -
don't just suggest "collect more data" or "use a bigger model," we already ruled those out
with real experiments. 250 words max.
"""

resp = client.models.generate_content(model="gemini-flash-lite-latest", contents=[prompt])
print(resp.text)
