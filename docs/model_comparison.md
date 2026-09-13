# Old vs new detection settings

Same trained model in both columns -- `models/fall_classifier_v3.onnx`
(md5 `194614047877dc8e9ff896e5331170f7`) on the `yolo26s-pose` backbone. What
changed is the inference configuration, both of which were previously unmeasured
library defaults:

| setting | old | new |
|---|---|---|
| pose input size (`V3_IMGSZ`) | 640 | **960** |
| pose confidence (`V3_POSE_CONF`) | 0.50 | **0.30** |

Every number below comes from the multi-person entry point
(`detect_v3_fall_multi`), the one the camera loop actually runs.

## Results

| what is measured | old | new |
|---|---|---|
| GMDCSA24 val - falls caught | 15/15 (100%) | 15/15 (100%) |
| GMDCSA24 val - normal clips with no false alarm | 9/16 (56%) | 10/16 (62%) |
| train50 - falls caught | 22/25 (88%) | 24/25 (96%) |
| train50 - normal clips with no false alarm | 22/25 (88%) | 22/25 (88%) |
| two people, one falls - fall caught | 12/15 (80%) | 13/15 (87%) |
| two people, nobody falls - no false alarm | 4/4 (100%) | 4/4 (100%) |
| small-person control - fall caught | 10/15 (67%) | 14/15 (93%) |
| speed on GPU | 30.5 ms/frame (32.7 fps) | 37.5 ms/frame (26.6 fps) |

## Reading this

A camera needs about 25 fps to be followed without dropping frames, so both
columns are fast enough on the GPU and the new column costs a few ms.

The two-person and control rows exist because every other dataset here has
exactly one person in frame, which is not the situation this system is for.
The control is the same fall in the same widened frame with *nobody* beside it:
it separates "a bystander broke detection" from "the person is now rendered at
half the pixels", which the composite changes at the same time.

Reproduce with `python training/compare_old_vs_new.py` (needs the GMDCSA24 clips
in `training/data/` and the composites from
`python training/make_multiperson_testset.py`).

## Gemini frame-by-frame verification

Dataset labels only say whether a clip contains a fall. They cannot say whether the system
fired at the right moment, so every alert from both settings was re-extracted as a fresh
frame from source and judged one by one by Gemini (`training/verify_production_alerts.py`,
summarised by `training/summarize_gemini_verification.py`), across all 17 `Test/` clips.

| | old (640 / 0.50) | new (960 / 0.30) |
|---|---|---|
| alerts raised | 59 | 70 |
| Gemini says a real fall | 46 | **57** |
| Gemini says not a fall | 13 | **13** |
| verified precision | 78.0% | **81.4%** |

**The 11 extra alerts are 11 extra real falls.** The false-alarm count does not move at all,
which is the result an alert count on its own could never have shown: more alerts here means
more catches, not more noise.

Two frames were also read directly rather than taken on Gemini's word. One of the alerts only
the new settings raise is an elderly person on the ground beside a walker with a second person
bending over them -- two people in frame, the exact scenario this system exists for, and the
old settings missed it. One of the false alarms is an empty deck at night with no person in
frame at all, matching the night-vision/furniture pattern already documented in SKILL.md.

### What the remaining 13 false alarms are

- **sitting, leaning, standing still** (6): Person is standing and adjusting a purse; person standing and using a phone on porch; sitting or rocking on a porch swing; woman is standing and interacting with pets; person leaning over the side of a boat to reach the water; Men hanging from a horizontal bar
- **walking, including on stairs** (4): Person is walking briskly away from the camera; man is stepping down stairs while carrying cups and woman is walking normally; person is walking down the stairs upright; child with backpack walking away from open door
- **no person in frame (night / empty scene)** (3): darkness with text overlay, no visible person or fall event; surveillance footage of an empty deck at night; view of stairs and doorway

All of them are outdoor doorbell footage, which is not the indoor care setting this system
targets -- but the categories are the same ones SKILL.md has been tracking since SS20, and the
first group is the clearest lead: alerts fired on frames with no person in them at all.
