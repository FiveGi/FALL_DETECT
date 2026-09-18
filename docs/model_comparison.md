# What is deployed, and what each change bought

Two separate things have been measured on this system and they are easy to confuse: the
**inference settings** (input size, pose confidence, smoothing) and the **trained model**.
What follows first is the current state, where both have changed; the settings comparison,
which holds the model fixed, is kept below it as history.

## Deployed now

| piece | value |
|---|---|
| pose backbone | `yolo26s-pose`, input 960, pose confidence 0.30 |
| classifier | `models/fall_classifier_v3.onnx` (md5 `194614047877dc8e9ff896e5331170f7`) |
| trained from | `yolopose_aug_seed42.pt` -- SS35's recipe |
| alerting | one positive window out of the last three |
| alert wording | never decided by the score; see "The alert tier" below |

Accuracy on the dataset nothing here has been tuned against (**URFD**, 60 fall clips and 40
normal-activity clips, none of them used in training) alongside GMDCSA24, which has been used
for tuning repeatedly and therefore reads high:

| what is measured | deployed |
|---|---|
| URFD falls caught | **43/60 (72%)** |
| URFD normal clips with no false alarm | **27/40 (68%)** |
| GMDCSA24 val falls | 15/15 |
| GMDCSA24 val ADL clean | 10/16 |
| GMDCSA24 train50 falls | 24/25 |
| GMDCSA24 train50 ADL clean | 20/25 |
| two people, one falls | 13/15 |
| two people, nobody falls | 4/4 |

**Quote the URFD number.** GMDCSA24 overstates this system by roughly 25 points of recall
because its clips have been used to pick thresholds, input size and pose confidence.

### A model change that did not survive a proper check

A model trained with half of URFD's normal-activity clips added as hard negatives looked
better on one seed (+1 fall, +1 clean clip) and was deployed. Re-run across three seeds with
and without that data, URFD recall came out **identical on average, 44.0 versus 44.0**, with
seed-to-seed spread of 1-3 clips on every column -- the same size as the reported gain. It was
reverted, because it cost half of the only untuned dataset for nothing measurable. Details and
the full table are in SKILL.md SS49.

The lesson generalises: on this system a difference of one or two clips between two trained
models is noise, and any model claim needs three seeds compared by mean.

## The alert tier

Alerts used to be worded two ways, split at a confidence of 0.85: above it the message stated a
fall, below it asked someone to look. Re-measured on the quantity the system actually uses --
the score at the instant the alert fires, across 154 alerts on four labelled surfaces
(`training/measure_alert_tier.py`) -- that split does not separate anything:

| bar | real-fall alerts above it | false alarms above it | precision above it |
|---|---|---|---|
| 0.50 | 100% | 100% | 85% |
| 0.70 | 46% | 26% | 91% |
| 0.85 | 10% | 9% | 87% |

At the bar the system used, a genuine-fall alert and a false alarm are about equally likely to
clear it. The reason written in the code -- "nothing above 0.85 was a false alarm" -- is false:
a man getting up from a bed (`s4_ADL_08`) alerts at 0.88. And the bar demoted 90% of genuine
falls to "please check" anyway.

Every fall alert now asks a human to look. The urgent wording is reserved for an alert nobody
acknowledged, which is a fact about the response rather than a guess about the footage.

---

# Old vs new inference settings (historical)

This half holds the trained model fixed at the SS35 export and varies only the inference
configuration, both of which were previously unmeasured library defaults. It predates the model
change above; regenerate the equivalent for the current model with
`python training/compare_old_vs_new.py`, which writes `docs/settings_comparison.md`.

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

Reproduce with `python training/compare_old_vs_new.py` (writes `settings_comparison.md`; needs the GMDCSA24 clips
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

## Verified again as CLIPS, not stills

A fall is a motion. A single frame cannot tell a fall from someone who lay down, and it
cannot tell a real fall caught mid-motion from ordinary walking -- so the still-based numbers
above are measuring the wrong thing in both directions. Redone with
`training/verify_alerts_with_clips.py`: each alert becomes a short video around the moment,
Gemini judges the motion, and a third verdict is available that a still cannot express --
ALREADY_DOWN, meaning someone is on the ground the whole time without falling.

| | old (640 / 0.50) | new (960 / 0.30) |
|---|---|---|
| alerts | 59 | 70 |
| a real fall happens | 53 | **60** |
| already on the ground, no fall in the clip | 1 | 3 |
| nobody falls | 5 | 6 |
| unusable (shot shorter than 1s) | 0 | 1 |
| precision | 89.8% | 87.0% |

**The new settings catch 7 more real falls.** Precision moves from
89.8% to 87.0% -- slightly
down, because the extra alerts include two more ALREADY_DOWN and one more NOT_A_FALL. Alerting
on a person who is already on the floor is not useless (they may still need help) but it is
not fall detection, so it is counted separately rather than folded into either column.

Both columns score *higher* here than the still-based pass (90% and
86% versus 78% and 81%), because several alerts a still called "not a fall"
are genuine falls caught mid-motion, where one frame looks like walking.

### One methodological trap worth knowing about

The first clip-based attempt returned "FALL" for 68 of 70 alerts, including an alert on an
empty night-time deck that had already been confirmed by eye to contain no person. The cause
is the source material: the `Test/` clips are social-media compilations that cut between
unrelated incidents every few seconds, so a 7-second window around an alert often contained
two or three different scenes -- and someone else's fall. `verify_alerts_with_clips.py` now
detects hard cuts and trims each window to the shot the alert is actually in; the empty-deck
alert then correctly comes back NOT_A_FALL. **Any future clip-based verification on
compilation footage has to do this, or it measures the wrong scene.**
