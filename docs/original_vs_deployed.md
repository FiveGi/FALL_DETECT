# Detector configurations compared

Every column is the **whole detector**, not a weight file: the original column is
this repository's first commit, with its own pose extractor, window and alerting
rule. Each is fed the same clips at the frame rate it would actually run at,
because the window is a fixed number of frames and the rate decides how much real
time it covers — every number published here before that was understood came from
reading every frame of a file, which a live camera never does.

|  | original @15fps | original @30fps | deployed | 30-frame @24fps |
| --- | --- | --- | --- | --- |
| pose extractor | mediapipe | mediapipe | yolo26s-pose, input 960 | yolo26s-pose, input 960 |
| window | 30 frames | 30 frames | 15 frames | 30 frames |
| threshold | 0.5 | 0.5 | 0.65 | 0.5 |
| alerting rule | 2 of 3 windows | 2 of 3 windows | 1 of 3 windows | 1 of 3 windows |
| fed at | 15 fps | 30 fps | 15 fps | 24 fps |

## Results on the lab datasets

| what is measured | original @15fps | original @30fps | deployed | 30-frame @24fps |
| --- | --- | --- | --- | --- |
| URFD — falls caught | 27/60 | 54/60 | 41/60 | 39/60 |
| URFD — normal clips with no false alarm | 25/40 | 24/40 | 34/40 | 30/40 |
| GMDCSA24 val — normal clips with no false alarm | 10/16 | 9/16 | 7/16 | 11/16 |
| GMDCSA24 — falls caught (mostly training clips) | 75/79 | 76/79 | 69/79 | 78/79 |
| GMDCSA24 train50 — normal clips clean (training clips) | 21/25 | 20/25 | 21/25 | 20/25 |
| **falls caught, data never trained on** | **27/60** (45%) | **54/60** (90%) | **41/60** (68%) | **39/60** (65%) |
| **clean, data never trained on** | **35/56** (62%) | **33/56** (59%) | **41/56** (73%) | **41/56** (73%) |

### How many URFD falls each column can score at all

The classifier cannot produce a number until its window holds a full set of frames
with a person in them, and some URFD clips end before that happens. The ceiling
camera on the standing falls is the clear case: the room is empty for two thirds of
the clip and the person walks into view as they land, so at 15 fps there are 43
sampled frames, a person in 14 of them, and a window that needs 15. Those clips
score exactly 0.00 — not a fall the model rejected, a fall it was never shown.

A longer window at a lower rate needs more real time of visible person before it can
say anything at all, so this is part of what the recall column is measuring.

| column | window must cover | falls it cannot score | recall over the rest |
| --- | --- | --- | --- |
| original @15fps | 2.00 s | 24/60 | 25/36 (69%) |
| original @30fps | 1.00 s | 7/60 | 49/53 (92%) |
| deployed | 1.00 s | 7/60 | 41/53 (77%) |
| 30-frame @24fps | 1.25 s | 9/60 | 37/51 (73%) |

### URFD by fall type

URFD recorded two kinds of fall and alternates them by sequence number. They are
not equally hard, and reporting only the total hides which one a change helped.
Falls from standing are also the ones the ceiling camera barely sees: the room is
empty for two thirds of those clips and the person walks into view as they land.

| fall type | original @15fps | original @30fps | deployed | 30-frame @24fps |
| --- | --- | --- | --- | --- |
| from standing (odd sequences) | 15/30 | 25/30 | 13/30 | 14/30 |
| out of a chair (even sequences) | 12/30 | 29/30 | 28/30 | 25/30 |

### URFD split in half

Half A is what a decision may be made on; half B only confirms it. A configuration
that wins on A but not on B won on noise.

The split takes sequences two at a time rather than by odd and even, because URFD
alternates its fall types: odd-numbered sequences are falls from standing and
even-numbered ones are falls out of a chair, so an odd/even split compares two
different tasks rather than two samples of one.

| half | original @15fps | original @30fps | deployed | 30-frame @24fps |
| --- | --- | --- | --- | --- |
| A — may choose on, falls caught | 13/32 | 27/32 | 22/32 | 22/32 |
| A — may choose on, clean | 12/20 | 13/20 | 18/20 | 16/20 |
| B — confirms only, falls caught | 14/28 | 27/28 | 19/28 | 17/28 |
| B — confirms only, clean | 13/20 | 11/20 | 16/20 | 14/20 |

## Results on the Test/ clips

These are the only footage here that is not a lab dataset. `13`-`16` are real falls,
one each, so a column is simply caught or missed. `17` has no fall in it at all — a
crowd doing an outdoor exercise routine — so the right answer there is silence, and
it is scored that way rather than counted as a miss. `1`-`12` are compilations cut
from social media with several incidents and long stretches of ordinary activity
between them, so an alert count there is neither right nor wrong on its own and is
reported as a count.

### Real falls (`Test/13`-`16`), and one clip that must stay silent

| configuration | 13.mp4 | 14.mp4 | 15.mp4 | 16.mp4 | caught | 17.mp4 (no fall) |
| --- | --- | --- | --- | --- | --- | --- |
| original MediaPipe 30-frame 0.50 @15fps | missed 0.46 | **caught** 0.98 | **caught** 0.89 | **caught** 0.98 | 3/4 | silent 0.68 |
| original MediaPipe 30-frame 0.50 @30fps | missed 0.56 | **caught** 0.98 | **caught** 0.91 | **caught** 0.99 | 3/4 | **false alarm** 0.69 |
| deployed 15-frame 0.65 @15fps | missed 0.18 | **caught** 0.88 | **caught** 0.74 | **caught** 0.88 | 3/4 | silent 0.38 |
| previous 30-frame 0.50 @15fps | missed 0.38 | **caught** 0.91 | **caught** 0.83 | **caught** 0.85 | 3/4 | silent 0.38 |
| previous 30-frame 0.50 @25fps | **caught** 0.62 | **caught** 0.93 | **caught** 0.83 | **caught** 0.85 | 4/4 | silent 0.40 |
| previous 30-frame 0.50 @24fps | **caught** 0.62 | **caught** 0.93 | **caught** 0.83 | **caught** 0.85 | 4/4 | silent 0.40 |

The number after each verdict is the highest score that configuration reached on
the clip, because missed-by-a-little and never-came-close are different problems.

### Compilations (`Test/1`-`12`) — alerts raised, not a score

| configuration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original MediaPipe 30-frame 0.50 @15fps | 8 | 3 | 2 | 11 | 7 | 1 | 3 | 7 | 7 | 3 | 1 | 10 | 63 |
| original MediaPipe 30-frame 0.50 @30fps | 14 | 6 | 4 | 20 | 11 | 1 | 6 | 9 | 15 | 6 | 2 | 13 | 107 |
| deployed 15-frame 0.65 @15fps | 12 | 1 | 2 | 7 | 15 | 1 | 4 | 5 | 7 | 0 | 0 | 7 | 61 |
| previous 30-frame 0.50 @15fps | 9 | 3 | 2 | 7 | 10 | 1 | 3 | 5 | 5 | 0 | 0 | 4 | 49 |
| previous 30-frame 0.50 @25fps | 12 | 5 | 2 | 7 | 10 | 0 | 5 | 6 | 8 | 0 | 0 | 9 | 64 |
| previous 30-frame 0.50 @24fps | 12 | 4 | 2 | 6 | 10 | 1 | 5 | 6 | 8 | 0 | 0 | 7 | 61 |

## What changed, clip by clip

**original @30fps** against original @15fps:

- falls original @15fps missed and this catches: **30**
- falls original @15fps caught and this misses: **2** (fall-09-cam0.mp4, s2_Fall_20.mp4)
- false alarms removed: **1**
- false alarms introduced: **4** (adl-11-cam0.mp4, adl-20-cam0.mp4, s3_ADL_04.mp4, s4_ADL_08.mp4)

**deployed** against original @15fps:

- falls original @15fps missed and this catches: **18**
- falls original @15fps caught and this misses: **10** (fall-09-cam0.mp4, fall-19-cam0.mp4, fall-29-cam0.mp4, s1_Fall_14.mp4, s3_Fall_15.mp4, s4_Fall_04.mp4, s4_Fall_09.mp4, s4_Fall_10.mp4, s4_Fall_11.mp4, s4_Fall_16.mp4)
- false alarms removed: **12**
- false alarms introduced: **6** (adl-11-cam0.mp4, s2_ADL_13.mp4, s2_ADL_16.mp4, s2_ADL_20.mp4, s3_ADL_13.mp4, s4_ADL_08.mp4)

**30-frame @24fps** against original @15fps:

- falls original @15fps missed and this catches: **20**
- falls original @15fps caught and this misses: **5** (fall-04-cam0.mp4, fall-19-cam0.mp4, fall-20-cam0.mp4, fall-29-cam0.mp4, s4_Fall_09.mp4)
- false alarms removed: **11**
- false alarms introduced: **6** (adl-11-cam0.mp4, adl-38-cam0.mp4, s2_ADL_08.mp4, s2_ADL_16.mp4, s3_ADL_20.mp4, s4_ADL_08.mp4)

## What this adds up to

The deployed detector is the right one to run **on this machine**, and the reason is
not that its model is better everywhere.

**Fed every frame, the original is better on URFD than anything deployed since** —
54/60 falls (90%) against 41/60 (68%). That is not a trick of its alerting rule: its
"the person vanished" collapse rule was switched off and re-run, and URFD recall was
**54/60 either way**, so the recall is the classifier. Nor is it a threshold artefact:
at a threshold loose enough to give a *better* clean rate than the original (0.50,
36/56 clean against 33/56), the deployed model still catches only 43/60.

**It cannot be fed every frame.** Timed on the same 1080p frames on an idle machine,
the original needs **70.3 ms per frame (14.2 fps)** against the deployed detector’s
**54.4 ms (18.4 fps)** — MediaPipe on the CPU is slower than YOLO26s-pose on the GPU,
before decoding or anything else in the loop. 30 fps was never available to it. At the
rate it can actually sustain, its own numbers are **27/60 (45%)**.

So the honest summary of the rewrite is: **the system now works at the speed it
really runs**, which is worth +14 falls and +6 clean clips at 15 fps, not that the
model learned more. The 90% figure is what a faster machine would be worth, and it is
the strongest argument in this document for spending effort on frame rate rather than
on the classifier.

The 30-frame model at 24 fps was measured to settle whether to move back to it. It is
behind on URFD falls (39/60), behind on URFD clean (30/40), behind or level on both
halves of the split, and has more clips it cannot score (9 against 7). It wins one
real-footage clip, `Test/13` — a moving, zooming camera on a young adult who catches
himself on his hands, measured at 4.8 px/frame of background motion against 0.03 for
the fixed-camera clips. Nothing in this pipeline is built for a moving camera, and
seed variance alone moves one to three clips per surface. **The deployed
configuration stays.**

Reproduce the columns with the runners named in SKILL.md SS55 and SS60, then
regenerate this file with
`python training/compare_original_vs_deployed.py "name=a.json" "name=b.json" ...`.
