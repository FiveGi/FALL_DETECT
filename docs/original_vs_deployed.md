# The original detector versus the deployed one

Both columns are the **whole detector**, not two weight files: the original is this
repository's first commit (ORIGINAL ce401fa) with its own pose extractor,
window and alerting rule. Both are fed the same clips at the same frame rate, because
the window is a fixed number of frames and the rate decides how much real time it
covers — every number this project published before that was understood came from
reading every frame of a file, which a live camera never does.

| | original | deployed |
|---|---|---|
| pose extractor | mediapipe | yolo26s-pose, input 960 |
| window | 30 frames | 15 frames |
| threshold | 0.5 | 0.65 |
| alerting rule | 2 of 3 windows | 1 of 3 windows |
| fed at | 15 fps | 15 fps |

## Results

| what is measured | original | deployed | change |
|---|---|---|---|
| URFD — falls caught | 27/60 | 41/60 | **+14** |
| URFD — normal clips with no false alarm | 25/40 | 34/40 | **+9** |
| GMDCSA24 val — normal clips with no false alarm | 10/16 | 7/16 | **-3** |
| GMDCSA24 — falls caught (mostly training clips) | 75/79 | 69/79 | **-6** |
| GMDCSA24 train50 — normal clips clean (training clips) | 21/25 | 21/25 | 0 |
| **falls caught, data never trained on** | **27/60** (45%) | **41/60** (68%) | **+14** |
| **clean, data never trained on** | **35/56** (62%) | **41/56** (73%) | **+6** |

## What changed, clip by clip

- falls the original missed and the deployed detector catches: **18**
- falls the original caught and the deployed detector misses: **10** (fall-09-cam0.mp4, fall-19-cam0.mp4, fall-29-cam0.mp4, s1_Fall_14.mp4, s3_Fall_15.mp4, s4_Fall_04.mp4, s4_Fall_09.mp4, s4_Fall_10.mp4, s4_Fall_11.mp4, s4_Fall_16.mp4)
- false alarms removed: **12**
- false alarms introduced: **6** (adl-11-cam0.mp4, s2_ADL_13.mp4, s2_ADL_16.mp4, s2_ADL_20.mp4, s3_ADL_13.mp4, s4_ADL_08.mp4)

Reproduce both columns with the runners named in SKILL.md SS55, then regenerate this
file with `python training/compare_original_vs_deployed.py <original.json> <deployed.json>`.

## The same original detector, fed every frame

The table above feeds both detectors 15 fps, the rate `V3_TARGET_FPS` pins the camera loop to.
Fed every frame of the same files instead, the original changes beyond recognition:

| original, fed at | URFD falls caught | URFD clean |
|---|---|---|
| 30 fps (every frame of the file) | **54/60 (90%)** | 24/40 (60%) |
| 15 fps (what a camera here delivers) | 27/60 (45%) | 25/40 (62%) |

**Half its recall is frame rate alone.** Two things follow, and the second matters more than
the comparison above.

First, this confirms the reconstruction is faithful: run the way it was originally measured,
the original detector reproduces the strong recall that was reported for it.

Second, **the original design was not weak, it was starved of frames.** At 30 fps it catches
more falls than the deployed detector does at 15 (54 versus 41), though it pays for that with
far more false alarms (24/40 clean versus 34/40). Its 30-frame window spans 1.0s at 30 fps and
2.0s at 15 fps, and a fall stretched to twice its length stops looking like one.

So the honest reading of this document is not "the new detector is better". It is:

- at the frame rate this hardware actually sustains, the deployed detector is clearly better on
  both axes, and that is the situation the system is in;
- at 30 fps the original is better at catching falls, so **raising the achievable frame rate is
  worth more than any model change measured on this project so far** — pose extraction alone
  runs at ~71 fps on the GPU (14.1 ms/frame), and the loop achieves 16.8-20, so the gap is in
  the surrounding work rather than the model.
