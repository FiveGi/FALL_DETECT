# What is left to do, ordered for the CPU server

The deployment target is a four-core KVM VM with no GPU. This list is ordered by what that
machine gains, not by what is interesting. SKILL.md records *why* each item is here and what was
measured to get there; this file is what to pick up next.

Three measured facts frame everything (SKILL.md SS60, SS62, SS63):

- **A low-resolution camera substream is worth 22% of the frame rate, for free** (SS66) — the
  first thing to check on any install.
- **On CPU, speed and accuracy are the same thing.** At input size 320, URFD recall is 13/60 at
  6 fps, 32/60 at 8 fps, 34/60 at 9 fps. Every millisecond saved becomes recall.
- **The current architecture's best point has been found.** Seven input sizes and two pose
  models were measured, each paired with the rate it actually reaches. Going further means
  changing how it works, not what it is set to.
- **A second, separate loss: at 8 fps, 16 of 60 URFD falls cannot be scored at all** — the clip
  ends before the window holds enough frames with a person in it. At 15 fps it is 7. This is a
  multiplier on top of the frame-rate problem and it gets worse as the machine gets slower.

---

## Stopped mid-run, pick these up first

Two measurements were interrupted rather than finished. Both resume; neither needs a decision
first.

### A. The frame-position A/B (item 8 below)
**3 of 6 training runs done. About 35 minutes of GPU to finish, then 10 minutes to decide.**

`USE_FRAME_POSITION` / `V3_FRAME_POSITION` are committed and **both default to off**, so the
deployed detector is untouched. The feature construction is verified — training and runtime
produce bit-identical windows, the flip augmentation leaves the two new channels alone, and a
model that disagrees with the flag is refused rather than scored.

Done: `fp0_seed42`, `fp0_seed7`, `fp0_seed123` (val F1 0.588 / 0.588 / 0.598).
Left: the three `fp1` seeds, plus two control runs with `SKIP_POSE_DIRS=poses_fallvision`.

```
bash <scratchpad>/fp/train_all.sh        # skips runs whose ONNX already exists
bash <scratchpad>/fp/train_controls.sh   # the COORD_SCALE control
bash <scratchpad>/fp/sweep_all.sh        # threshold curve per model, ~12s each
python training/measure/pick_threshold.py "baseline=...fp0*" "frame position=...fp1*"
```

**Decide with `pick_threshold.py`, never at a shared threshold.** Two models trained the same
way sit at different places on the score axis, and at a fixed threshold that gap reads as nine
URFD clips which vanish once each model is given its own operating point (SS72).

The control matters: FallVision stores keypoints in pixels and is 58% of the training set, so
`dataset.COORD_SCALE` divides it by 640. That 640 is measured from the coordinates rather than
documented, and the arm trained without FallVision is what checks it. **If the two arms
disagree about frame position, the scale is wrong, not the idea.**

### B. ~~Does preprocessing change an ALERT?~~ — it does, and it is a clean win
**Done 2026-09-25. One measurement left before it can be switched on.**

I predicted no change at all. Wrong, in the good direction. CPU profile, 220 clips:

| | URFD falls | held-out clean | URFD half A clean | half B clean |
|---|---|---|---|---|
| off | 45/60 | 41/56 | 17/20 | 16/20 |
| **auto (below 32)** | **45/60** | **43/56** | **18/20** | **17/20** |

**Not one fall lost on any surface, two false alarms gone**, and the gain shows on both halves
of URFD including the half reserved for confirming. The clips are `adl-22` and `adl-23`, the
two darkest in the corpus.

**The mechanism is not the one it was built for.** Person-found on those clips barely moved
(29→30 and 28→30 frames), so the metric I picked as decisive under-reported it by almost
everything. What changed is that the keypoints the model *does* find are steadier: in a dark
room joint jitter reads as high velocity, which is what a fall looks like. It is a false-alarm
feature, not a recall feature.

**Cost is entirely the source resolution**, and the fix is a setting the README already asks
for: 47 ms at 1080p (38% of the CPU budget, unaffordable), **5 ms at the 640×360 substream
(4%)**, 0.16 ms on any lit frame. That makes the substream a prerequisite for turning this on,
not a nice-to-have.

**What is left:** the same measurement on the GPU profile (960 @ 20 fps), which is one cache
and one replay, about 25 minutes. Then `V3_PREPROCESS=auto` can become the default and the
published numbers re-stated. Until then it stays off, because a default that alters frames
makes every published number describe a detector nobody measured.

---

---

## 0. Before trusting any number here: measure on the server itself
**Effort: two hours. Gain: none directly — but without it every figure below is provisional.**

7.4 fps was measured in a four-core container on a fast desktop chip. The server is a QEMU
virtual CPU with slower cores and will sit below that. If it lands at 6 fps, recall is 22% not
53%, and the input size has to come down another step. Do this first, and re-read SS62's table
at the rate the server actually achieves.

---

## 1. Bring the server's offline CPU cores online
**Gain: potentially 3.5x compute. Effort: an hour, mostly someone else's.**

An earlier SSH check recorded **14 vCPUs allocated but only 4 online** (cores 3, 5, 6, 11). If
those can be enabled, the loop moves from ~8 fps into the 20s and leaves the region where two
frames per second decide whether a fall is caught. **Nothing else on this list comes close.**

Verify first — the reading is from a previous session. Needs SSH, which is not stored anywhere;
ask for it. `lscpu`, `nproc --all`, `/sys/devices/system/cpu/cpu*/online`.

## 2. ~~Try OpenVINO~~ — done, and rejected. Use the camera's low-resolution substream instead
**Done 2026-09-22. The 22% went to a camera setting, not a runtime.**

OpenVINO was measured (SKILL.md SS66). With its thread count fixed — it ignores the cgroup
quota exactly as torch and onnxruntime do — it is 1.38x faster in isolation but only **9%** in
the live loop, and it needs a dependency, an export and a patch on ultralytics.

**Lowering the source resolution beats it and costs nothing**: on four cores, a 640x360 source
runs the loop at **9.8 fps against 8.0 fps from 1080p, a 22% gain**, because the detector
resizes everything to `V3_IMGSZ` anyway and the extra pixels are decoded and thrown away. The
two do not stack — with a small source OpenVINO is *behind* PyTorch (9.3 against 9.8). Its win
was absorbing a preprocessing cost that is better removed than optimised.

**What is left of this item:** make sure the installed camera is actually pointed at its
substream. That is now in README's installation notes, and it is the single largest free
speed-up available on CPU.

ONNX was also tried and is 2.3x *slower* on CPU. Both routes are closed.

## 3. ~~Score a partially filled window~~ — done, and it was the biggest single gain yet
**Done 2026-09-22 (SS70). CPU URFD 32/60 -> 45/60, GPU 45/60 -> 56/60.**

`V3_PARTIAL_MIN=4`: once four real frames are in the buffer, the window is padded at the front
with the earliest observed frame and scored. 4 is the peak of the curve (6 gives 38/60, 2 gives
41/60 on the CPU profile). Held-out clean cost 3 clips on CPU and 1 on GPU, where URFD clean did
not move at all. Confirmed on the reserved URFD half: CPU 16/28 -> 21/28, GPU 21/28 -> 26/28.

The original description is kept below.

<details><summary>the problem it solved</summary>

**Gain: unlocks up to a quarter of falls the detector currently cannot score at all on CPU.
Effort: half a day. Free in runtime cost.**

At 8 fps a 15-frame window needs nearly two seconds of continuously visible person before the
classifier will produce any number at all. **Sixteen of 60 URFD falls never get one.** This is
not a dataset artefact — somebody walking into a room and falling within the first second is
exactly the case the window cannot see, and on a slower machine it gets worse, not better.

Pad a short window with its first observed frame, or score a shorter prefix at a higher
threshold. `training/measure/urfd_window_fill.py` already measures how many clips each variant
unlocks at each rate. **This is the best accuracy-per-hour item on the list for CPU** and it
costs nothing at runtime, because on a camera that is already running the window is full
anyway — it only changes the first seconds after a person appears.

</details>

## 4. A motion gate
**Gain: large on a quiet house; the only credible route to more than one camera on four cores.
Effort: a day, including getting the failure case right.**

One camera at input size 320 uses most of the server, and the detector is ~90% of the loop. In
an elderly person's home most minutes are an empty room. A frame-difference check costing under
a millisecond could skip the pose pass when nothing has moved, and the CPU returned goes
straight into frame rate when something does.

**The failure case that must not happen:** a person already on the floor and not moving must
not be gated out — that is exactly the state an alert needs to keep reporting. Gate the pose
pass only, never the state machine, and hold the last known state through gated frames.

## 5. ~~Re-tune the alert threshold for 8 fps~~ — swept, and 0.65 stands
**Done 2026-09-23 (SS71). Seven thresholds x two smoothing rules, CPU profile, three minutes.**

The premise was sound — 0.65 was chosen at 15 fps on a GPU and the CPU profile runs at 8 fps —
but it is measurably not wrong. On the half of URFD a decision may be made on, 0.65 and 0.70
are identical and nothing beats them; 0.50 buys one fall for one clean clip, and on the
confirming half 0.65 is a fall ahead of 0.70. **Leave it.**

The sweep did find something else: **`2 of 3` smoothing is catastrophic at 8 fps** — URFD 45/60
falls down to 13/60 — because two positives out of three at that rate means two of three
consecutive quarter-seconds, and partial-window scoring makes the early windows of an incident
the weakest ones. At this rate 1-of-3 is a requirement, not a preference.

**Still open:** the same sweep for the GPU profile at 20 fps. The pose cache for it is half
built (`training/measure/cache_pose_streams.py`, `V3_IMGSZ=960 TARGET_FPS=20`); finish it and
the sweep is another three minutes.

<details><summary>the original reasoning</summary>

Threshold 0.65 was chosen at **15 fps on a GPU** (SS51). The CPU profile runs at 8 fps, where
the score distribution is different — that is exactly why SS51 had to pick a new threshold when
the window changed. **The CPU profile is running a threshold that was never tuned for it**, and
so is the GPU profile now that it is at 20 fps. This is a loose end created by SS61/SS62.

Sweep 0.50 / 0.55 / 0.60 / 0.65 / 0.70 at each profile's rate and input size. Choose on half of
URFD (**pairs of sequences, not odd/even** — SS60), confirm on the other half.

</details>

## 6. ~~Finish the pose-model comparison~~ — done. yolo26s-pose stays
**Done 2026-09-22 (SS68). Nothing beat it at matched cost.**

At the input size where each costs what s@320 costs (m at 224, l at 160), person-found on
URFD/GMDCSA24 falls is s 87%/85%, m 80%/84%, l 87%/78%. Neither is better, so no accuracy
sweep was run. `l@160` is level with s on URFD and 11.5 points worse on GMDCSA24's last
third — URFD's 320x240 frames cannot see what a 160 input costs a 720p camera, which is
exactly why item E's real-resolution evaluation set matters.

The original reasoning is kept below because the argument for trying was sound.

<details><summary>why it was worth testing</summary>

Only `n` and `s` were measured. `m`, `l`, `x` were skipped on the assumption that bigger is
slower and therefore worse on CPU — an assumption, not a measurement.

Against: extrapolating the measured n/s ratio (1.8x at imgsz 320) and parameter counts, m at
320 would be ~185 ms and l ~310 ms, so matching s@320's 77 ms means dropping to roughly imgsz
224 for m and 160 for l, where a person four metres away is 30-60 pixels tall. For: **"bigger
model at a smaller input" already won once** — s@320 (34/60) beat n@384 (26/60) — and the
reason n lost is that it **fails to find the person 7.8 points more often**, which a larger
model should improve.

Test m and l; x is almost certainly out of reach. Measure the matched-cost input size and the
person-found rate first. If the bigger model does not find people better than s@320, stop —
no accuracy sweep needed.

</details>

## 7. Remove the second YOLO in alone-detection
**Gain: 12% of the frame rate back, plus a model and a worker slot per camera. Effort: half a day.**

Alone-detection loads `yolo26l.pt` and opens a second video stream to answer "is exactly one
person present", which the fall loop already answers through `fall_state.seen_count`. Measured
on four cores: 6.9 fps without it against 6.1 with. On CPU, 12% of the frame rate is real
recall.

## 8. ~~Let the classifier see where the person is in the frame~~ — measured, and it does not help
**Done 2026-09-26 (SS72). Six training runs plus two controls. The flags ship off.**

The reasoning was sound: `_normalize_and_velocity` subtracts the hip centre and divides by
torso size, so a body on a bed and a body on the floor are the same picture, and height in the
frame is exactly what normalisation throws away. Six of seven GMDCSA24 val false alarms are
beds. Adding the hip's height and the apparent torso size back should have separated them.

It does not. Three seeds each arm, each model given **its own threshold** (two models trained
the same way sit at different points on the score axis, so a shared threshold compares
operating points rather than detectors — SS72):

| view of URFD half B | baseline | frame position | |
|---|---|---|---|
| threshold chosen on half A | 1.452 | 1.502 | +0.050, inside a within-arm spread of 0.20 |
| best achievable | 1.564 | 1.552 | **−0.012** |
| mean over every threshold | 1.444 | 1.417 | **−0.027** |

And on the thing it was actually built for, compared at **matched URFD recall of 45/60**, which
is the only fair way to ask:

| at 45/60 URFD falls | baseline | frame position |
|---|---|---|
| GMDCSA24 val clean (the bed clips) | **8.7/16** | 7.7/16 |
| held-out clean | **39.3/56** | 34.7/56 |

**Worse on exactly the failure it was meant to fix.** The control arms, trained with FallVision
dropped from both sides, agree: baseline 1.650 against 1.586.

Whether it memorised the four rooms GMDCSA24 was filmed in or simply added noise next to the
85 channels that were already there, the answer to "does this help" is no, and the pre-declared
rule was that URFD decides. `USE_FRAME_POSITION` and `V3_FRAME_POSITION` stay in the tree, off,
so nobody spends another day finding this out again.

**Two things worth keeping came out of it.** FallVision stores keypoints in pixels and is 58%
of the training set, which had never mattered because every existing channel is scale
invariant — `dataset.COORD_SCALE` fixes that for good. And `pick_threshold.py`, which exists
because the first comparison read nine URFD clips of difference that turned out to be an
operating point.

## 9. Escalate when the person is STILL down — never cancel when they get up
**Direction corrected by measurement 2026-09-26. The detector already counts it; nothing acts on it yet.**

The original idea was to cancel an alert once the person who triggered it is upright again.
Measured on the cached pose stream, the signal separates — 6% of real falls against 50% of
false alarms get back up — but **six per cent of real falls cancelled is not a trade this
system can make**, and it would be made silently by a rule nobody sees.

The same signal the other way round costs nothing and asserts nothing false: **still on the
floor ten seconds later** holds for 94% of real falls and half the false alarms, and it can
only ever raise urgency. That is the footing the tier already stands on — escalate because
nobody answered, not because the model was confident.

Also learned, and it bounds what this can ever do: only five of URFD's forty-five alerting fall
clips run sixteen scored frames past the alert, and **zero** of the ceiling-camera ones do. The
clips end too soon to ask. GMDCSA24 is the only surface with enough footage after the event, so
anything built on this is confirmed on the dataset this project has tuned against most.

`V3FallDetectionState.frames_since_upright` and `ever_upright` are counted per person now.
What is left is the alerting side: carry it onto the notification, let escalation use it, and
say "still on the floor" in the message rather than a number.

## 10. Is the alert TIER right, and where exactly does it go wrong?
**Gain: the honest answer to "does it cry wolf". Effort: two days, most of it ground truth.**

`alert_tier()` returns `check` for every fresh alert and `confirmed` only once nobody
acknowledged it. That was deliberate — the highest-scoring alert in the whole corpus, 0.96, is
a man getting up from a bed, so the score cannot assert a fall (SS53). But nothing has measured
the tier as a *classifier*: of the alerts that say "please check", how many were real, and of
the real falls, how many never reached the urgent wording because somebody acknowledged a
different alert first.

What is needed that does not exist yet: per-alert ground truth at the instant the alert fires,
not per clip. `training/measure_alert_tier.py` already emits one row per rising edge with its
score — the missing half is a label on each of those rows, and the frame it fired on, so a
wrong tier can be looked at rather than counted.

Report it as a confusion matrix over tiers, plus the frames of every disagreement.

## 11. Cut the compilations into single incidents
**Gain: turns 12 unscoreable clips into a real test set. Effort: two days.**

`Test/1`-`12` hold several incidents each with no per-incident ground truth, so today they are
counted and never scored — `test_result/README.md` says so, and reporting them as accuracy
would be inventing a denominator. Cutting them into one-incident clips with a start and end
time, and labelling each as fall / not-fall, makes them scoreable and roughly triples the
amount of real (non-lab) footage this project can measure against.

Method, agreed: **all of them** — Gemini reads each clip first, then every proposed boundary is
checked by eye on rendered frames before it becomes ground truth. Gemini's free tier is 20
requests a day, so this spans several days or needs a paid key.

Then re-run the comparison against the MediaPipe original on the cut clips, which is the
number worth presenting.

## 12. Choose the model and the preprocessing from the web UI
**Gain: the settings stop being an SSH job. Effort: a day.**

`V3_POSE_MODEL`, `V3_IMGSZ`, `V3_PREPROCESS` and `V3_THRESHOLD` are environment variables read
at import, so changing one means editing `.env` and restarting a container. They belong on the
camera or system settings page.

**The hard part is not the form.** These values are not free choices: input size and frame rate
are one decision on CPU, the threshold is not independent of the window or the rate, and
`tools/check_config_coherence.py` exists because a combination that was never measured end to
end is a detector nobody has tested. A UI that lets someone pick any combination silently
un-measures the system. Offer named profiles that are each measured — "GPU", "CPU server",
"CPU server, dark room" — rather than free-form knobs, and show what each one scored.

## 13. LINE alerts into a group
**Gain: the family sees it, not one person. Effort: half a day.**

Most of this already exists: `LineSettings` stores a per-user channel token and target, the
web UI has the form, the toggle is off by default and the test button refuses to fire while it
is off. What is missing is the group:

- the target field is labelled `LINE User ID` and the UI tells the user to add the bot as a
  friend; a group needs the group ID, which begins with `C`.
- `line_webhook` only handles `postback` events, so when the bot is added to a group the group
  ID is never captured. Handle `join` and store it, which is the only way a user can get that
  ID without reading raw webhook logs.
- the push target is a single value; sending to a person *and* a group means a list.

**Do not send a real message while building this.** The toggle exists so that switching it on
is a deliberate act, and a test push lands on somebody's actual phone.


---

## Deployment and safety — small, and none of it optional

- ~~**The admin password is still `admin123`**~~ — partly done. `ADMIN_PASSWORD` now sets it at
  first start and `SEED_TEST_USER=0` drops the second account; while the default is in place
  the backend warns on every startup and in the System Logs page. The default itself is
  unchanged on purpose, so a fresh clone still matches the README and the smoke tests still
  sign in. **What is left is the actual install: set it.**
- **LINE is off** and needs a channel secret, `PUBLIC_BASE_URL` and a tunnel before it can
  notify anyone. Credentials are saved; nothing has ever been sent.
- ~~**A deployment note for the server**~~ — written: `docs/deploying_on_a_cpu_server.md`,
  covering the compose file to use, the camera substream, the admin password, what to do when
  the loop reports BELOW TARGET, and the things already tried that are not worth repeating.
- ~~**RTSP recovery has never been tested.**~~ — tested, and it was broken: one failed read
  ended the loop permanently while the row stayed active. Fixed in `0281d0b` with a backoff
  reconnect, verified against a stream that was really taken away and given back (SS69).

## Closed by measurement, so they are not tried again

- **Averaging three training seeds (`V3_ENSEMBLE`) is a threshold shift wearing a disguise**
  (SS71). Three seeds at 0.60 and one seed at 0.70 are the same detector to the clip on both
  held-out axes, and the single model is a clip ahead on the half a choice may be made on.
  Everything the ensemble buys, the threshold already buys for free — against three ONNX
  sessions per window per tracked person. Not taken on accuracy grounds; the speed question
  never had to be asked.

---

## Open, with no obvious next move

- **`Test/13` is the one real fall the deployed configuration misses.** A moving, zooming camera
  (4.8 px/frame of background flow against 0.03 for the fixed-camera clips), a young adult,
  ending on hands and knees. Nothing here is built for a moving camera — a named open item
  rather than a defect to chase.
- **Multi-camera capacity has never been measured.** On CPU one camera uses most of the box, so
  the answer is probably one until item 1 or 4 changes it.
- **URFD's usable frame is 320x240**, so it cannot settle input-size questions for a real
  camera; GMDCSA24 (720p) and the `Test/` clips carry that. A held-out set at realistic
  resolution would remove the caveat attached to every input-size decision.
