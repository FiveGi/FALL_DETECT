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

## What is actually next

**Blocked on someone with SSH to the server** — items 0 and 1 below. Between them they are
worth more than everything else on this list: the server has 14 vCPUs allocated and 4 online,
and on CPU frame rate is recall.

**Ready to work on, in order of what it buys:**

1. **Finish labelling the 126 incident segments** (item 11). 13 done. This is the binding
   constraint on every other measurement — item 10 could not measure the alert tier because
   every lab clip ends within seconds of its event, and these are the only real footage with
   anything after it.
2. **A motion gate** (item 4). The only credible route to more than one camera on four cores.
3. **Multi-camera capacity has never been measured at all** — see the last section. One camera
   is the assumption, not a finding.

**Needs a decision from whoever owns the product, not more measurement:** whether a care-home
corridor or a lift lobby counts as in scope. It is the only thing the two labelling sources
have disagreed about, and it changes what the accuracy figures are *of*.

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

## 16. "They vanished mid-frame" as a caution — measured, and it does not pay on this data
**Measured 2026-09-29. The idea is sound; the corpus cannot support it and may not be able to.**

The idea, and it is a good one: a tracked person who disappears might have fallen into a
position the pose model cannot read — prone behind furniture, curled on the floor — and the
system would say **nothing at all**, because everything downstream only sees frames where a
person was found. A fall nobody sees is the worst failure this system has and it leaves no
trace in any accuracy number.

A narrow version already ships (`COLLAPSE_ENABLED`) and earns nothing, because it also requires
the classifier to have been above 0.6 first, which almost never holds. The broad version would
fire every time somebody walks out of the room — so the question is whether **where** they
vanish separates the two: a doorway is at the frame edge, a floor is not.

Over all 220 clips, a tracked person gone long enough for the tracker to forget them:

| | clips | vanish, anywhere | at the edge | **away from the edge** |
|---|---|---|---|---|
| fall | 139 | 26 (19%) | 6 | **22 (16%)** |
| no fall | 81 | 12 (15%) | 2 | **10 (12%)** |

16% against 12% is not separation. And against what already ships, as an extra caution:

**+6 falls, +7 false alarms** — one for one. Worse, **all six are GMDCSA24**, the surface this
project has tuned against for months. On URFD, the only independent set, fifteen falls are
missed and **not one of them vanishes mid-frame**. A gain that appears only on the tuned
surface is the pattern this project has learned to distrust.

**But the null result is weak evidence, and that is worth saying.** The failure this idea
targets is a person who becomes undetectable and *stays* undetectable — and every lab clip here
ends within a second or two of its event, so the corpus structurally cannot contain it. This is
the same wall item 10 hit measuring the alert tier. Item 11's real footage is what would settle
it.

**It is also a product judgement, not only a measurement one.** Six falls noticed for seven
extra "please check" messages may be a trade worth making when the alternative is silence — but
that is a decision about what a family will tolerate, and nothing here can make it.

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

## 7. ~~Remove the second YOLO in alone-detection~~ — done for the deployed path
**Done 2026-09-22 (SS67). What is left is dead weight on the v1 path, not a cost anyone pays.**

A `fall_v2` camera -- which is every camera -- runs one task, and alone-detection is answered
inline from the frame the fall loop already has. It cost 12% of the frame rate on four cores as
a separate task with its own stream and its own model, and on CPU frame rate is recall.

`process_alone_detection` and `yolo26l.pt` still exist for `detection_type='fall'`, the
MediaPipe v1 path that nothing selects. Deleting them is a tidy-up with a small risk attached
and no measurable gain, so it is not on this list as work -- it is here so the next person
reading `detection_dispatch.py` knows why there are two answers to the same question.

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

## 9. ~~Escalate when the person is STILL down~~ — built and live-tested
**Done 2026-09-26.**

The original idea was to cancel an alert once the person is upright again. Measured, that
would suppress **6% of real falls**, silently, by a rule nobody sees. The same signal the other
way round costs nothing and asserts nothing false: still on the floor ten seconds later holds
for 94% of real falls and can only ever raise urgency.

`V3FallDetectionState.frames_since_upright` feeds `NotificationHistory.still_down_seconds`,
`alert_tier` may raise `check` to `confirmed` on it and never lowers one, the LINE message says
*which* fact raised it, and the dashboard shows a badge. `tools/check_alert_rules.py` holds the
backend and the web UI to the same rule and to the same threshold, which the API now sends with
each notification so an environment override cannot desync them.

Live-tested against a real clip through the real worker, which found three defects no offline
measurement could: it answered before it asked, the seconds conversion could never reach its
own threshold, and the stored value undercounted the same way. See SS73.

**What the corpus still cannot confirm:** at the deployed ten seconds it never fires on any lab
clip, because they all end too soon (item 10). Only real footage with a minute after the event
settles that, which is item 11.

## 10. ~~Is the alert TIER right?~~ — measured, and the honest answer is "it never says go now"
**Done 2026-09-26. `training/measure/tier_accuracy.py`. The finding is a data gap, not a bug.**

Two facts can raise a fresh alert to `confirmed`, and only one of them can be measured from a
recording: **the person is still on the floor.** "Nobody acknowledged" is about the people
receiving the alert and no clip can answer it. So the tier a clip reaches on its own evidence
is the floor of the tier, never the ceiling.

Over all 130 alerts in the 220-clip set, at the deployed ten-second threshold:

| | `confirmed` | `check` |
|---|---|---|
| alerts on a real fall (112) | **0** | 112 |
| alerts that are false alarms (18) | **0** | 18 |

**It never fires.** Not because the rule is wrong, but because at 8 fps a ten-second window
needs eighty scored frames after the alert and almost no clip in this corpus runs that long.
Sweeping the threshold shows exactly where the data runs out:

| threshold | real falls confirmed | false alarms confirmed | precision of "go now" |
|---|---|---|---|
| 2 s | 32/112 (29%) | 4/18 (22%) | 89% |
| 3 s | 14/112 (12%) | 1/18 (6%) | **93%** |
| 5 s | 4/112 (4%) | 1/18 (6%) | 80% |
| 7 s | 0 | 0 | — |
| **10 s (deployed)** | **0** | **0** | **unmeasurable here** |

**Ten seconds stays.** Lowering it to two so the corpus can score it would be fitting the rule
to the clip lengths of a lab dataset, which is the mistake this project keeps catching itself
making — and two seconds is weak evidence anyway, since somebody who trips and gets up takes
longer than that. The physical argument for ten seconds is strong: nobody who merely bent over
is still down after ten seconds.

**What this actually exposes is the data gap.** Every lab clip ends within a second or two of
the event, so nothing here can measure anything about the period *after* a fall — which is the
period a carer cares about. It blocked item 9's confirmation the same way. Real footage that
keeps running for half a minute after the event is the single most useful thing that could be
added to this project's data, and it is exactly what item 11 would produce.

## 14. ~~The dark costs ten falls in sixty~~ — the gate moved to 70 and it is free
**Done 2026-09-27. The gate default is 70. It costs nothing in the light and is worth three
falls and four clean clips at half light.**

Not one fall clip in any dataset here is dark, so every claim about darkness used to be an
extrapolation from thirteen clips containing no falls. `SIMULATE_DARK` dims footage whose
correct answer is known and adds sensor noise as the signal falls.

**At full light, gates of 32, 50 and 70 are bit-identical on every surface, both profiles** —
CPU 45/60 falls and 43/56 held-out clean, GPU 56/60 and 41/56. The gate costs nothing in a lit
room.

**In a dim one it is the whole difference** (CPU profile, URFD):

| light | falls: off / 32 / **70** | held-out clean: off / 32 / **70** |
|---|---|---|
| full | 45 / 45 / **45** | 43 / 43 / **43** |
| 50% | 40 / 40 / **43** | 41 / 43 / **45** |
| 30% | 35 / 34 / **34** | 43 / 44 / **47** |

At half light it recovers three of the five falls the dark costs *and* gains four clean clips,
both halves of URFD agreeing. 32 does nothing there because **zero of the sixty fall clips fall
below 32 at half light**; it was chosen as provably harmless on a corpus with almost nothing
between luminance 30 and 70, which is the exact band a dim room occupies.

**The lesson worth keeping is about the metric, not the number.** 70 was rejected earlier on
person-found, where it finished four frames behind doing nothing across 7067 frames. On alerts
it is identical in the light and clearly better in the dark. Person-found is a proxy, and it
under-reported this by everything that mattered.

**Still open:** at 30% light nothing recovers the falls — 35/60 becomes 34/60 whatever the gate
does. Below about a third of normal room light this pipeline loses a fifth of its recall and no
setting here gets it back. That is a camera and lighting problem, and it belongs in the install
notes rather than in the code.

## 15. Crop to the people and shrink the input — measured, available, not defaulted
**Done 2026-09-27. `V3_ROI_IMGSZ=256` is at least as good as the deployed frame for 17% less
compute. It stays off, and the reason is a risk the clips cannot show.**

Ultralytics resizes whatever it is given to `imgsz`, so cropping alone saves nothing — it
raises the effective resolution on the person. Cropping *and* lowering `imgsz` trades area for
compute. Timed on four threads at the 640×360 substream:

| | ms/frame | fps where the deployed profile gets 8.0 |
|---|---|---|
| whole frame, imgsz 320 (deployed) | 36.7 | 8.0 |
| crop, imgsz 288 | 34.1 | 8.6 |
| **crop, imgsz 256** | **30.5** | **9.6** |
| crop, imgsz 192 | 23.2 | 12.7 |

Accuracy, CPU profile, against the deployed whole frame at 320:

| | URFD falls | held-out clean | half A (choose) | half B (confirms) |
|---|---|---|---|---|
| deployed, full@320, 8 fps | 45/60 | 43/56 | 24/32, 18/20 | 21/28, 17/20 |
| whole frame @192, 8 fps | 34/60 | 42/56 | 17/32, 16/20 | 17/28, 16/20 |
| crop @192, 8 fps | 38/60 | 42/56 | 20/32, 16/20 | 18/28, 17/20 |
| **crop @256, 8 fps** | **46/60** | 42/56 | **25/32, 18/20** | 21/28, 17/20 |
| crop @256, 10 fps | **48/60** | 42/56 | 25/32, 17/20 | **23/28, 18/20** |

**The mechanism is proven**: at the same input size and rate, the crop is worth four falls
(38/60 against 34/60) for the same cost and the same clean rate.

**crop@256 at 8 fps is better on the half a choice may be made on** (25/32 falls against 24,
clean equal) and identical on the confirming half, at 17% less compute. **crop@256 at 10 fps is
much better on the confirming half** (+2 falls, +1 clean) but slightly worse on the choosing
half, so by this project's own rule it cannot be adopted on that evidence.

**Why it is still off.** The gain over the deployed point is one clip, which is inside the
range a single borderline score flips. What is not inside any measurement here is the risk: the
crop only rescans the whole frame every `V3_ROI_FULL_EVERY` frames, so **somebody walking into
the room can go unseen for up to a second**. Almost every clip in this corpus has its person
present from the first frame, so no number above can see that cost. A 17% compute saving does
not buy that risk on a one-camera install; it might on a machine trying to run three.

`V3_ROI_IMGSZ=256` is there, measured, for the deployment that needs the headroom.

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

## 13. ~~LINE alerts into a group~~ — built, and still switched off
**Done 2026-09-26.**

`LineSettings` carries a group id beside the user id and `targets()` returns both; one push per
target, because LINE's multicast endpoint takes many recipients but refuses group ids. A
failure on one does not stop the others — the reason a group is worth having is that the
individual phone may be asleep.

The hard part was never the push: **a group id cannot be typed in.** It is delivered once, in a
webhook event, when the bot is invited. `LineDiscoveredTarget` records every group the bot
joins and the settings page offers the list. Being in a group is not consent to be alerted in
it — nothing is sent until a target is chosen and the switch is on, and **the switch is still
off. No message has ever been sent.**

Verified in the running container and a real browser: a bad signature is rejected, a join is
recorded, a 1:1 chat is *not* recorded as a group, a leave marks the row rather than deleting
it, and typing a group id in the page round-trips to the database.

**What is left is not code:** a channel secret, `PUBLIC_BASE_URL` and a tunnel, then somebody
deciding to turn it on.

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
