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

## 3. Score a partially filled window instead of waiting for a full one
**Gain: unlocks up to a quarter of falls the detector currently cannot score at all on CPU.
Effort: half a day. Free in runtime cost.**

At 8 fps a 15-frame window needs nearly two seconds of continuously visible person before the
classifier will produce any number at all. **Sixteen of 60 URFD falls never get one.** This is
not a dataset artefact — somebody walking into a room and falling within the first second is
exactly the case the window cannot see, and on a slower machine it gets worse, not better.

Pad a short window with its first observed frame, or score a shorter prefix at a higher
threshold. `scratchpad/urfd_window_fill.py` already measures how many clips each variant
unlocks at each rate. **This is the best accuracy-per-hour item on the list for CPU** and it
costs nothing at runtime, because on a camera that is already running the window is full
anyway — it only changes the first seconds after a person appears.

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

## 5. Re-tune the alert threshold for 8 fps
**Gain: free accuracy, no new code. Effort: three hours of sweeps.**

Threshold 0.65 was chosen at **15 fps on a GPU** (SS51). The CPU profile runs at 8 fps, where
the score distribution is different — that is exactly why SS51 had to pick a new threshold when
the window changed. **The CPU profile is running a threshold that was never tuned for it**, and
so is the GPU profile now that it is at 20 fps. This is a loose end created by SS61/SS62.

Sweep 0.50 / 0.55 / 0.60 / 0.65 / 0.70 at each profile's rate and input size. Choose on half of
URFD (**pairs of sequences, not odd/even** — SS60), confirm on the other half.

## 6. Finish the pose-model comparison: yolo26m-pose and yolo26l-pose
**Gain: unknown, evidence points both ways. Effort: 20 minutes for a first answer.**

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

## 7. Remove the second YOLO in alone-detection
**Gain: 12% of the frame rate back, plus a model and a worker slot per camera. Effort: half a day.**

Alone-detection loads `yolo26l.pt` and opens a second video stream to answer "is exactly one
person present", which the fall loop already answers through `fall_state.seen_count`. Measured
on four cores: 6.9 fps without it against 6.1 with. On CPU, 12% of the frame rate is real
recall.

## 8. Let the classifier see where the person is in the frame
**Gain: targets the dominant false-alarm mode. Effort: a retrain plus three seeds, a day.
Runtime cost: about 0.007 ms per window against a 125 ms budget.**

Six of seven GMDCSA24 val false alarms are beds, and they have survived every decision rule
tried. The cause is structural: `_normalize_and_velocity` subtracts the hip centre and divides
by torso size, so **the model cannot distinguish lying on a bed from lying on the floor** —
both are a horizontal body, identical once centred. Height in the frame is what separates them,
and it is deliberately discarded.

The stored `.npz` keypoints are normalised to [0,1] of the frame, so hip height, hip x and
apparent torso size are **already in the training data**. A retrain, not a re-extraction —
which matters, because the source videos for 97% of the training frames are gone (SS64).

**The risk is overfitting, not speed.** Absolute position lets the model memorise the four rooms
GMDCSA24 was filmed in. Three seeds, and URFD decides: if URFD does not improve, it learned the
rooms rather than the physics.

## 9. Cancel an alert when the person gets up
**Gain: fewer false alarms, no model change, no runtime cost. Effort: half a day.**

The tier keys off escalation rather than the score (SS53), but nothing uses the strongest
evidence available: the pipeline tracks people across frames, so it knows whether the person who
triggered an alert is upright again thirty seconds later. That is how a human judges it, and it
reads honestly in the UI — "they got up" is a fact, not a confidence.

---

## Deployment and safety — small, and none of it optional

- **The admin password is still `admin123`** on a system that will be reachable from a phone.
- **LINE is off** and needs a channel secret, `PUBLIC_BASE_URL` and a tunnel before it can
  notify anyone. Credentials are saved; nothing has ever been sent.
- **A deployment note for the server**: use `docker-compose.yml` alone, **not** the GPU overlay,
  and read the loop's reported frame rate on the first day. Running the GPU settings on a CPU
  host gives 1.5 fps and catches almost nothing.
- **RTSP recovery has never been tested.** If the camera drops, does the loop reconnect or sit
  there marked active? Same class of failure as the worker-restart bug fixed in `b5c683d`, and
  just as silent.

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
