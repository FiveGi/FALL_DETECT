# Deploying on a server with no GPU

Everything here was measured on a container limited to four cores with CUDA off, which is the
shape of the machine this ships to. The numbers move with the hardware; the order of operations
does not.

The one thing to understand before anything else: **on a CPU host, speed is accuracy.** The
classifier's window is a fixed number of frames, so the frame rate decides how much of a fall
it sees. At input size 320, URFD recall is 13 of 60 falls at 6 fps, 32 at 8 fps and 34 at 9.
Two frames per second is the difference between a system that works and one that does not, so
every step below is really about frame rate.

---

## 1. Use the CPU compose file, not the GPU one

```sh
docker compose up -d --build          # docker-compose.yml alone IS the CPU deployment
```

Do **not** add `-f docker-compose.gpu.yml`. That overlay sets input size 960 and 20 fps, which
a four-core machine cannot feed: one camera manages 1.5 fps and catches 5 of 60 URFD falls.
The CPU file sets input size 320 and pins the rate at 8, which catches 32.

## 2. Point the camera at its low-resolution substream

The largest free speed-up available, and it is a setting on the camera rather than a change
here. Most IP cameras publish a second, smaller RTSP stream:

| source | achieved frame rate |
|---|---|
| 1080p main stream | 8.0 fps |
| 640x360 substream | **9.8 fps** |

The detector resizes everything to `V3_IMGSZ` anyway, so the extra pixels are decoded and then
thrown away — and on a CPU that decoding and resizing is real work. There is nothing to gain
from a higher-resolution stream and 22% of the frame rate to lose.

## 3. Set the admin password before anyone can reach it

```sh
# in .env, before the first start
ADMIN_PASSWORD=something-only-you-know
SEED_TEST_USER=0          # leaves out the second testuser/user123 account
```

This is a login that gets opened from a phone to be told somebody has fallen. While the default
password from the README is still in place the backend says so on every startup and writes a
warning to the System Logs page; that warning stops once it is changed.

## 4. Watch the frame rate on the first day

The camera loop prints its achieved rate once a minute:

```
[Camera 1] V2 Fall Log: no_fall, Confidence: 0.05, People seen: 1,
           Detection rate: 7.4/8 fps  [read 2% clip 1% detect 40% other 5%]
```

`<-- BELOW TARGET` after the rate means the machine is not keeping up, and the accuracy figures
above no longer describe it. If that happens, in this order:

1. Check step 2 — a 1080p stream is the usual cause.
2. Give the worker more of the machine: `cpus: '3.5'` under `celery_worker` in
   `docker-compose.yml`. It is a ceiling, not a reservation, and the backend and the
   maintenance worker are idle almost all the time.
3. Lower `V3_IMGSZ` one step (320 → 256) and lower `V3_TARGET_FPS` to what the loop actually
   reaches, then **re-read the accuracy table in SKILL.md SS62 at that rate** rather than
   assuming the old numbers still hold.

`python tools/check_config_coherence.py` refuses any combination that has never been measured
end to end, which is the point of it.

## 5. What this machine can and cannot do

- **One camera.** One at input size 320 uses most of four cores, and the detector is about 90%
  of the loop. Multi-camera capacity has not been measured because there is nothing to measure
  it with yet.
- **A wall, not a ceiling.** From directly overhead the pose model finds a person in 12-29% of
  frames against 78-88% from a wall, and the classifier never runs at all.
- **Roughly half the falls of a GPU host**: 32 of 60 URFD falls against 45, with a slightly
  *better* false-alarm rate. The gap is frame rate, not a different model — the same weights
  run in both.

## Things that were tried and are not worth doing

Recorded so nobody spends the afternoon again:

- **ONNX Runtime for the pose model**: 2.3x *slower* than PyTorch on CPU.
- **OpenVINO**: 9% faster in the live loop once its thread count is fixed, which is less than
  step 2 gives for free, and the two do not stack — with a small source it is *behind*
  PyTorch. Details in SKILL.md SS66.
- **A smaller pose model** (`yolo26n-pose`): twice as fast, so it can run at a larger input for
  the same cost, and it still catches fewer falls — it loses the person 7.8 points more often,
  which is what matters. Bigger ones (`m`, `l`) lose too, because matching the cost forces the
  input down to where a person four metres away is a few dozen pixels. SS64 and SS68.
- **Pinning CPU affinity** instead of thread counts: catastrophic, 4.7 seconds per frame. The
  container's quota is CFS time across all cores, not a set of cores.
