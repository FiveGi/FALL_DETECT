# Elderly Surveillance System

A Flask backend (bed-exit and fall detection, camera management, logging, Telegram/LINE
notifications) plus its Vue 3 admin dashboard, in one repo so a single clone gets a working
system end to end.

- `./` (root) - Flask/Celery backend, see below
- `frontend/` - Vue 3 + Vite admin dashboard, see [frontend/README.md](frontend/README.md)

## Installation

Model weights under `models/*.onnx` are stored with [Git LFS](https://git-lfs.com/). Install it
**before** cloning, or the app will fail at startup with `InvalidProtobuf: Protobuf parsing failed`
(the checkout silently leaves small LFS pointer files in place of the real weights):

```sh
git lfs install
git clone <this repo>
# already cloned without LFS? fetch the real files in place:
git lfs pull
```

### Backend

1. Copy `.env.example` to `.env` and configure your values
2. Build and run with Docker Compose:
   ```sh
   docker compose up -d --build
   ```
   (older Docker installs may only have the hyphenated `docker-compose` command instead -- same
   thing, just swap the command name in every example below)
3. For higher load, scale workers:
   ```sh
   docker compose up --scale celery_worker=5 -d
   ```

### Frontend

```sh
cd frontend
cp .env.example .env      # VITE_API_BASE_URL already points at the backend's default port
npm install
npm run dev                # http://localhost:3000
```

`VITE_FIREBASE_*` values in `frontend/.env` are only needed for the notification features that
use Firebase; the dashboard runs without them.

## Usage

- The API will be available at `http://localhost:8932/api/`
- The dashboard (once `npm run dev` is running) at `http://localhost:3000/`
- Flower (Celery monitoring) at `http://localhost:5555/`
- Default admin credentials: `admin` / `admin123`
- See [API Documentation](API_DOCUMENTATION.md) for complete endpoint reference

### Trying it without a real camera

`Test/` has 17 real-world video clips (see [Test/README.md](Test/README.md)) that the backend can
run detection on exactly like a live camera. In the dashboard's "เพิ่มกล้องใหม่"/"แก้ไขกล้อง" form,
choose "ไฟล์วิดีโอทดสอบ" as the video source and pick one from the dropdown -- no need to type a
path or own a camera to see fall detection working end to end. `Test/13.mp4`-`Test/16.mp4` are real
fall footage, one fall each. `Test/17.mp4` has **no fall in it** -- it is a crowd doing an outdoor
exercise routine, and it is in here on purpose, because staying silent through a lot of
squatting and bending is as much a result as catching a fall. `1.mp4`-`12.mp4` are multi-scene
compilation clips (harder, mixed camera angles -- expect the model to catch some falls in them
but not every single one).

## Checking it works

Four checks, in the order worth running them. The first two need nothing but Python; the third
needs the stack up; the fourth needs a browser.

```bash
python tools/check_config_coherence.py     # the model, the runtime and the training defaults agree
python tools/check_alert_rules.py          # the backend and the web UI word alerts the same way
python training/smoke_test_api.py          # 28 endpoint checks against a running backend
node tools/smoke_test_frontend.mjs         # loads all 8 pages in a real browser
```

The frontend check needs a Chromium listening for CDP:

```bash
msedge --headless=new --remote-debugging-port=9333 --user-data-dir=/tmp/cdp about:blank
```

`check_config_coherence.py` is the one to run after changing anything about the detector. The
classifier's window is a fixed number of *frames*, so the window size, the training stride and
the camera's sampling rate (`V3_TARGET_FPS`) are one decision living in four files; changing one
alone leaves a system that starts normally, passes every other check, and behaves like a
detector nobody measured.

## How accurate is it, really

On **URFD**, a public dataset that was never used to train or tune anything here, fed at the
rate each deployment profile actually runs at:

| | with a GPU (20 fps) | four CPU cores (8 fps) |
|---|---|---|
| falls caught | 45 of 60 (75%) | 32 of 60 (53%) |
| normal-activity clips with no false alarm | 33 of 40 (83%) | 34 of 40 (85%) |

The gap between the columns is frame rate, not a different model: the same weights run in both.
A CPU machine cannot feed the detector as many frames per second, and the classifier's window
is a fixed number of frames, so it sees a slower, coarser version of the same fall.

Seven of those 60 URFD clips are ones no classifier here could have scored: they are the
dataset's ceiling camera on its standing falls, where the room is empty for two thirds of the
clip and the person walks into view as they land, so the clip ends before the 15-frame window
has 15 frames with a person in them. Over the 53 clips that can be scored, GPU recall is
**44 of 53 (83%)**. Both figures are honest; the table above is the conservative one.

Two numbers you will see quoted elsewhere for systems like this, and why they are not these:

- **Accuracy on the dataset a model was tuned against runs ~25 points higher.** GMDCSA24 reports
  96-100% here; it has been used to pick thresholds and input sizes repeatedly, so it measures
  tuning as much as ability.
- **Accuracy measured by reading every frame of a video file runs ~10-15 points higher than the
  same model on a live camera**, because a live camera delivers fewer frames per second and the
  classifier's window then spans a different amount of real time. Both numbers above are
  measured at the live rate.

Practical consequences worth knowing before installing a camera:

- **Mount it on a wall, not the ceiling.** From directly overhead the pose model finds a person
  in only 12-29% of frames versus 78-88% from a wall, and the classifier never even runs.
- **It runs on CPU, but only with the CPU settings.** `docker-compose.yml` on its own is the
  CPU deployment and is tuned for a four-core machine: input size 320 and the camera rate
  pinned to 8 fps. Measured there, it catches 32 of 60 URFD falls and 64 of 79 GMDCSA24 falls,
  against 5 and 48 with the GPU settings left in place — which is the difference between a
  system that works and one that does not. `docker-compose.gpu.yml` overlays the GPU settings
  (input size 960, 20 fps) and reaches 45 of 60 URFD falls. Do not run the GPU settings on a
  CPU host: one camera manages 1.5 fps there and almost nothing is caught.
- **Point it at the camera's low-resolution substream, not the main stream.** Most IP cameras
  publish a second, smaller RTSP stream. On a CPU host that one setting is the largest free
  speed-up there is: measured on four cores, a 640x360 source runs the loop at **9.8 fps
  against 8.0 fps from 1080p**, a 22% gain for no change to the system. The detector resizes
  everything to `V3_IMGSZ` anyway, so the extra pixels are decoded and thrown away, and on CPU
  that decoding and resizing is real work.
- **Frame rate is the thing to watch.** The loop prints its achieved rate once a minute and
  says `<-- BELOW TARGET` when it cannot keep up. If it does, lower `V3_TARGET_FPS` to what the
  machine reaches and re-check the numbers rather than leaving it short.
- **Every fall alert asks a human to look** rather than asserting a fall, because the model's
  confidence score was measured and does not separate real falls from false alarms. Urgency
  comes from an alert going unacknowledged, which is a fact rather than a guess.
