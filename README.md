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
path or own a camera to see fall detection working end to end. `Test/13.mp4`-`Test/17.mp4` are real
elderly-fall footage; `1.mp4`-`12.mp4` are multi-scene compilation clips (harder, mixed camera
angles -- expect the model to catch some falls in them but not every single one).

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
15 fps the camera loop is pinned to:

| | |
|---|---|
| falls caught | 41 of 60 (68%) |
| normal-activity clips with no false alarm | 34 of 40 (85%) |

Across everything never used in training — URFD plus the held-out GMDCSA24 split — that is
**56 of 75 falls caught**, with 41 of 56 normal clips silent.

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
- **It needs a GPU.** On CPU the pipeline manages 2-8 fps, which is far below what the model
  needs; detection effectively stops working.
- **Every fall alert asks a human to look** rather than asserting a fall, because the model's
  confidence score was measured and does not separate real falls from false alarms. Urgency
  comes from an alert going unacknowledged, which is a fact rather than a guess.
