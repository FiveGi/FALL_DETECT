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
   docker-compose up -d --build
   ```
3. For higher load, scale workers:
   ```sh
   docker-compose up --scale celery_worker=5 -d
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
