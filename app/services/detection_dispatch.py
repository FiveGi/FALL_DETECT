"""Start a camera's detection tasks, and notice when the ones it should have are gone.

Two things made this necessary.

**A camera can report "monitoring" while nothing is monitoring it.** `Camera.is_active` is a
database column, and the loops that read it are Celery tasks held in a worker process. Restart
the worker -- a `docker compose restart`, a machine reboot, an image rebuild -- and every task
dies while every row still says `is_active = True`. The dashboard shows the camera running, the
status endpoint says "running", and `/start` refuses with "already running". Nothing detects
anything, and there is no sign of it anywhere in the UI. For a system whose whole job is to
notice a fall, silently not watching is the worst failure it has.

**The list of tasks a camera needs lived only in the route.** `start_camera` knew that
`fall_v2` means a fall task *and* an alone task; nothing else did, so nothing else could bring
a camera back up without repeating that knowledge. It lives here now and the route calls it.

`running_camera_ids()` asks the workers what they are actually running rather than trusting the
column, so "is this camera really being watched" has one answer both the route and the restart
path use.
"""
from app import celery
from app.config import Config

# The tasks a camera of each detection_type needs. Names rather than imports, because importing
# camera_manager at module scope would be circular -- it imports this module's siblings.
# `fall_v2` is one task, not two. Alone-detection used to be its own task with its own
# VideoCapture and its own YOLO model, decoding the same camera a second time to answer a
# question the fall loop's pose model can answer from the frame it already has. Merging it
# gave back 12% of the fall loop's frame rate on four CPU cores, freed a prefork slot per
# camera, and is *more* accurate: against Gemini-verified counts, pose at 960 gets "exactly
# one person" right 83.1% of the time against the old detector's 80.5%.
TASKS_BY_TYPE = {
    'bed_exit': ['process_bed_exit_detection'],
    'fall': ['process_fall_detection', 'process_alone_detection'],
    'fall_v2': ['process_v2_fall_detection'],
}
DEFAULT_TASKS = ['process_fall_detection']


def task_config():
    """The config dict every detection task is handed. One copy, so a task started by the
    route and the same task restarted after a crash are given identical settings."""
    return {
        'BED_EXIT_MODEL_PATH': Config.BED_EXIT_MODEL_PATH,
        'FALL_DETECTION_MODEL_PATH': Config.FALL_DETECTION_MODEL_PATH,
        'V2_FALL_DETECTION_MODEL_DIR': Config.V2_FALL_DETECTION_MODEL_DIR,
        'V2_FALL_DETECTION_ONNX_PATH': Config.V2_FALL_DETECTION_ONNX_PATH,
        'V2_FALL_DETECTION_CENTER_PATH': Config.V2_FALL_DETECTION_CENTER_PATH,
        'V2_FALL_DETECTION_NORMALIZATION_PATH': Config.V2_FALL_DETECTION_NORMALIZATION_PATH,
        'V2_FALL_DETECTION_THRESHOLD_PATH': Config.V2_FALL_DETECTION_THRESHOLD_PATH,
        'LOGGING_INTERVAL': Config.LOGGING_INTERVAL,
    }


def dispatch_for(camera, config=None):
    """Queue every detection task this camera's type needs. -> list of task ids."""
    from app.services import camera_manager

    config = task_config() if config is None else config
    task_ids = []
    for name in TASKS_BY_TYPE.get(camera.detection_type, DEFAULT_TASKS):
        task = getattr(camera_manager, name)
        task_ids.append(task.apply_async(args=[camera.id, config]).id)
    return task_ids


def running_camera_ids(timeout=2.0):
    """Camera ids that some worker is running a detection task for right now.

    Returns None -- not an empty set -- when the workers cannot be reached, so callers can tell
    "nothing is running" from "I do not know". Guessing the wrong way here would either refuse
    to start a camera that is stopped or start a second loop on one that is already running,
    and a duplicate loop halves the frame rate silently.
    """
    try:
        active = celery.control.inspect(timeout=timeout).active()
    except Exception:
        return None
    if not active:
        return None
    known = set(sum(TASKS_BY_TYPE.values(), [])) | set(DEFAULT_TASKS)
    ids = set()
    for tasks in active.values():
        for task in tasks or []:
            short = (task.get('name') or '').rsplit('.', 1)[-1]
            if short not in known:
                continue
            args = task.get('args') or []
            if args and isinstance(args[0], int):
                ids.add(args[0])
    return ids


def is_really_running(camera_id):
    """True / False / None, where None means the workers could not be asked."""
    ids = running_camera_ids()
    return None if ids is None else (camera_id in ids)


def resume_active_cameras():
    """Re-queue detection for every camera the database says is active but nobody is running.

    Called when a camera worker comes up. A worker restart is the ordinary case -- the tasks
    are infinite loops, so they never survive one -- and without this the cameras stay marked
    active and stay unwatched until somebody notices and presses stop then start.
    """
    from app.models.camera import Camera
    from app.services.logging_service import save_system_log

    running = running_camera_ids()
    resumed = []
    for camera in Camera.query.filter_by(is_active=True).all():
        if running is not None and camera.id in running:
            continue
        dispatch_for(camera)
        resumed.append(camera.name)
    if resumed:
        save_system_log('INFO', 'Detection resumed after worker start for: '
                        + ', '.join(resumed), 'CAMERA')
    return resumed

# --- one loop per camera, enforced where it cannot be raced ------------------------------

_LOCK_TTL_S = 30          # a dead loop's claim expires this long after its last heartbeat
_LOCK_PREFIX = 'camera-loop:'


def _redis():
    """The broker's Redis, which every worker already talks to. None if it cannot be reached."""
    try:
        import redis
        return redis.Redis.from_url(Config.CELERY_BROKER_URL)
    except Exception:
        return None


def claim_camera(camera_id, token):
    """Try to become the one loop for this camera. -> True if the claim is ours.

    `running_camera_ids()` asks the workers what they are running, which is right for a UI
    question and wrong as a guard: there is a window during worker startup where a task has
    been queued but is not yet reported as active, and in that window `/start` and the
    resume-on-worker-start both believe nothing is running and each dispatch a loop. Two loops
    on one camera halve its frame rate, silently -- it has cost this project three wrong
    diagnoses and, on the CPU profile, halving the rate is most of the recall.

    So the loop itself claims the camera, atomically, and a second loop exits instead of
    competing. If Redis is unreachable the claim is granted: dropping detection because a lock
    could not be taken would be worse than the duplicate it prevents.
    """
    r = _redis()
    if r is None:
        return True
    try:
        return bool(r.set(_LOCK_PREFIX + str(camera_id), token, nx=True, ex=_LOCK_TTL_S))
    except Exception:
        return True


def hold_camera(camera_id, token):
    """Refresh our claim. -> False if someone else now holds it (then stop looping)."""
    r = _redis()
    if r is None:
        return True
    try:
        current = r.get(_LOCK_PREFIX + str(camera_id))
        if current is not None and current.decode() != token:
            return False
        r.set(_LOCK_PREFIX + str(camera_id), token, ex=_LOCK_TTL_S)
        return True
    except Exception:
        return True


def release_camera(camera_id, token):
    """Give up the claim, but only if it is still ours."""
    r = _redis()
    if r is None:
        return
    try:
        current = r.get(_LOCK_PREFIX + str(camera_id))
        if current is not None and current.decode() == token:
            r.delete(_LOCK_PREFIX + str(camera_id))
    except Exception:
        pass
