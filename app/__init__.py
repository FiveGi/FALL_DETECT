import os
from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from celery import Celery
from dotenv import load_dotenv
from flask_cors import CORS


from sqlalchemy import text
from sqlalchemy import inspect


load_dotenv()


db = SQLAlchemy()
jwt = JWTManager()
celery = Celery(__name__)

# Configured at module level, not inside create_app(): the worker and beat containers start
# with `celery -A app.celery ...`, which only imports this package -- create_app() never runs
# there, so anything set inside it is invisible to beat and the schedule below would silently
# never fire. Flask's own process still calls create_app() and re-applies the same config.
from app.config import Config as _Config

# Only the keys beat/worker need, and only in celery's modern names: Config carries both
# old-style (CELERY_BROKER_URL) and new-style (task_acks_late) keys, and handing celery both
# at once makes it refuse to start with "Cannot mix new and old setting keys".
celery.conf.broker_url = _Config.CELERY_BROKER_URL
celery.conf.result_backend = _Config.CELERY_RESULT_BACKEND
celery.conf.timezone = 'Asia/Bangkok'
# Camera tasks are infinite loops that hold a prefork slot for as long as the camera is
# active, so anything sharing their queue waits forever behind them. Maintenance work gets
# its own queue, served by the celery_maintenance container.
celery.conf.task_routes = {
    'app.services.escalation_service.*': {'queue': 'maintenance'},
}
celery.conf.beat_schedule = {
    'check-pending-acknowledgements': {
        'task': 'app.services.escalation_service.check_pending_acknowledgements',
        'schedule': float(_Config.ESCALATION_CHECK_SECONDS),
        'options': {'queue': 'maintenance'},
    },
}


import app.services.camera_manager
# Imported for its @celery.task side effect, same as camera_manager above: without this
# the worker never registers the task and beat's messages die as "unregistered task".
import app.services.escalation_service


# A camera worker coming up re-queues detection for every camera the database still says is
# active. Detection tasks are infinite loops, so none of them survives a worker restart, and
# before this the rows stayed `is_active = True` with nothing running: the dashboard said
# "monitoring", the status endpoint said "running", `/start` refused as "already running", and
# no fall would ever have been detected again. Measured, not guessed -- a
# `docker compose restart celery_worker` produced exactly that, with zero detection log lines
# afterwards.
#
# It lives here rather than in celery_worker.py because the workers start with
# `celery -A app.celery`, which imports this package and never imports that file -- a handler
# put there is dead code, which is how the first attempt at this failed.
#
# Guarded by RESUME_ACTIVE_CAMERAS so only the camera worker does it: the maintenance worker
# shares this module and serves its own queue, and must not start camera loops.
if os.environ.get('RESUME_ACTIVE_CAMERAS') == '1':
    from celery.signals import worker_ready as _worker_ready

    @_worker_ready.connect
    def _resume_cameras_on_start(**_):
        from app.services.detection_dispatch import resume_active_cameras
        try:
            with get_worker_app().app_context():
                resumed = resume_active_cameras()
        except Exception as exc:                  # never take the worker down over this
            print(f'[Celery Worker] could not resume active cameras: {exc}', flush=True)
            return
        print(f'[Celery Worker] resumed detection for: {", ".join(resumed)}' if resumed
              else '[Celery Worker] no active cameras needed resuming', flush=True)


_worker_app = None


def get_worker_app():
    """The one Flask app for this process. Use this, not create_app(), from celery tasks and
    notification threads.

    create_app() builds a new SQLAlchemy engine -- and therefore a new connection pool --
    every time it is called, and nothing disposes the old one. Called per task, per alert and
    once a minute from the escalation sweep, that exhausted postgres' connection slots and
    took the database down entirely. Caching is safe here because a celery prefork process
    runs one task at a time and the app holds no per-camera state.
    """
    global _worker_app
    if _worker_app is None:
        _worker_app = create_app()
    return _worker_app


def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')
    db.init_app(app)
    jwt.init_app(app)
    # celery config is applied at module level above (see comment there); re-applying
    # app.config here would push flask's old-style CELERY_* keys back in and trip celery's
    # "cannot mix new and old setting keys" check inside the worker.
   
    CORS(app, resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Content-Type", "Authorization"],
            "supports_credentials": False
        }
    })


    # Paths a request arriving over the public tunnel is allowed to reach. Everything LINE
    # needs, nothing else.
    PUBLIC_EDGE_PREFIXES = ('/api/alert-images/', '/api/line/webhook')

    @app.before_request
    def _restrict_public_edge():
        """404 anything on the public hostname that LINE does not need.

        The host is derived from PUBLIC_BASE_URL, so this only engages once a tunnel is
        actually configured; with it empty (the default) nothing changes. X-Forwarded-Host is
        checked too because cloudflared and similar proxies rewrite Host.
        """
        public_base = (app.config.get('PUBLIC_BASE_URL') or '').strip()
        if not public_base:
            return None
        public_host = public_base.split('//')[-1].split('/')[0].lower()
        seen = (request.headers.get('X-Forwarded-Host') or request.host or '').lower()
        if not seen or public_host not in seen:
            return None  # local/LAN request -- untouched
        if request.path.startswith(PUBLIC_EDGE_PREFIXES):
            return None
        from flask import abort
        abort(404)

    @app.before_request
    def _handle_options_requests():
        if request.method == 'OPTIONS':
            response = app.make_default_options_response()
            return response


    # Add static file serving for videos
    from flask import send_from_directory
   
    @app.route('/videos/<filename>')
    def serve_video(filename):
        video_dir = '/app/Test'  # Use absolute path to videos folder
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        return send_from_directory(video_dir, filename)

    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        try:
            from app.models.token_blocklist import TokenBlocklist
            jti = jwt_payload['jti']
            return TokenBlocklist.is_jti_blacklisted(jti)
        except Exception as e:
            print(f"Error checking token blacklist: {e}")
            return False


    from .models import user, camera, detection_log, system_log, notification_history, line_settings, token_blocklist, thai_frat_assessment
   
    with app.app_context():
        try:
            db.create_all()
            from app.models.user import User, UserRole


            # Automatically add telegram_chat_id column if it is missing in existing DB
            inspector = inspect(db.engine)
            if 'users' in inspector.get_table_names() and 'telegram_chat_id' not in [c['name'] for c in inspector.get_columns('users')]:
                try:
                    db.session.execute(text('ALTER TABLE users ADD COLUMN telegram_chat_id VARCHAR(255)'))
                    db.session.commit()
                    print('Added missing column telegram_chat_id to users table')
                except Exception as e:
                    db.session.rollback()
                    print(f'Could not add telegram_chat_id column automatically: {e}')


            # Same pattern as telegram_chat_id above: add the notification_history columns
            # introduced after the table shipped, so alert history keeps loading on an
            # existing database instead of erroring on a missing column.
            if 'notification_history' in inspector.get_table_names():
                existing = [c['name'] for c in inspector.get_columns('notification_history')]
                for column, ddl in (
                    ('confidence', 'ALTER TABLE notification_history ADD COLUMN confidence FLOAT'),
                    ('acknowledged_at', 'ALTER TABLE notification_history ADD COLUMN acknowledged_at TIMESTAMP'),
                    ('acknowledged_by', 'ALTER TABLE notification_history ADD COLUMN acknowledged_by INTEGER'),
                    ('escalation_count', 'ALTER TABLE notification_history ADD COLUMN escalation_count INTEGER NOT NULL DEFAULT 0'),
                    ('clip_path', 'ALTER TABLE notification_history ADD COLUMN clip_path VARCHAR(512)'),
                ):
                    if column in existing:
                        continue
                    try:
                        db.session.execute(text(ddl))
                        db.session.commit()
                        print(f'Added missing column {column} to notification_history table')
                    except Exception as e:
                        db.session.rollback()
                        print(f'Could not add {column} column automatically: {e}')

            if not User.query.filter_by(username='admin').first():
                admin = User(username='admin', role=UserRole.ADMIN)
                admin.set_password('admin123')
                db.session.add(admin)
                print("Created admin user: username='admin', password='admin123', role='admin'")
           
            if not User.query.filter_by(username='testuser').first():
                test_user = User(username='testuser', role=UserRole.USER)
                test_user.set_password('user123')
                db.session.add(test_user)
                print("Created test user: username='testuser', password='user123', role='user'")
           
            db.session.commit()
            print("Database tables created successfully!")
        except Exception as e:
            print(f"Error creating tables: {e}")
            db.session.rollback()


    from .routes import register_blueprints
    register_blueprints(app)


    return app

