"""Re-send fall alerts nobody has acknowledged.

A single push notification is easy to miss -- the phone is on silent, the caregiver is in
another room, the shift just changed. This runs on celery beat and re-sends any fall alert
that has gone unacknowledged past ESCALATION_DELAY_MINUTES, up to MAX_ESCALATIONS times,
stopping as soon as someone presses acknowledge in the web UI.

Only falls escalate. Bed-exit and alone alerts are advisory and would just add noise if
they nagged, and a nagging channel is one people learn to ignore -- which would undo the
point of escalating the alerts that do matter.
"""
from datetime import datetime, timedelta

import pytz

from app import celery, db
from app.config import Config
from app.models.notification_history import NotificationHistory
from app.models.camera import Camera
from app.models.line_settings import LineSettings

tz = pytz.timezone('Asia/Bangkok')
MAX_SENDS_PER_SWEEP = 20


@celery.task(name='app.services.escalation_service.check_pending_acknowledgements')
def check_pending_acknowledgements():
    """Re-send unacknowledged fall alerts that are past their escalation delay.

    Returns a small summary dict instead of None so the result backend / flower shows what
    a run actually did, which is the difference between "beat is healthy but idle" and
    "beat stopped firing" when someone asks why no escalation arrived."""
    from app import get_worker_app
    from app.services.notification_service import notify_alert

    # Cached app, not create_app(): this runs every ESCALATION_CHECK_SECONDS, and a new
    # engine per run exhausted postgres' connection slots until it refused every client.
    app = get_worker_app()
    with app.app_context():
        now = datetime.now(tz).replace(tzinfo=None)
        cutoff = now - timedelta(minutes=Config.ESCALATION_DELAY_MINUTES)
        # Lower bound as well as upper: without it, the first run after this feature ships
        # (or after any downtime) treats the entire unacknowledged history as due and
        # re-sends all of it at once. Escalation is only meaningful while someone can still
        # act on the fall, so anything older than the window is left alone.
        floor = now - timedelta(minutes=Config.ESCALATION_MAX_AGE_MINUTES)

        pending = (NotificationHistory.query
                   .filter(NotificationHistory.detection_type.like('%fall%'))
                   .filter(NotificationHistory.acknowledged_at.is_(None))
                   .filter(NotificationHistory.escalation_count < Config.MAX_ESCALATIONS)
                   .filter(NotificationHistory.sent_at <= cutoff)
                   .filter(NotificationHistory.sent_at >= floor)
                   .order_by(NotificationHistory.sent_at.asc())
                   # Every row in the window: it is bounded by ESCALATION_MAX_AGE_MINUTES, and a
                   # row limit let undeliverable alerts (LINE off for that user) hold every slot so
                   # a deliverable one -- older or newer -- was never tried (Codex P2, twice).
                   .all())

        escalated = attempts = 0
        line_on, broken = {}, set()
        for notif in pending:
            camera = Camera.query.get(notif.camera_id) if notif.camera_id else None
            if camera is None:
                # Camera was deleted after the alert fired. Nobody can act on this any
                # more, so stop it escalating forever rather than leaving it pending.
                notif.escalation_count = Config.MAX_ESCALATIONS
                continue
            # LINE off for this camera's owner: nothing can be delivered, so skip without an
            # attempt (and without counting) instead of spending a send on it.
            if camera.user_id not in line_on:
                line_on[camera.user_id] = bool(LineSettings.get_settings(camera.user_id).enabled)
            if not line_on[camera.user_id]:
                continue
            # Cap real send attempts per sweep, not rows, so a failing network (10 s timeout per
            # target) cannot make one sweep run for minutes. Skipped rows above cost nothing.
            if attempts >= MAX_SENDS_PER_SWEEP:
                break
            # An owner whose send already failed this sweep (bad token, LINE down for them) is
            # skipped for the rest of it: a broken channel costs ONE attempt per sweep, so any
            # number of broken owners cannot use up the cap and starve a healthy one (Codex P2).
            if camera.user_id in broken:
                continue
            attempts += 1

            # Counted only when the re-send reached someone. Counting before sending meant a
            # LINE switched off, a bad token or a network failure used up every escalation
            # (MAX_ESCALATIONS) without one message delivered, and the alert then went quiet
            # for good while still unacknowledged.
            level = (notif.escalation_count or 0) + 1
            delivered = notify_alert(
                camera.id, camera.name, camera.room_name,
                notif.detection_type, notif.sent_at.isoformat(), notif.image_path,
                confidence=notif.confidence,
                escalation_level=level,
                notification_id=notif.id,
                wait=True,
            )
            if delivered:
                notif.escalation_count = level
                escalated += 1
            else:
                broken.add(camera.user_id)

        db.session.commit()
        if escalated:
            print(f'[Escalation] re-sent {escalated} unacknowledged alert(s)')
        return {'checked': len(pending), 'escalated': escalated}
