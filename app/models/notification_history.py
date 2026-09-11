from app import db
from datetime import datetime
import pytz
tz = pytz.timezone('Asia/Bangkok')

class NotificationHistory(db.Model):
    __tablename__ = 'notification_history'
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.Integer, db.ForeignKey('cameras.id'), nullable=True)
    sent_at = db.Column(db.DateTime, default=lambda: datetime.now(tz), index=True)
    detection_type = db.Column(db.String(32), nullable=False)
    image_path = db.Column(db.String(256), nullable=True)
    # Model score behind this alert. Nullable because bed-exit/alone alerts have no single
    # score, and because rows written before this column existed have none. Stored so the
    # UI can show whether an alert was confident or one of the ambiguous ones staff are
    # asked to verify -- previously the value was passed to save_alert_log as
    # additional_info and only printed to the log, so it was lost.
    confidence = db.Column(db.Float, nullable=True)
    # Escalation state. A fall alert nobody has acknowledged is re-sent by
    # escalation_service.check_pending_acknowledgements until someone responds or
    # MAX_ESCALATIONS is reached -- acknowledged_at is what stops that loop, and
    # escalation_count is what bounds it if nobody ever does.
    acknowledged_at = db.Column(db.DateTime, nullable=True)
    acknowledged_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    escalation_count = db.Column(db.Integer, nullable=False, default=0, server_default='0')
    # mp4 of the ~60s before the alert, cut from the in-memory ring buffer at alert
    # time (app/services/clip_buffer.py) and sent to whoever acknowledges the alert.
    clip_path = db.Column(db.String(512), nullable=True)

    camera = db.relationship('Camera', backref=db.backref('notifications', lazy=True)) 