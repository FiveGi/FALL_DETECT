from app import db
from datetime import datetime
import pytz

from app.config import Config

tz = pytz.timezone('Asia/Bangkok')


class LineSettings(db.Model):
    """Per-user LINE Messaging API credentials.

    Before this, LINE was configured only through environment variables, which meant a
    caregiver could not point alerts at their own LINE account without editing .env and
    restarting the containers. The env vars still act as the seed for a user who has never
    saved settings, so an existing single-user deployment keeps working untouched.
    """
    __tablename__ = 'line_settings'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    channel_access_token = db.Column(db.Text, nullable=True)
    line_user_id = db.Column(db.String(255), nullable=True)
    # Off unless someone deliberately turns it on: enabling this pushes messages to a real
    # phone, so it must never become true as a side effect of creating a row.
    enabled = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(tz))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(tz), onupdate=lambda: datetime.now(tz))

    @classmethod
    def get_settings(cls, user_id):
        settings = cls.query.filter_by(user_id=user_id).first()
        if not settings:
            settings = cls(
                user_id=user_id,
                channel_access_token=Config.LINE_CHANNEL_ACCESS_TOKEN or None,
                line_user_id=Config.LINE_USER_ID or None,
                enabled=Config.LINE_ENABLED,
            )
            db.session.add(settings)
            db.session.commit()
        return settings

    @classmethod
    def update_settings(cls, user_id, channel_access_token=None, line_user_id=None, enabled=None):
        settings = cls.get_settings(user_id)
        if channel_access_token is not None:
            settings.channel_access_token = channel_access_token
        if line_user_id is not None:
            settings.line_user_id = line_user_id
        if enabled is not None:
            settings.enabled = bool(enabled)
        settings.updated_at = datetime.now(tz)
        db.session.commit()
        return settings

    def to_dict(self):
        # The token is a credential: the UI only ever needs to know whether one is stored
        # and to recognise the one it saved, so send a masked tail instead of the value.
        token = self.channel_access_token or ''
        return {
            'id': self.id,
            'enabled': self.enabled,
            'has_token': bool(token),
            'token_preview': f'...{token[-6:]}' if token else '',
            'line_user_id': self.line_user_id or '',
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
