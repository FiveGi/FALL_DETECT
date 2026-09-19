import base64
import hashlib
import hmac
import os
from datetime import datetime

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

import requests

from app import db
from app.config import Config
from app.models.line_settings import LineSettings
from app.models.notification_history import NotificationHistory

bp = Blueprint('line', __name__, url_prefix='/api/line')


@bp.route('/settings', methods=['GET'])
@jwt_required()
def get_line_settings():
    try:
        user_id = int(get_jwt_identity())
        settings = LineSettings.get_settings(user_id)
        data = settings.to_dict()
        # What the acknowledge button and the 60-second clip need on top of a channel token.
        # Without these, LINE alerts still arrive but pressing "รับทราบ" does nothing and no
        # clip is ever sent, with nothing on screen explaining why -- so the settings page can
        # say which piece is missing instead of leaving the feature quietly half-working.
        base = (Config.PUBLIC_BASE_URL or '').rstrip('/')
        data['acknowledge_ready'] = bool(base) and bool(Config.LINE_CHANNEL_SECRET)
        data['public_base_url'] = base
        data['channel_secret_set'] = bool(Config.LINE_CHANNEL_SECRET)
        data['webhook_url'] = f'{base}/api/line/webhook' if base else ''
        return jsonify({'success': True, 'data': data}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to get LINE settings: {str(e)}'}), 500


@bp.route('/settings', methods=['POST'])
@jwt_required()
def update_line_settings():
    try:
        user_id = int(get_jwt_identity())
        data = request.get_json() or {}

        token = data.get('channel_access_token')
        line_user_id = data.get('line_user_id')
        enabled = data.get('enabled')

        if token is None and line_user_id is None and enabled is None:
            return jsonify({'success': False, 'error': 'Nothing to update'}), 400

        # An empty token from the UI means "leave the stored one alone" -- the field is
        # rendered blank because the real value is never sent back to the browser, so
        # treating blank as a delete would wipe the credential every time someone saves
        # after only flipping the toggle.
        if token is not None and token.strip() == '':
            token = None
        if line_user_id is not None and line_user_id.strip() == '':
            return jsonify({'success': False, 'error': 'LINE user ID cannot be empty'}), 400

        settings = LineSettings.get_settings(user_id)
        will_have_token = bool(token) or bool(settings.channel_access_token)
        will_have_user = bool(line_user_id) or bool(settings.line_user_id)
        if enabled and not (will_have_token and will_have_user):
            return jsonify({
                'success': False,
                'error': 'Cannot enable LINE alerts without both a channel access token and a LINE user ID'
            }), 400

        settings = LineSettings.update_settings(
            user_id=user_id,
            channel_access_token=token.strip() if token else None,
            line_user_id=line_user_id.strip() if line_user_id else None,
            enabled=enabled,
        )
        return jsonify({'success': True, 'message': 'LINE settings updated', 'data': settings.to_dict()}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to update LINE settings: {str(e)}'}), 500


@bp.route('/test', methods=['POST'])
@jwt_required()
def test_line_settings():
    """Send one test message to the saved LINE target.

    Deliberately refuses while alerts are switched off: the toggle is what stops messages
    reaching a real phone, and a test button that ignored it would make the switch a lie.
    """
    try:
        user_id = int(get_jwt_identity())
        settings = LineSettings.get_settings(user_id)

        if not settings.enabled:
            return jsonify({'success': False, 'error': 'LINE alerts are turned off. Turn them on first.'}), 400
        if not settings.channel_access_token or not settings.line_user_id:
            return jsonify({'success': False, 'error': 'Channel access token and LINE user ID are required'}), 400

        resp = requests.post(
            'https://api.line.me/v2/bot/message/push',
            headers={
                'Authorization': f'Bearer {settings.channel_access_token}',
                'Content-Type': 'application/json',
            },
            json={
                'to': settings.line_user_id,
                'messages': [{'type': 'text', 'text': 'ทดสอบการแจ้งเตือนจากระบบเฝ้าระวังผู้สูงอายุ'}],
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return jsonify({'success': True, 'message': 'Test message sent'}), 200
        # LINE's own error body is the only thing that explains a bad token or user id,
        # so pass it through rather than flattening it to "failed".
        return jsonify({'success': False, 'error': f'LINE API {resp.status_code}: {resp.text}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to send test message: {str(e)}'}), 500


def _reply(reply_token, messages):
    """LINE's reply API -- free and rate-limit-friendly, unlike push, and valid for about
    30s after the event, which is plenty for answering a button press."""
    try:
        requests.post(
            'https://api.line.me/v2/bot/message/reply',
            headers={'Authorization': f'Bearer {_token()}', 'Content-Type': 'application/json'},
            json={'replyToken': reply_token, 'messages': messages},
            timeout=10,
        )
    except Exception as e:
        print(f'[LINE] reply failed: {e}')


def _token():
    # The webhook is not authenticated as any particular app user, so it uses whichever
    # enabled LINE configuration exists -- in practice this deployment has one.
    row = LineSettings.query.filter(LineSettings.enabled.is_(True)).first()
    return row.channel_access_token if row else ''


@bp.route('/webhook', methods=['POST'])
def line_webhook():
    """Receives the "รับทราบ" button press from LINE.

    Deliberately unauthenticated in the JWT sense -- LINE's servers call it, not a logged-in
    browser -- so the X-Line-Signature HMAC is the only thing standing between this endpoint
    and anyone on the internet marking alerts as handled. It is verified before the body is
    parsed, and a missing channel secret is treated as "reject everything" rather than
    "trust everything".
    """
    body = request.get_data()
    signature = request.headers.get('X-Line-Signature', '')
    secret = Config.LINE_CHANNEL_SECRET
    if not secret:
        print('[LINE] webhook called but LINE_CHANNEL_SECRET is not set -- rejecting')
        return 'unconfigured', 403

    expected = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
    if not hmac.compare_digest(expected, signature):
        return 'bad signature', 403

    try:
        events = (request.get_json(silent=True) or {}).get('events', [])
    except Exception:
        events = []

    for event in events:
        if event.get('type') != 'postback':
            continue
        data = event.get('postback', {}).get('data', '')
        if not data.startswith('ack='):
            continue
        try:
            notification_id = int(data.split('=', 1)[1])
        except ValueError:
            continue
        _handle_ack(notification_id, event.get('replyToken'))

    # LINE retries on any non-2xx, so return 200 even for events this app ignores.
    return 'ok', 200


def _handle_ack(notification_id, reply_token):
    notification = NotificationHistory.query.get(notification_id)
    if not notification:
        if reply_token:
            _reply(reply_token, [{'type': 'text', 'text': 'ไม่พบการแจ้งเตือนนี้ในระบบ'}])
        return

    already = notification.acknowledged_at is not None
    if not already:
        # acknowledged_by stays NULL: it is a FK to users, and whoever pressed the button in
        # LINE is not necessarily a user of the web app. The timestamp is what stops
        # escalation_service re-sending, which is the point.
        notification.acknowledged_at = datetime.now()
        db.session.commit()

    messages = [{
        'type': 'text',
        'text': 'รับทราบแล้ว ระบบจะหยุดแจ้งซ้ำ' + ('' if not already else ' (มีคนรับทราบไปก่อนหน้านี้แล้ว)'),
    }]

    base_url = (Config.PUBLIC_BASE_URL or '').rstrip('/')
    clip = notification.clip_path
    if clip and base_url and os.path.isfile(clip):
        clip_url = f'{base_url}/api/alert-images/{os.path.basename(clip)}'
        preview = clip_url
        if notification.image_path:
            preview = f'{base_url}/api/alert-images/{os.path.basename(notification.image_path)}'
        # The 60s leading up to the fall -- what actually tells a caregiver whether this was
        # a fall or someone lying down, and whether they moved afterwards.
        messages.append({
            'type': 'video',
            'originalContentUrl': clip_url,
            'previewImageUrl': preview,
        })
    elif clip and not base_url:
        messages.append({'type': 'text', 'text': 'มีคลิปเหตุการณ์ แต่ยังไม่ได้ตั้งค่า PUBLIC_BASE_URL จึงส่งไม่ได้'})

    if reply_token:
        _reply(reply_token, messages)
