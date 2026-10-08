"""F2/F5 (9 Oct plan): escalation pushes only count when delivered; the 30-minute window holds.

No real LINE call is made: requests.post is replaced by a stub, and a throw-away SQLite database
is used, so this can run on any machine without Docker or the owner's phone.
Run: python tests/test_line_escalation.py
"""
import os, sys, tempfile
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
db_file = os.path.join(tempfile.mkdtemp(), 't.db')
os.environ.update(DATABASE_URL='sqlite:///' + db_file.replace('\\', '/'), LINE_ENABLED='false',
                  ESCALATION_DELAY_MINUTES='3', ESCALATION_MAX_AGE_MINUTES='30', MAX_ESCALATIONS='2',
                  LINE_CHANNEL_ACCESS_TOKEN='', LINE_USER_ID='')
import pytz
import requests
import app as app_pkg
from app import db
from app.models.camera import Camera
from app.models.notification_history import NotificationHistory
from app.models.line_settings import LineSettings
from app.models.user import User
from app.services import escalation_service

tz = pytz.timezone('Asia/Bangkok')
fails = []
calls = []
status = {'code': 200}


class _Resp:
    def __init__(self, code):
        self.status_code, self.text = code, 'stub'


accepted_keys = set()


def fake_post(url, headers=None, json=None, timeout=None):
    """Behaves like LINE: a retry key already accepted answers 409; status['code'] == 'timeout'
    means LINE ACCEPTS the push but the response is lost."""
    calls.append(json)
    key = (headers or {}).get('X-Line-Retry-Key')
    if key and key in accepted_keys:
        return _Resp(409)
    if status['code'] == 'timeout':
        if key:
            accepted_keys.add(key)
        raise requests.exceptions.Timeout('stub timeout')
    if status['code'] == 200 and key:
        accepted_keys.add(key)
    return _Resp(status['code'])


requests.post = fake_post
escalation_service_requests = requests


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + (' -- %s' % detail if detail else ''), flush=True)
    if not cond:
        fails.append(name)


app = app_pkg.get_worker_app()
with app.app_context():
    db.create_all()
    user = User(username='t', password_hash='x')
    db.session.add(user); db.session.commit()
    cam = Camera(name='cam', url='rtsp://x', room_name='room', detection_type='fall_detection', user_id=user.id)
    db.session.add(cam); db.session.commit()
    now = datetime.now(tz).replace(tzinfo=None)

    def alert(minutes_ago):
        n = NotificationHistory(camera_id=cam.id, detection_type='fall_red',
                                sent_at=now - timedelta(minutes=minutes_ago))
        db.session.add(n); db.session.commit()
        return n.id

    a = alert(10)
    # 1. LINE switched off: nothing delivered -> the escalation must NOT be used up
    escalation_service.check_pending_acknowledgements()
    n = db.session.get(NotificationHistory, a)
    check('LINE off -> escalation not counted', (n.escalation_count or 0) == 0 and not calls, n.escalation_count)
    # 2. LINE on but the push fails (HTTP 500) -> not counted
    LineSettings.update_settings(user.id, channel_access_token='tok', line_user_id='Uabc', enabled=True)
    status['code'] = 500
    escalation_service.check_pending_acknowledgements()
    db.session.expire_all(); n = db.session.get(NotificationHistory, a)
    check('push fails -> escalation not counted', (n.escalation_count or 0) == 0 and len(calls) == 1, n.escalation_count)
    # 3. delivered -> counted once
    status['code'] = 200
    escalation_service.check_pending_acknowledgements()
    db.session.expire_all(); n = db.session.get(NotificationHistory, a)
    check('delivered -> escalation counted', n.escalation_count == 1, n.escalation_count)
    # 4. acknowledged -> no further pushes
    n.acknowledged_at = now; db.session.commit(); before = len(calls)
    escalation_service.check_pending_acknowledgements()
    check('acknowledged -> no push', len(calls) == before, len(calls) - before)
    # 5. 30-minute window edges (delay 3 min): 29 min old escalates, 31 min old does not
    in_window, too_old = alert(29), alert(31); before = len(calls)
    escalation_service.check_pending_acknowledgements()
    db.session.expire_all()
    check('29 min old -> escalated', db.session.get(NotificationHistory, in_window).escalation_count == 1)
    check('31 min old -> left alone', (db.session.get(NotificationHistory, too_old).escalation_count or 0) == 0)
    # 7. accepted-then-timeout: counted once on the next sweep (409 under the same key), and
    #    the phone got exactly one message
    t = alert(12); status['code'] = 'timeout'
    escalation_service.check_pending_acknowledgements()
    db.session.expire_all()
    first = db.session.get(NotificationHistory, t).escalation_count or 0
    status['code'] = 200
    escalation_service.check_pending_acknowledgements()
    db.session.expire_all()
    check('accepted-then-timeout -> not lost, not doubled',
          first == 0 and db.session.get(NotificationHistory, t).escalation_count == 1)
    # 8. fairness: 20 newer-but-undeliverable... -> use a user with LINE off for 20 alerts, then one deliverable alert
    for row in NotificationHistory.query.all():
        row.acknowledged_at = now
    db.session.commit()
    user2 = User(username='off', password_hash='x'); db.session.add(user2); db.session.commit()
    LineSettings.update_settings(user2.id, enabled=False)   # LINE off for this user (row made up front: SQLite locks)
    cam2 = Camera(name='cam2', url='rtsp://y', room_name='r2', detection_type='fall_detection', user_id=user2.id)
    db.session.add(cam2); db.session.commit()
    for i in range(20):
        db.session.add(NotificationHistory(camera_id=cam2.id, detection_type='fall_red',
                                           sent_at=now - timedelta(minutes=20 + i * 0.1)))
    db.session.commit()
    fresh = alert(5)
    escalation_service.check_pending_acknowledgements()
    db.session.expire_all()
    check('20 undeliverable older alerts do not starve a new one',
          db.session.get(NotificationHistory, fresh).escalation_count == 1)
    # 9. fairness the other way: 20 NEWER undeliverable alerts must not starve an OLDER deliverable one
    for row in NotificationHistory.query.all():
        row.acknowledged_at = now
    db.session.commit()
    older = alert(25)
    for i in range(20):
        db.session.add(NotificationHistory(camera_id=cam2.id, detection_type='fall_red',
                                           sent_at=now - timedelta(minutes=5 + i * 0.1)))
    db.session.commit()
    escalation_service.check_pending_acknowledgements()
    db.session.expire_all()
    check('20 undeliverable newer alerts do not starve an older one',
          db.session.get(NotificationHistory, older).escalation_count == 1)
    # 10. a user whose LINE always fails (bad token) cannot starve another user over repeated sweeps
    for row in NotificationHistory.query.all():
        row.acknowledged_at = now
    db.session.commit()
    bad = User(username='bad', password_hash='x'); db.session.add(bad); db.session.commit()
    LineSettings.update_settings(bad.id, channel_access_token='badtoken', line_user_id='Ubad', enabled=True)
    cam3 = Camera(name='cam3', url='rtsp://z', room_name='r3', detection_type='fall_detection', user_id=bad.id)
    db.session.add(cam3); db.session.commit()
    for i in range(20):
        db.session.add(NotificationHistory(camera_id=cam3.id, detection_type='fall_red',
                                           sent_at=now - timedelta(minutes=25 - i * 0.1)))
    db.session.commit()
    good = alert(4)
    real_post = requests.post
    requests.post = lambda url, headers=None, json=None, timeout=None: (
        _Resp(401) if json and json.get('to') == 'Ubad' else fake_post(url, headers, json, timeout))
    for _ in range(2):
        escalation_service.check_pending_acknowledgements()
    requests.post = real_post
    db.session.expire_all()
    check('bad-token user does not starve a healthy user (2 sweeps)',
          db.session.get(NotificationHistory, good).escalation_count >= 1)
    # 11. FOUR broken owners x 5 alerts must not starve a healthy fifth owner (Codex P2)
    for row in NotificationHistory.query.all():
        row.acknowledged_at = now
    db.session.commit()
    bad_ids = []
    for b in range(4):
        u = User(username='bad%d' % b, password_hash='x'); db.session.add(u); db.session.commit()
        LineSettings.update_settings(u.id, channel_access_token='bt', line_user_id='Ubad%d' % b, enabled=True)
        c = Camera(name='cb%d' % b, url='rtsp://b', room_name='rb', detection_type='fall_detection', user_id=u.id)
        db.session.add(c); db.session.commit()
        for i in range(5):
            db.session.add(NotificationHistory(camera_id=c.id, detection_type='fall_red',
                                               sent_at=now - timedelta(minutes=28 - b - i * 0.1)))
    db.session.commit()
    healthy = alert(4)
    real_post = requests.post
    requests.post = lambda url, headers=None, json=None, timeout=None: (
        _Resp(401) if json and str(json.get('to', '')).startswith('Ubad') else fake_post(url, headers, json, timeout))
    escalation_service.check_pending_acknowledgements()
    requests.post = real_post
    db.session.expire_all()
    check('4 broken owners x 5 alerts do not starve a healthy owner (1 sweep)',
          db.session.get(NotificationHistory, healthy).escalation_count == 1)
    # 6. timestamps are Bangkok local time
    check('container/app time zone is Asia/Bangkok', escalation_service.tz.zone == 'Asia/Bangkok')
    # 7. no reachable webhook -> no "รับทราบ" button that would do nothing; the text says where to ack
    from app.config import Config
    from app.services.line_service import send_line_message_async
    def sent_message(base, secret):
        Config.PUBLIC_BASE_URL, Config.LINE_CHANNEL_SECRET = base, secret
        del calls[:]
        send_line_message_async(cam.id, 'cam', 'room', 'fall', now, None, notification_id=a, wait=True)
        return calls[-1]['messages'][0]
    for base, secret in (('', ''), ('https://example.test', ''), ('', 'sec')):
        m = sent_message(base, secret)
        check('webhook not configured (base=%r secret=%r) -> plain text pointing to the web' % (bool(base), bool(secret)),
              len(calls[-1]['messages']) == 1 and m['type'] == 'text' and 'มอนิเตอร์' in m['text'], m['type'])
    m = sent_message('https://example.test', 'sec')
    check('webhook ready -> ack button', m['type'] == 'template'
          and m['template']['actions'][0]['data'] == 'ack=%s' % a, m['type'])

print('RESULT', 'FAIL' if fails else 'PASS', fails)
sys.exit(1 if fails else 0)
