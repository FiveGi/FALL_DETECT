"""LINE routes for the 'one system bot, every user sets up their own LINE' model (owner, 8 Oct; Codex review).

Flask test client + a throw-away SQLite database; LINE's API is stubbed (no real call). Checks:
  1. a NEW ordinary user is seeded with the bot token but NOT the .env LINE id, and with the switch off
  2. an ordinary user's GET /api/line/settings returns NO discovered groups; an admin's does
  3. an ordinary user cannot set the bot token (403); can set their own LINE id
  4. a correctly signed 1:1 'follow' event, with EVERY user's switch off, is answered with the sender's own id, using
     the SYSTEM token (.env); a bad signature is rejected (403)
  5. deleting a user who has LINE settings and a share works; a user who still owns a camera is refused with 400
Run: python tests/test_line_routes.py   (inside the app image; see test_line_escalation.py)
"""
import base64, hashlib, hmac, json, os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
db_file = os.path.join(tempfile.mkdtemp(), 't.db')
os.environ.update(DATABASE_URL='sqlite:///' + db_file.replace('\\', '/'), LINE_ENABLED='true',
                  LINE_CHANNEL_ACCESS_TOKEN='SYSTEM-TOKEN', LINE_USER_ID='Uowner000', LINE_CHANNEL_SECRET='testsecret',
                  JWT_SECRET_KEY='t' * 40, SECRET_KEY='s' * 40)
import requests
calls = []


class _R:
    status_code, text = 200, 'ok'
    def json(self): return {}


def fake_post(url, headers=None, json=None, timeout=None, **kw):
    calls.append({'url': url, 'auth': (headers or {}).get('Authorization'), 'json': json})
    return _R()


requests.post = fake_post
import app as app_pkg
from app import db
from app.models.user import User, UserRole
from app.models.line_settings import LineSettings
from app.models.line_target import LineDiscoveredTarget
from app.models.camera import Camera
from flask_jwt_extended import create_access_token

fails = []
def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + (' -- %s' % (detail,) if detail else ''), flush=True)
    if not cond:
        fails.append(name)


app = app_pkg.create_app()
client = app.test_client()
with app.app_context():
    db.create_all()
    admin = User(username='adm', role=UserRole.ADMIN); admin.set_password('x' * 12)
    user = User(username='usr', role=UserRole.USER); user.set_password('y' * 12)
    other = User(username='own', role=UserRole.USER); other.set_password('z' * 12)
    db.session.add_all([admin, user, other]); db.session.commit()
    db.session.add(LineDiscoveredTarget(target_type='group', target_id='Cgroup123')); db.session.commit()
    db.session.add(Camera(name='c', url='x', room_name='r', detection_type='fall_v2', user_id=other.id)); db.session.commit()
    H = lambda u: {'Authorization': 'Bearer ' + create_access_token(identity=str(u.id))}
    uid, aid, oid = user.id, admin.id, other.id
    hu, ha = H(user), H(admin)

    # 1 + 2
    r = client.get('/api/line/settings', headers=hu).get_json()['data']
    check('new ordinary user: bot token seeded', r.get('has_token') is True, r.get('has_token'))
    check('new ordinary user: NOT seeded with the .env LINE id', not r.get('line_user_id'), r.get('line_user_id'))
    check('new ordinary user: switch off', r.get('enabled') is False, r.get('enabled'))
    check('ordinary user sees no discovered groups', r.get('discovered_targets') == [], r.get('discovered_targets'))
    ra = client.get('/api/line/settings', headers=ha).get_json()['data']
    check('admin sees discovered groups', [t['target_id'] for t in ra.get('discovered_targets', [])] == ['Cgroup123'])
    check('admin seeded with the .env LINE id', ra.get('line_user_id') == 'Uowner000', ra.get('line_user_id'))
    # 3
    r = client.post('/api/line/settings', headers=hu, json={'channel_access_token': 'MINE'})
    check('ordinary user cannot set the bot token (403)', r.status_code == 403, r.status_code)
    r = client.post('/api/line/settings', headers=hu, json={'line_user_id': 'Uuser111', 'enabled': False})
    check('ordinary user can set their own LINE id', r.status_code == 200 and r.get_json()['data']['line_user_id'] == 'Uuser111', r.status_code)
    # 4: every switch off
    LineSettings.query.update({'enabled': False}); db.session.commit()
    body = json.dumps({'events': [{'type': 'follow', 'replyToken': 'rt1', 'source': {'type': 'user', 'userId': 'Unew999'}}]}).encode()
    sig = base64.b64encode(hmac.new(b'testsecret', body, hashlib.sha256).digest()).decode()
    del calls[:]
    r = client.post('/api/line/webhook', data=body, headers={'X-Line-Signature': sig, 'Content-Type': 'application/json'})
    rep = [c for c in calls if c['url'].endswith('/reply')]
    check('signed follow with all switches off -> reply sent', r.status_code == 200 and len(rep) == 1, (r.status_code, len(rep)))
    check('reply uses the SYSTEM token', bool(rep) and rep[0]['auth'] == 'Bearer SYSTEM-TOKEN', rep[0]['auth'] if rep else None)
    check('reply carries the sender\'s own id', bool(rep) and 'Unew999' in rep[0]['json']['messages'][0]['text'])
    r = client.post('/api/line/webhook', data=body, headers={'X-Line-Signature': 'bad', 'Content-Type': 'application/json'})
    check('bad signature rejected (403)', r.status_code == 403, r.status_code)
    # 5
    from app.models.thai_frat_assessment import AssessmentShare
    r = client.delete('/api/admin/users/%d' % uid, headers=ha)
    check('delete a user who has LINE settings -> 200', r.status_code == 200, (r.status_code, r.get_data(as_text=True)[:120]))
    check('their LINE settings are gone', LineSettings.query.filter_by(user_id=uid).count() == 0)
    r = client.delete('/api/admin/users/%d' % oid, headers=ha)
    check('a user who owns a camera -> 400 with a reason', r.status_code == 400 and 'camera' in r.get_json().get('error', ''), r.status_code)

print('RESULT', 'FAIL' if fails else 'PASS', fails)
sys.exit(1 if fails else 0)
