"""Live security check of the MJPEG media token, against a running backend.

Every case Codex listed in the design review that can be exercised from outside, plus the ones
that need the server's secret (expired and forged tokens), which run inside the backend
container. Needs: a probe user and a camera that user owns; another camera the user does NOT own.

Usage:
    API=http://127.0.0.1:8932/api PROBE_USER=... PROBE_PASS=... OWN_CAM=14 OTHER_CAM=11 \
        python tools/check_media_token.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

API = os.environ.get('API', 'http://127.0.0.1:8932/api')
USER, PASS = os.environ['PROBE_USER'], os.environ['PROBE_PASS']
OWN, OTHER = int(os.environ['OWN_CAM']), int(os.environ['OTHER_CAM'])
failures = []


def req(method, path, token=None, body=None, stream_peek=False):
    r = urllib.request.Request(API + path, method=method,
                               data=json.dumps(body).encode() if body is not None else None)
    if body is not None:
        r.add_header('Content-Type', 'application/json')
    if token:
        r.add_header('Authorization', 'Bearer ' + token)
    try:
        resp = urllib.request.urlopen(r, timeout=15)
        data = resp.read(2048) if stream_peek else resp.read()
        return resp.status, dict(resp.headers), data
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def check(name, ok, detail=''):
    print('%-4s %s %s' % ('PASS' if ok else 'FAIL', name, detail))
    if not ok:
        failures.append(name)


def login():
    s, _, b = req('POST', '/auth/login', body={'username': USER, 'password': PASS})
    d = json.loads(b)
    return d.get('access_token') or d.get('data', {}).get('access_token')


def media(jwt, cam):
    s, h, b = req('POST', '/stream/camera/%d/media-token' % cam, token=jwt)
    return s, h, (json.loads(b).get('token') if s == 200 else None)


jwt = login()
s, h, tok = media(jwt, OWN)
check('owner can get a media token', s == 200 and bool(tok), 'HTTP %s' % s)
check('issuance is Cache-Control: no-store', 'no-store' in h.get('Cache-Control', ''))
s, _, _ = media(jwt, OTHER)
check("cannot get a token for someone else's camera", s == 404, 'HTTP %s' % s)
s, _, _ = req('POST', '/stream/camera/%d/media-token' % OWN)
check('token issuance needs a login', s == 401, 'HTTP %s' % s)

s, h, b = req('GET', '/stream/camera/%d?t=%s' % (OWN, tok), stream_peek=True)
check('owner with a token gets the stream', s == 200 and b'--frame' in b, 'HTTP %s' % s)
check('stream carries no CORS header', 'Access-Control-Allow-Origin' not in h,
      h.get('Access-Control-Allow-Origin', ''))
s, _, _ = req('GET', '/stream/camera/%d' % OWN)
check('no token -> 401', s == 401, 'HTTP %s' % s)
s, _, _ = req('GET', '/stream/camera/%d?t=%s' % (OTHER, tok))
check('token for one camera refused on another', s == 401, 'HTTP %s' % s)
s, _, _ = req('GET', '/stream/camera/%d?t=%s' % (OWN, tok[:-3] + 'AAA'))
check('tampered token -> 401', s == 401, 'HTTP %s' % s)
s, _, _ = req('GET', '/stream/camera/%d?t=%s' % (OWN, jwt))
check('a login JWT is not accepted as a media token', s == 401, 'HTTP %s' % s)
s, _, _ = req('GET', '/stream/camera/%d/status' % OWN, token=tok)
check('a media token is not accepted as a login', s in (401, 422), 'HTTP %s' % s)
s, _, _ = req('GET', '/stream/camera/%d?t=%s' % (OWN, 'x' * 600))
check('oversized token -> 401', s == 401, 'HTTP %s' % s)

# Needs the server secret: forge an expired and a wrong-salt token inside the container.
forge = r'''
import sys; sys.path.insert(0, "/app")
from app import create_app
from itsdangerous import URLSafeTimedSerializer
from itsdangerous.timed import TimestampSigner
import time
app = create_app()
with app.app_context():
    key = app.config["SECRET_KEY"]
    class Old(URLSafeTimedSerializer):
        def make_signer(self, salt=None):
            s = super().make_signer(salt)
            s.get_timestamp = lambda: int(time.time()) - 120
            return s
    print(Old(key, salt="mjpeg-view-v1").dumps({"u": %d, "c": %d, "j": "x"}))
    print(URLSafeTimedSerializer(key, salt="other-salt").dumps({"u": %d, "c": %d, "j": "x"}))
''' % tuple([int(json.loads(req('GET', '/auth/me', token=jwt)[2]).get('id', 0)
                 if req('GET', '/auth/me', token=jwt)[0] == 200 else 0), OWN] * 2)
out = subprocess.run(['docker', 'compose', 'exec', '-T', 'backend', 'python', '-c', forge],
                     capture_output=True, text=True, env=dict(os.environ, MSYS_NO_PATHCONV='1'))
lines = [l for l in out.stdout.strip().splitlines() if l and ' ' not in l][-2:]
if len(lines) == 2:
    expired, wrong_salt = lines
    s, _, _ = req('GET', '/stream/camera/%d?t=%s' % (OWN, expired))
    check('token older than 60 s -> 401', s == 401, 'HTTP %s' % s)
    s, _, _ = req('GET', '/stream/camera/%d?t=%s' % (OWN, wrong_salt))
    check('token signed under another salt -> 401', s == 401, 'HTTP %s' % s)
else:
    check('forged-token cases ran', False, (out.stderr or out.stdout)[-200:])

# Logout revokes: a token issued in a session is refused once that session logs out.
jwt2 = login()
_, _, tok2 = media(jwt2, OWN)
s, _, _ = req('POST', '/auth/logout', token=jwt2)
s2, _, _ = req('GET', '/stream/camera/%d?t=%s' % (OWN, tok2))
check('logout revokes that session\'s media tokens', s2 == 401, 'logout HTTP %s, view HTTP %s' % (s, s2))

print()
print('FAILED: %s' % failures if failures else 'all media-token checks passed')
sys.exit(1 if failures else 0)
