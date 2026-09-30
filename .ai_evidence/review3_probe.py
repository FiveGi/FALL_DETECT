"""Isolated REVIEW-3 reproductions; no live DB, cameras, or production changes."""
import ast
import importlib.util
import logging
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch

import yaml
from flask import Flask
from itsdangerous import URLSafeTimedSerializer
from itsdangerous.timed import TimestampSigner

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m = load('review_media', 'app/services/media_token.py')
c = load('review_config', 'tools/check_config_coherence.py')
user = types.SimpleNamespace(id=7, is_admin=lambda: False)
camera = types.SimpleNamespace(id=14, user_id=7)
calls = []
class Query:
    def __init__(self, name, obj): self.name, self.obj = name, obj
    def get(self, key):
        calls.append(self.name)
        return self.obj
U = types.SimpleNamespace(query=Query('user', user))
C = types.SimpleNamespace(query=Query('camera', camera))
B = types.SimpleNamespace(is_jti_blacklisted=lambda j: j == 'revoked')
for path, name, value in [('user', 'User', U), ('camera', 'Camera', C),
                          ('token_blocklist', 'TokenBlocklist', B)]:
    sys.modules['app.models.' + path] = types.SimpleNamespace(**{name: value})
app = Flask(__name__)
app.config['SECRET_KEY'] = 'isolated-review-strong-key-not-a-deployment-secret'
with app.app_context():
    good = m.issue(7, 14, 'session')
    cases = [(None, 14, 401), ('x'*513, 14, 401), (good, 15, 401),
             (good[:-3]+'BAD', 14, 401), (m.issue(7, 14, 'revoked'), 14, 401),
             (URLSafeTimedSerializer(app.secret_key, salt='wrong').dumps({}), 14, 401)]
    for token, cam, expected in cases:
        calls.clear()
        assert m.verify(token, cam)[1] == expected
        assert 'camera' not in calls
    assert m.verify(good, 14)[0] is camera
    camera.user_id = 8
    assert m.verify(good, 14)[1] == 404
    camera.user_id = 7
    print('AUTH: 6 invalid/revoked cases rejected before camera lookup; valid owner accepted; transferred camera rejected (mock models).')
    with patch.object(TimestampSigner, 'get_timestamp', return_value=1000):
        boundary = m.issue(7, 14, 'session')
    for now, status in [(1060, None), (1061, 401), (999, 401)]:
        with patch.object(TimestampSigner, 'get_timestamp', return_value=now):
            assert m.verify(boundary, 14)[1] == status
    for payload in [[], {}, {'u':'bad','c':14,'j':'session'}]:
        assert m.verify(m._signer().dumps(payload),14)[1] == 401
    U.query.obj = None
    assert m.verify(good,14)[1] == 401
    U.query.obj = user
    C.query.obj = None
    assert m.verify(good,14)[1] == 404
    C.query.obj = camera
    user.is_admin = lambda: True
    camera.user_id = 8
    assert m.verify(good,14)[0] is camera
    camera.user_id = 7
    user.is_admin = lambda: False
    print('AUTH: exact age 60 accepted, 61/future rejected; 3 malformed signed payloads rejected; deleted user/camera rejected; admin accepted (9 additional assertions).')
    app.config['SECRET_KEY'] = 'dev'
    forged = URLSafeTimedSerializer('dev', salt='mjpeg-view-v1').dumps({'u':7,'c':14,'j':'invented'})
    assert m.verify(forged, 14)[0] is camera
    print('BUG default secret: forged media token with invented jti accepted (mock models, real serializer).')

tree = ast.parse((ROOT/'app/models/token_blocklist.py').read_text())
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'is_jti_blacklisted')
fn.decorator_list = []
ns = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<actual-blocklist-method>', 'exec'), ns)
class BrokenQuery:
    def filter_by(self, **kw): raise RuntimeError('isolated lookup failure')
assert ns['is_jti_blacklisted'](types.SimpleNamespace(query=BrokenQuery()), 'revoked') is False
print('BUG revocation lookup: actual method returns False on query exception (fail open).')

clock = [0]
closed = []
def delayed():
    try:
        clock[0] = 601
        yield b'late-private-frame'
    finally: closed.append(True)
with patch.object(m.time, 'monotonic', lambda: clock[0]):
    result = list(m.bounded(delayed(), seconds=600))
assert result == [b'late-private-frame'] and closed
print('BUG bounded: frame produced at t=601 is yielded despite deadline=600; underlying stream eventually closed.')

for query in ['t=probe', '%74=probe', 'token=probe']:
    record = logging.LogRecord('werkzeug', 20, '', 0, 'GET /api/stream/camera/14?'+query, (), None)
    m.RedactMediaTokens().filter(record)
    with app.test_request_context('/?'+query):
        from flask import request
        parsed = request.args.get('t') or request.args.get('token')
    print('REDACTION:', query.split('=')[0], 'Flask value=', parsed, 'log=', record.getMessage())
m.install_log_redaction()
print('FILTERS:', {n: len(logging.getLogger(n).filters) for n in ['werkzeug','gunicorn.access',app.logger.name]})

print('COMPOSE current:', c.compose_profiles())
with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    base = yaml.safe_load((ROOT/'docker-compose.yml').read_text(encoding='utf-8'))
    overlay = yaml.safe_load((ROOT/'docker-compose.gpu.yml').read_text(encoding='utf-8'))
    (root/'docker-compose.gpu.yml').write_text(yaml.safe_dump(overlay))
    env = base['services']['celery_worker']['environment']
    env[:] = ['V3_ROI_FULL_EVERY=0' if s.startswith('V3_ROI_FULL_EVERY=') else s for s in env]
    (root/'docker-compose.yml').write_text(yaml.safe_dump(base))
    with patch.object(c, 'ROOT', folder):
        p = c.compose_profiles()['cpu']
        key = c.measured_key(p['imgsz'],15,p['fps'],4,('auto',),.65,p['roi'],p['roi_every'])
        print('BUG zero cadence: checker=',p['roi_every'],'runtime max(0,1)=1; measured=', key in c.MEASURED)
        assert p['roi_every'] == 8 and key in c.MEASURED
        env[:] = [s for s in env if not s.startswith('V3_ROI_')]
        base['services']['celery_worker']['env_file'] = ['fixture.env']
        (root/'fixture.env').write_text('V3_ROI_IMGSZ=300\nV3_ROI_FULL_EVERY=1\n')
        (root/'docker-compose.yml').write_text(yaml.safe_dump(base))
        p = c.compose_profiles()['cpu']
        key = c.measured_key(p['imgsz'],15,p['fps'],4,('auto',),.65,p['roi'],p['roi_every'])
        assert p['roi'] == 0 and key in c.MEASURED
        print('BUG env_file: fixture crop=300/1 ignored; checker crop=',p['roi'],'measured=',key in c.MEASURED)
print('Dataset: synthetic control-flow/config fixtures; 0 media clips; no live verification.')
