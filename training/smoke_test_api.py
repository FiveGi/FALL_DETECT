"""Call every API endpoint against a running system and report what actually works.

Existing tests here all measure the detection model. Nothing exercised the API surface, so a
route could be broken -- wrong status, 500, missing field -- and the only way to find out was
a user hitting it. This logs in as admin, calls every GET endpoint, and exercises the
non-destructive POSTs, checking status codes and the shape of what comes back.

Deliberately does NOT call anything that would send a message to a real person (the LINE test
endpoint) or delete data -- those are listed as skipped rather than silently omitted.

Usage:
    python training/smoke_test_api.py            # against localhost:8932
    API=http://host:8932 python training/smoke_test_api.py
"""
import json
import os
import sys

import requests

API = os.environ.get('API', 'http://localhost:8932')
USER = os.environ.get('SMOKE_USER', 'admin')
PASSWORD = os.environ.get('SMOKE_PASSWORD', 'admin123')
TIMEOUT = float(os.environ.get('SMOKE_TIMEOUT', 30))

# Endpoints that reach a real person or destroy data. Named explicitly so the report says
# what was not covered instead of quietly leaving gaps.
SKIPPED = {
    '/api/line/test': 'would push a real LINE message',
    '/api/line/webhook': 'called by LINE, needs a signed body',
    '/api/auth/logout': 'would invalidate the token this run is using',
    '/api/auth/logout-all': 'would invalidate every session',
    'DELETE /api/cameras/<id>': 'destructive',
    'DELETE /api/thai-frat/assessments/<id>': 'destructive',
}

results = []


def record(name, ok, detail):
    results.append((ok, name, detail))
    print(f'{"PASS" if ok else "FAIL"}  {name:58} {detail}', flush=True)


def call(method, path, token=None, **kw):
    headers = kw.pop('headers', {})
    if token:
        headers['Authorization'] = f'Bearer {token}'
    try:
        r = requests.request(method, API + path, headers=headers, timeout=TIMEOUT, **kw)
        return r
    except Exception as e:
        return e


def check(name, method, path, token, expect=(200,), want_keys=None, **kw):
    r = call(method, path, token, **kw)
    if isinstance(r, Exception):
        record(name, False, f'request error: {str(r)[:70]}')
        return None
    if r.status_code not in expect:
        record(name, False, f'status {r.status_code} (expected {expect}) {r.text[:70]}')
        return None
    body = None
    if r.headers.get('content-type', '').startswith('application/json'):
        try:
            body = r.json()
        except Exception:
            record(name, False, 'status ok but body is not valid JSON')
            return None
    if want_keys and isinstance(body, dict):
        missing = [k for k in want_keys if k not in body]
        if missing:
            record(name, False, f'status ok but missing keys {missing}')
            return body
    size = len(body) if isinstance(body, (list, dict)) else len(r.content)
    record(name, True, f'{r.status_code}, {size} items/keys')
    return body


def main():
    print(f'API: {API}\n')

    r = call('POST', '/api/auth/login', json={'username': USER, 'password': PASSWORD})
    if isinstance(r, Exception) or r.status_code != 200:
        print('cannot log in -- is the stack running?', r)
        return 1
    token = r.json()['access_token']
    record('POST /api/auth/login', True, '200, token received')

    check('GET  /api/auth/verify', 'GET', '/api/auth/verify', token)
    check('GET  /api/health', 'GET', '/api/health', token)
    check('GET  /api/health/camera-detection-service', 'GET',
          '/api/health/camera-detection-service', token)

    cameras = check('GET  /api/cameras', 'GET', '/api/cameras', token)
    cam_id = None
    if isinstance(cameras, list) and cameras:
        cam_id = cameras[0].get('id')
    elif isinstance(cameras, dict):
        items = cameras.get('cameras') or []
        cam_id = items[0]['id'] if items else None

    check('GET  /api/cameras/test-videos', 'GET', '/api/cameras/test-videos', token)
    if cam_id:
        check(f'GET  /api/cameras/{cam_id}', 'GET', f'/api/cameras/{cam_id}', token)
        check(f'GET  /api/cameras/{cam_id}/status', 'GET', f'/api/cameras/{cam_id}/status', token)
        check(f'GET  /api/stream/camera/{cam_id}/status', 'GET',
              f'/api/stream/camera/{cam_id}/status', token)
        check(f'GET  /api/alerts/camera/{cam_id}/history', 'GET',
              f'/api/alerts/camera/{cam_id}/history', token)
        check(f'GET  /api/alerts/camera/{cam_id}/statistics', 'GET',
              f'/api/alerts/camera/{cam_id}/statistics', token)
        check(f'GET  /api/detection-logs/notifications/{cam_id}', 'GET',
              f'/api/detection-logs/notifications/{cam_id}', token)
    else:
        record('camera-scoped endpoints', False, 'no camera exists to test against')

    check('GET  /api/alerts/user/summary', 'GET', '/api/alerts/user/summary', token)
    check('GET  /api/detection-logs', 'GET', '/api/detection-logs', token)

    notifications = check('GET  /api/detection-logs/notifications', 'GET',
                          '/api/detection-logs/notifications', token)
    if isinstance(notifications, list) and notifications:
        expected = {'id', 'camera_id', 'sent_at', 'detection_type', 'image_path',
                    'confidence', 'acknowledged_at', 'escalation_count', 'clip_path'}
        missing = expected - set(notifications[0])
        record('     notification payload shape', not missing,
               'all fields present' if not missing else f'missing {sorted(missing)}')

    check('GET  /api/system-logs', 'GET', '/api/system-logs?per_page=5', token,
          want_keys=['logs', 'total', 'pages', 'current_page'])
    check('GET  /api/system-logs/levels', 'GET', '/api/system-logs/levels', token)
    check('GET  /api/system-logs/components', 'GET', '/api/system-logs/components', token)

    check('GET  /api/admin/dashboard', 'GET', '/api/admin/dashboard', token)
    check('GET  /api/admin/users', 'GET', '/api/admin/users', token)

    line = check('GET  /api/line/settings', 'GET', '/api/line/settings', token)
    if isinstance(line, dict) and isinstance(line.get('data'), dict):
        d = line['data']
        # The token must never reach the browser; only a masked tail and a boolean.
        leaked = 'channel_access_token' in d
        record('     LINE settings hides the token', not leaked,
               'masked' if not leaked else 'TOKEN LEAKED IN RESPONSE')

    check('GET  /api/thai-frat/question-options', 'GET', '/api/thai-frat/question-options', token)
    check('GET  /api/thai-frat/assessments', 'GET', '/api/thai-frat/assessments', token,
          expect=(200, 404))

    check('GET  /api/stream/stats', 'GET', '/api/stream/stats', token)

    # Authentication must actually be enforced, not just present.
    r = call('GET', '/api/detection-logs/notifications')
    record('     unauthenticated request is rejected',
           not isinstance(r, Exception) and r.status_code in (401, 422),
           f'status {getattr(r, "status_code", r)}')

    r = call('POST', '/api/line/webhook', json={'events': []})
    record('     LINE webhook rejects an unsigned body',
           not isinstance(r, Exception) and r.status_code == 403,
           f'status {getattr(r, "status_code", r)}')

    print('\nskipped on purpose:')
    for name, why in SKIPPED.items():
        print(f'   {name:46} {why}')

    failed = [n for ok, n, _ in results if not ok]
    print(f'\n{len(results) - len(failed)}/{len(results)} checks passed')
    if failed:
        print('failing:')
        for n in failed:
            print('   ' + n)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
