"""9 Oct system-test rehearsal driver (plan S2/S3/S5): add simulated RTSP cameras through the real
API, start detection, and report what the running system did.

Usage:
  python tools/system_test.py add N [rtsp_base]      add + start cameras cam1..camN (default base
                                                     rtsp://host.docker.internal:8554/cam)
  python tools/system_test.py clips 14 15 ...        add + start one camera per Test/ clip ("ST clip14", ...),
                                                     the same /app/Test/<n>.mp4 source the web dropdown sets
  python tools/system_test.py status                 cameras, recent alerts, worker fps lines
  python tools/system_test.py remove                 stop + delete the cameras this tool added
Env: API (default http://localhost:8932), ST_USER/ST_PASSWORD (default admin/admin123).
Cameras are named "ST camK" / "ST clipN" so they can be found and removed again.
"""
import os
import subprocess
import sys

import requests

API = os.environ.get('API', 'http://localhost:8932')


def login():
    r = requests.post(API + '/api/auth/login', json={'username': os.environ.get('ST_USER', 'admin'),
                                                     'password': os.environ.get('ST_PASSWORD', 'admin123')}, timeout=15)
    r.raise_for_status()
    return {'Authorization': 'Bearer ' + r.json()['access_token']}


def cameras(h):
    r = requests.get(API + '/api/cameras', headers=h, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else data.get('cameras', data.get('data', []))


def add(n, base):
    _add([('ST cam%d' % k, '%s%d' % (base, k)) for k in range(1, n + 1)])


def _add(named_urls):
    h = login()
    have = {c['name']: c for c in cameras(h)}
    for name, url in named_urls:
        if name not in have:
            r = requests.post(API + '/api/cameras', headers=h, timeout=15, json={
                'name': name, 'room_name': 'rehearsal', 'url': url,
                'detection_type': 'fall_v2', 'alert_start_time': '00:00', 'alert_end_time': '23:59'})
            print(name, 'create', r.status_code, r.text[:120])
            cam_id = r.json().get('id') or r.json().get('camera', {}).get('id')
        else:
            cam_id = have[name]['id']
        r = requests.post(API + '/api/cameras/%s/start' % cam_id, headers=h, timeout=30)
        print(name, 'start', r.status_code, r.text[:120])


def status():
    h = login()
    for c in cameras(h):
        if c['name'].startswith(('ST cam', 'ST clip', 'ST empty')):
            s = requests.get(API + '/api/cameras/%s/status' % c['id'], headers=h, timeout=15).json()
            print(c['id'], c['name'], c.get('url'), '| active', c.get('is_active'), '|', str(s)[:160])
    r = requests.get(API + '/api/detection-logs/notifications', headers=h, timeout=15).json()
    rows = r if isinstance(r, list) else r.get('notifications', r.get('data', []))
    print('recent notifications:', len(rows))
    for n in rows[:8]:
        print('  ', n.get('id'), n.get('camera_id'), n.get('sent_at'), n.get('detection_type'),
              'ack' if n.get('acknowledged_at') else 'unack', 'still_down', n.get('still_down_seconds'))
    log = subprocess.run(['docker', 'logs', '--since', '3m', 'backend-elderly-surveillance-main-celery_worker-1'],
                         capture_output=True, text=True, encoding='utf-8', errors='replace')
    lines = (log.stdout + log.stderr).splitlines()
    keep = [l for l in lines if any(k in l for k in ('model identity', 'Detection rate', 'FALL ALERT', 'STILL DOWN',
                                                     'reconnect', 'tracks reset', 'still-down', 'Error', 'Traceback'))]
    print('worker log (3 min):'); print('\n'.join('   ' + l[-220:] for l in keep[-25:]))


def remove():
    h = login()
    for c in cameras(h):
        if c['name'].startswith(('ST cam', 'ST clip', 'ST empty')):
            requests.post(API + '/api/cameras/%s/stop' % c['id'], headers=h, timeout=30)
            r = requests.delete(API + '/api/cameras/%s' % c['id'], headers=h, timeout=30)
            print('removed', c['name'], r.status_code)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'add':
        add(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else 'rtsp://host.docker.internal:8554/cam')
    elif cmd == 'clips':
        _add([('ST clip%s' % n, '/app/Test/%s.mp4' % n) for n in sys.argv[2:]])
    elif cmd == 'status':
        status()
    elif cmd == 'remove':
        remove()
    else:
        print(__doc__)
