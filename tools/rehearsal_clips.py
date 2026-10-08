"""R3 (overnight plan, agreed 8 Oct): docs/runbook_9oct.md section 9 run as written, on the dev stack, LINE off.

One camera at a time, added through the same API the web form uses (file source /app/Test/<n>.mp4):
  14, 15, 16  real falls  -> a fall alert expected
  13          real fall   -> known miss (silence is the documented result)
  17          crowd, NO fall -> no fall alert expected
For each: start, watch WATCH_S seconds, record fall alerts (count, seconds to the first one, image path), then
ACKNOWLEDGE every alert of that camera through /api/detection-logs/notifications/<id>/acknowledge, then stop and delete
the camera -- the runbook's ack-then-stop rule. Clip 14 is run a SECOND time and deliberately NOT acknowledged for
UNACK_S seconds: the escalation sweep must look at it (with LINE off it logs 'not sent' and does not count it), then it
is acknowledged. Output: training/data/system_test/rehearsal_clips.txt
Usage: python tools/rehearsal_clips.py   (needs the dev stack up; nothing else running on it)
"""
import datetime
import os
import subprocess
import sys
import time

import requests

API = os.environ.get('API', 'http://localhost:8932')
WATCH_S = int(os.environ.get('WATCH_S', 120))
UNACK_S = int(os.environ.get('UNACK_S', 300))
OUT = 'training/data/system_test/rehearsal_clips.txt'
EXPECT = {'14': 'alert', '15': 'alert', '16': 'alert', '13': 'known miss (silence ok)', '17': 'NO alert'}


def login():
    r = requests.post(API + '/api/auth/login', json={'username': 'admin', 'password': 'admin123'}, timeout=15)
    r.raise_for_status()
    return {'Authorization': 'Bearer ' + r.json()['access_token']}


def notifications(h, cam_id):
    r = requests.get(API + '/api/detection-logs/notifications', headers=h, timeout=15).json()
    rows = r if isinstance(r, list) else r.get('notifications', r.get('data', []))
    return [n for n in rows if n.get('camera_id') == cam_id]


def write(line):
    print(line, flush=True)
    with open(OUT, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def scenario(h, clip, ack=True):
    r = requests.post(API + '/api/cameras', headers=h, timeout=15, json={
        'name': 'R3 clip%s' % clip, 'room_name': 'rehearsal', 'url': '/app/Test/%s.mp4' % clip,
        'detection_type': 'fall_v2', 'alert_start_time': '00:00', 'alert_end_time': '23:59'})
    cam = r.json().get('id') or r.json()['camera']['id']
    try:
        _run(h, clip, cam, ack)
    finally:   # Gemini: never leave a rehearsal camera running if anything above fails
        requests.post(API + '/api/cameras/%s/stop' % cam, headers=h, timeout=30)
        requests.delete(API + '/api/cameras/%s' % cam, headers=h, timeout=30)
    time.sleep(10)


def _run(h, clip, cam, ack):
    t0 = time.time()
    requests.post(API + '/api/cameras/%s/start' % cam, headers=h, timeout=30)
    time.sleep(WATCH_S)
    rows = [n for n in notifications(h, cam) if 'fall' in (n.get('detection_type') or '')]
    first = None
    if rows:
        # sent_at is naive LOCAL time (the stack runs TZ=Asia/Bangkok, same as this host; checked 8 Oct: no 'Z', no
        # offset), so it is compared with the host's naive local clock. Fail loudly if that ever changes.
        ts = sorted(n['sent_at'] for n in rows)[0]
        if ts.endswith('Z') or '+' in ts[10:]:
            raise SystemExit('sent_at carries a zone now (%s) -- fix the latency arithmetic' % ts)
        first = round((datetime.datetime.fromisoformat(ts) - datetime.datetime.fromtimestamp(t0)).total_seconds(), 1)
    ok = (len(rows) > 0) if EXPECT[clip] == 'alert' else (len(rows) == 0 if clip == '17' else True)
    write('clip %s expect %-24s fall alerts %d first %ss images %s -> %s%s' % (
        clip, EXPECT[clip], len(rows), first, [os.path.basename(n.get('image_path') or '') for n in rows][:2],
        'OK' if ok else 'UNEXPECTED', '' if ack else '  (deliberately NOT acknowledged)'))
    if not ack:
        time.sleep(UNACK_S)
        sweep = subprocess.run(['docker', 'logs', '--since', '%ds' % (UNACK_S + 30),
                                'backend-elderly-surveillance-main-celery_maintenance-1'],
                               capture_output=True, text=True, encoding='utf-8', errors='replace')
        lines = [l for l in (sweep.stdout + sweep.stderr).splitlines() if 'scalat' in l or 'LINE' in l]
        write('   escalation log lines in %d s unacknowledged: %d; last: %s' % (UNACK_S, len(lines), lines[-1][-160:] if lines else '-'))
    acked = 0
    for n in notifications(h, cam):
        if not n.get('acknowledged_at'):
            acked += requests.post(API + '/api/detection-logs/notifications/%s/acknowledge' % n['id'], headers=h,
                                   timeout=15).status_code == 200
    left = [n for n in notifications(h, cam) if not n.get('acknowledged_at')]
    write('   acknowledged %d, unacknowledged left %d (camera is stopped + deleted next)' % (acked, len(left)))


def main():
    h = login()
    write('=== R3 clip rehearsal %s (watch %d s per clip) ===' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), WATCH_S))
    for clip in ('14', '15', '16', '13', '17'):
        scenario(h, clip)
        h = login()
    scenario(h, '14', ack=False)
    write('=== R3 done %s ===' % datetime.datetime.now().strftime('%H:%M'))


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
