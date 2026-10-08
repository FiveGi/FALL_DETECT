"""F1 (9 Oct system test plan): network cameras must fail fast and come back, never hang the loop.

Needs the simulator: tools/rtsp_sim.sh up 1 h264 720   (rtsp://localhost:8554/cam1)
Run:  python tests/test_rtsp_robustness.py
"""
import os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('STREAM_TIMEOUT_MS', '5000')
import importlib.util
spec = importlib.util.spec_from_file_location('cm_open', 'app/services/camera_manager.py')
# camera_manager imports the Flask app; only the two helpers are needed, so load them from source.
src = open('app/services/camera_manager.py', encoding='utf-8').read()
ns = {}
start = src.index('STREAM_TIMEOUT_MS = ')
end = src.index('@dataclass', start)
exec('import os, cv2, threading, time\n' + src[start:end], ns)
open_capture, is_network_source = ns['open_capture'], ns['is_network_source']
SIM = 'tools/rtsp_sim.sh'
URL = os.environ.get('TEST_URL', 'rtsp://localhost:8554/cam1')
PUB = os.environ.get('TEST_PUB', 'rtsp-pub1')
SRV = os.environ.get('TEST_SRV', 'rtsp-server')
failures = []

def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + (' -- ' + detail if detail else ''), flush=True)
    if not cond:
        failures.append(name)

# 1. unreachable host/port: open must give up within the timeout, not hang
t = time.time(); cap = open_capture('rtsp://127.0.0.1:8599/none'); dt = time.time() - t
check('unreachable camera fails fast', not cap.isOpened() and dt < 12, '%.1fs' % dt)
# 2. server up but no such stream: fails fast
t = time.time(); cap = open_capture(URL.rsplit('/', 1)[0] + '/does_not_exist'); dt = time.time() - t
check('missing stream fails fast', not cap.isOpened() and dt < 12, '%.1fs' % dt)
# 3. live stream reads frames
cap = open_capture(URL)
ok = cap.isOpened() and sum(cap.read()[0] for _ in range(25)) >= 20
check('live stream opens and reads', ok)
# 4a. camera FREEZES (connection open, no data): read must time out, not block forever
cap = open_capture(URL); cap.read()
subprocess.run(['docker', 'pause', PUB], check=True, capture_output=True)
t = time.time(); got = True
while got and time.time() - t < 40:
    got = cap.read()[0]
dt = time.time() - t
check('frozen camera -> read times out (no hang)', not got and dt < 20, '%.1fs' % dt)
# release() while the camera is still frozen (reader may be inside a blocking grab): must return,
# must not crash; the reader releases the capture itself when its grab ends (Codex P1)
t = time.time(); cap.release(); dt = time.time() - t
check('release while frozen returns without crash', dt < 20, '%.1fs' % dt)
subprocess.run(['docker', 'unpause', PUB], check=True, capture_output=True)
time.sleep(3)
cap = open_capture(URL)
# 4. camera stops sending mid-stream: read must return False within ~timeout (no hang)
subprocess.run(['docker', 'stop', PUB], check=True, capture_output=True)
t = time.time(); got = True
while got and time.time() - t < 30:
    got = cap.read()[0]
dt = time.time() - t
check('stalled camera -> read fails within timeout', not got and dt < 15, '%.1fs' % dt)
cap.release()
# 5. camera comes back: a new open succeeds
subprocess.run(['docker', 'start', PUB], check=True, capture_output=True)
t = time.time(); cap = None
while time.time() - t < 30:
    cap = open_capture(URL)
    if cap.isOpened() and cap.read()[0]:
        break
    time.sleep(1)
check('camera back -> reopen works', cap is not None and cap.isOpened(), '%.1fs' % (time.time() - t))
# 7. backpressure: read at the loop's 8 fps for 180 s from a 25 fps stream -- must not drop the
#    session (RTSP server write timeout) and frames must stay current (no growing backlog)
cap.release(); time.sleep(12)   # let earlier sessions close before measuring
since = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
cap = open_capture(URL)
t0 = time.time(); fails = 0; n = 0
while time.time() - t0 < 180:
    ok, f = cap.read()
    if not ok:
        fails += 1; cap.release(); cap = open_capture(URL); continue
    n += 1; time.sleep(0.125)
srv = subprocess.run(['docker', 'logs', '--since', since, SRV], capture_output=True, text=True).stdout +       subprocess.run(['docker', 'logs', '--since', since, SRV], capture_output=True, text=True).stderr
timeouts = srv.count('i/o timeout')
check('8 fps reader on a 25 fps stream: no drops for 180 s', fails == 0 and timeouts == 0,
      'reads %d, read failures %d, server write timeouts %d' % (n, fails, timeouts))
cap.release()
check('is_network_source', is_network_source('rtsp://x') and not is_network_source('/app/Test/1.mp4'))
print('RESULT', 'FAIL' if failures else 'PASS', failures)
sys.exit(1 if failures else 0)
