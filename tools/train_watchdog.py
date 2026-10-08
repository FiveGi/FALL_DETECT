"""Overnight resource guard (Codex, 8 Oct): runs a command and stops it (whole process tree) when
  - available RAM < MIN_GB (default 2.0) for 2 checks in a row, or
  - the R2 soak's mean camera fps (last sample in soak_clips.txt) falls > 10 % below its first sample, or
  - the clock passes STOP_AT (default 09:30, the overnight hard stop).
Checks every 60 s; every check and the outcome are appended to LOG. Exit code: the command's, or 3 when stopped.
Usage: STOP_AT='2026-10-08 09:30' LOG=path python tools/train_watchdog.py -- <command ...>
"""
import datetime
import os
import re
import subprocess
import sys
import time

import psutil

MIN_GB = float(os.environ.get('MIN_GB', 2.0))
# An ABSOLUTE deadline (Codex, 8 Oct: an hour-only check let launches after noon through).
STOP_AT = datetime.datetime.strptime(os.environ.get('STOP_AT', '2026-10-08 09:30'), '%Y-%m-%d %H:%M')
LOG = os.environ.get('LOG', 'training/data/system_test/watchdog.log')
SOAK = 'training/data/system_test/soak_clips.txt'


def soak_fps():
    """Mean fps over cameras for every 10-min soak sample, oldest first."""
    if not os.path.exists(SOAK):
        return []
    out = []
    for line in open(SOAK):
        vals = [float(v) for v in re.findall(r'cam\d+ ([0-9.]+)', line)]
        if vals:
            out.append(sum(vals) / len(vals))
    return out


def log(msg):
    with open(LOG, 'a') as f:
        f.write('%s %s\n' % (datetime.datetime.now().strftime('%H:%M:%S'), msg))


def stop(proc, why):
    log('STOP: ' + why)
    for c in psutil.Process(proc.pid).children(recursive=True):
        c.kill()
    proc.kill()
    sys.exit(3)


def main():
    cmd = sys.argv[sys.argv.index('--') + 1:]
    # Eligibility BEFORE launch (Codex P2): >= START_GB available, soak fps not already < 90 % of its first sample,
    # and not past the hard stop. Monitoring then continues every 60 s.
    gb, fps, now = psutil.virtual_memory().available / 2 ** 30, soak_fps(), datetime.datetime.now()
    if gb < float(os.environ.get('START_GB', 3.0)):
        log('NOT STARTED: available RAM %.1f GB < START_GB' % gb); sys.exit(4)
    if len(fps) >= 2 and fps[-1] < 0.9 * fps[0]:
        log('NOT STARTED: soak fps %.2f < 90 %% of %.2f' % (fps[-1], fps[0])); sys.exit(4)
    if now >= STOP_AT:
        log('NOT STARTED: past the hard stop %s' % STOP_AT); sys.exit(4)
    proc = subprocess.Popen(cmd)
    log('start pid %d: %s' % (proc.pid, ' '.join(cmd)))
    low = 0
    while proc.poll() is None:
        time.sleep(60)
        now = datetime.datetime.now()
        if now >= STOP_AT:
            stop(proc, 'hard stop %s reached' % STOP_AT)
        gb = psutil.virtual_memory().available / 2 ** 30
        low = low + 1 if gb < MIN_GB else 0
        fps = soak_fps()
        log('check: RAM %.1f GB, soak fps first %s last %s' % (gb, fps[0] if fps else '-', fps[-1] if fps else '-'))
        if low >= 2:
            stop(proc, 'available RAM %.1f GB < %.1f twice' % (gb, MIN_GB))
        if len(fps) >= 2 and fps[-1] < 0.9 * fps[0]:
            stop(proc, 'soak fps %.2f < 90 %% of %.2f' % (fps[-1], fps[0]))
    log('finished rc %d' % proc.returncode)
    sys.exit(proc.returncode)


if __name__ == '__main__':
    main()
