# -*- coding: utf-8 -*-
"""Matched false-alarm comparison for Stage 1 models -- targets pre-registered in AI_HANDOFF.md
(2026-10-01 "PRE-REGISTERED" entry) before any extended-grid number was seen.

For each model: eval_candidate.py --curve over the extended grid (0.35..0.99) on the phase-0 day
cache and the night caches (seeds 0/7/13); then, with no threshold chosen for deployment:
  - day falls /60 at the highest-recall grid point with URFD ADL false alarms <= 3/40 and <= 6/40
    (val ADL printed by the curve but never used: the deployed model trained on 13 of those 16);
  - night mean falls /60 at the highest-recall grid point with mean night false alarms <= 4/40.
"not attainable" when no grid point meets a target; nothing is interpolated.

Usage: python training/measure/stage1_matched.py name=dir [name=dir ...]
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
DAY = 'training/data/pose_cache/d647cd5294c3'          # conf 0.30, crop256, phase 0, reset per clip
NIGHT = ['training/data/pose_cache/6b574721a26e', 'training/data/pose_cache/93920df119fb',
         'training/data/pose_cache/14f1ad71eb27']         # same, simulated IR seeds 0 / 7 / 13
ROW = re.compile(r'^\s+([\d.]+)\s+(\d+)\s+(\d+) \((\d+)\+(\d+)\)\s+(\d+) (\d+) (\d+)\s+(\d+) (\d+) (\d+)')


def curve(name, d):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', THRESHOLD_GRID='extended',
               V3_POSE_MODEL=os.path.join(ROOT, 'models', 'yolo26s-pose.pt'))
    out = subprocess.run([sys.executable, 'training/measure/eval_candidate.py', '--day', DAY, '--ir'] + NIGHT +
                         ['--model', '%s=%s' % (name, d), '--curve'], capture_output=True, text=True,
                         env=env, encoding='utf-8').stdout
    rows = []
    for line in out.splitlines():
        m = ROW.match(line)
        if m:
            v = [float(m.group(1))] + [int(x) for x in m.groups()[1:]]
            rows.append({'thr': v[0], 'falls': v[1], 'urfd_fa': v[3], 'val_fa': v[4],
                         'n_falls': v[5:8], 'n_fa': v[8:11]})
    return rows, out


def best(rows, ok, key):
    c = [r for r in rows if ok(r)]
    return max(c, key=key) if c else None


def main(specs):
    print('%-14s %-22s %-22s %-26s' % ('model', 'day falls @URFD FA<=3', 'day falls @URFD FA<=6',
                                         'night falls @night FA<=4'))
    os.makedirs('training/data/stage1', exist_ok=True)
    for spec in specs:
        name, d = spec.split('=', 1)
        rows, raw = curve(name, d)
        open('training/data/stage1/%s_curve_g2.txt' % name, 'w', encoding='utf-8').write(raw)
        cells = []
        for target in (3, 6):
            b = best(rows, lambda r: r['urfd_fa'] <= target, lambda r: (r['falls'], -r['thr']))
            cells.append('%d (thr %.3f, FA %d)' % (b['falls'], b['thr'], b['urfd_fa']) if b else 'not attainable')
        b = best(rows, lambda r: sum(r['n_fa']) / 3.0 <= 4, lambda r: (sum(r['n_falls']), -r['thr']))
        cells.append('%.1f (thr %.3f, FA %.1f)' % (sum(b['n_falls']) / 3.0, b['thr'], sum(b['n_fa']) / 3.0)
                     if b else 'not attainable')
        print('%-14s %-22s %-22s %-26s' % (name, cells[0], cells[1], cells[2]), flush=True)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
