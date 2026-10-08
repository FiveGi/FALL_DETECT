# -*- coding: utf-8 -*-
"""One table for the overnight pose-model / preprocessing screen (CPU caches, deployed profile):
per model, URFD fall presence (descent / 2 s after, half A and B) per condition, and the deployed
classifier at its fixed threshold 0.65 (eval_candidate.py): half-B falls /28, ADL clean /20, and per
night cache falls /60 + ADL false alarms /40. Conditions found by cache key; a model missing a
condition prints '-' for it rather than borrowing anything.
Usage: python training/measure/pose_screen_report.py tag [tag ...]   (tag 'stock' = deployed pose)"""
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, 'training/measure')
sys.path.insert(0, 'training')
from fall_presence import presence  # noqa: E402
from urfd_split import half  # noqa: E402

CPU = 'training/data/pose_ir_eval_cpu'
STOCK_CPU = {('day', 0): 'training/data/pose_cache/d647cd5294c3', ('day', 4): 'training/data/pose_cache/f0f66ba29641',
             ('ir', 0): 'training/data/pose_cache/6b574721a26e', ('ir', 7): 'training/data/pose_cache/93920df119fb',
             ('ir', 13): 'training/data/pose_cache/14f1ad71eb27'}


def caches(tag):
    out = dict(STOCK_CPU) if tag == 'stock' else {}
    for k in glob.glob(os.path.join(CPU, tag, '*', 'key.json')):
        key = json.load(open(k))
        d = os.path.dirname(k).replace(os.sep, '/')
        if len(glob.glob(d + '/*.npz')) < 220:
            continue                                   # still building
        if key.get('night_alt'):
            out[('alt', key['simulation_seed'])] = d
        elif key['simulate_ir']:
            out[('ir', key['simulation_seed'])] = d
        else:
            out[('day', key['roi_phase'])] = d
    return out


def pres(d):
    r = presence(d)
    f = lambda h, i: sum(v[i] for s, v in r.items() if half(s + '-cam0.mp4') == h) / max(
        1, sum(1 for s in r if half(s + '-cam0.mp4') == h))
    return '%.2f/%.2f' % (f('B', 0), f('B', 1))


for tag in sys.argv[1:]:
    c = caches(tag)
    print('=== %s  (%d caches: %s)' % (tag, len(c), ' '.join('%s%d' % k for k in sorted(c))))
    print('  presence half B descent/after: ' + '  '.join('%s%d %s' % (k[0], k[1], pres(c[k])) for k in sorted(c)))
    day = [c[k] for k in sorted(c) if k[0] == 'day']
    night = [c[k] for k in sorted(c) if k[0] != 'day']
    if not day:
        continue
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    txt = subprocess.run([sys.executable, 'training/measure/eval_candidate.py', '--pool'] + day +
                         (['--ir'] + night if night else []) + ['--model', 'deployed=models', '--fixed-threshold', '0.65'],
                         capture_output=True, text=True, env=env, encoding='utf-8').stdout
    m = re.search(r'mean half B falls\s+([\d.]+).*?mean half B ADL clean\s+([\d.]+)', txt, re.S)
    print('  classifier @0.65: half B falls %s/28  ADL clean %s/20' % (m.group(1), m.group(2)) if m else txt[-500:])
    names = {v: '%s%d' % k for k, v in c.items()}
    for d, f, a in re.findall(r'NIGHT (\S+)\s+falls (\d+)/60\s+ADL false alarms (\d+)/40', txt):
        full = [n for v, n in names.items() if v.endswith(d)]
        print('    %-6s falls %s/60  FA %s/40' % (full[0] if full else d, f, a))
