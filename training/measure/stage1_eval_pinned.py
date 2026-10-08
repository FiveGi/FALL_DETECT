# -*- coding: utf-8 -*-
"""stage1_eval.py's frozen gates with every cache PINNED by path (Codex P1, 2026-10-01: automatic
discovery can mix pose models / devices / ROI variants). Same steps, same parsing:
  1. threshold chosen on pooled URFD half A of the 8 day caches (or --fixed);
  2. URFD half-B falls /28, ADL clean /20, GMDCSA val ADL clean /16 (means over the day caches);
  3. night: each IR cache -> falls /60, ADL false alarms /40;
  4. owner: each owner phase cache replayed at the threshold, scored by score_incidents.py.
The classifier is model_dir (+ V3_ENSEMBLE from the environment, as stage1_eval). The pose model is
whatever built the caches -- named in --pose for the record only.

Usage: python training/measure/stage1_eval_pinned.py --name N --model DIR --pose TAG \
          --day C1 .. C8 --ir I1 I2 I3 --owner O1 .. O8 [--fixed 0.65]
Appends a row to training/data/stage1/results_pinned.jsonl.
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
PY = sys.executable
OUT = os.path.join(ROOT, 'training', 'data', 'stage1')


def run(cmd, env=None):
    e = dict(os.environ, PYTHONIOENCODING='utf-8', **(env or {}))
    r = subprocess.run(cmd, capture_output=True, text=True, env=e, encoding='utf-8')
    if r.returncode != 0:
        # Codex review 2026-10-02 P1: a failed step must never let a stale file be scored.
        raise SystemExit('step failed (%d): %s\n%s' % (r.returncode, ' '.join(cmd[:3]), r.stderr[-2000:]))
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--pose', required=True)
    ap.add_argument('--day', nargs='+', required=True)
    ap.add_argument('--ir', nargs='+', required=True)
    ap.add_argument('--owner', nargs='+', required=True)
    ap.add_argument('--fixed', type=float)
    a = ap.parse_args()
    args = [PY, 'training/measure/eval_candidate.py', '--pool'] + a.day + ['--ir'] + a.ir + \
           ['--model', '%s=%s' % (a.name, a.model)]
    if a.fixed:
        args += ['--fixed-threshold', str(a.fixed)]
    txt = run(args)
    open(os.path.join(OUT, '%s_pinned_pool.txt' % a.name), 'w', encoding='utf-8').write(txt)
    thr = float(a.fixed) if a.fixed else float(re.search(r'threshold ([\d.]+) chosen', txt).group(1))
    mean = lambda what: float(re.search(r'mean %s\s+([\d.]+)' % what, txt).group(1))
    night = [(int(x), int(y)) for x, y in re.findall(r'NIGHT \S+\s+falls (\d+)/60\s+ADL false alarms (\d+)/40', txt)]
    row = {'name': a.name, 'model_dir': a.model, 'pose': a.pose, 'ensemble': os.environ.get('V3_ENSEMBLE', ''),
           'threshold': thr, 'fixed': bool(a.fixed), 'n_day': len(a.day), 'n_owner': len(a.owner),
           'halfB_falls': mean('half B falls'), 'halfB_adl_clean': mean('half B ADL clean'),
           'val_adl_clean': mean('val ADL clean'),
           'night_falls': [x for x, _ in night], 'night_fa': [y for _, y in night]}
    # People cap the replays applied (V3_NUM_POSES, read by v3 in each child process). Only when
    # not the default, so rows written before this stay comparable field for field.
    if os.environ.get('V3_NUM_POSES', '4') != '4':
        row['num_poses'] = int(os.environ['V3_NUM_POSES'])
    files = []
    for p, c in enumerate(a.owner):
        f = os.path.join(OUT, '%s_pinned_owner%d.json' % (a.name, p))
        if os.path.exists(f):
            os.remove(f)
        run([PY, 'training/measure/replay_owner_segments.py', c, a.model, f], {'V3_THRESHOLD': str(thr)})
        files.append(f)
    sc = run([PY, 'training/measure/score_incidents.py'] + files)
    caught, fa, multi = [], [], []
    for line in sc.splitlines()[1:]:
        m = re.search(r'(\d+)/75\s+(\d+)\s+(\d+)/25', line)
        if m:
            caught.append(int(m.group(1))); fa.append(int(m.group(2))); multi.append(int(m.group(3)))
    if len(caught) != len(a.owner):
        raise SystemExit('scorer returned %d owner rows for %d caches' % (len(caught), len(a.owner)))
    row.update(owner_caught=caught, owner_fa=fa, owner_multi=multi)
    with open(os.path.join(OUT, 'results_pinned.jsonl'), 'a') as fh:
        fh.write(json.dumps(row) + '\n')
    print(json.dumps(row))


if __name__ == '__main__':
    sys.exit(main())
