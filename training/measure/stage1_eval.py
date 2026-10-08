# -*- coding: utf-8 -*-
"""One classifier through every frozen gate (AI_HANDOFF.md, 2026-10-01 FROZEN entry), as a row.

  1. threshold: chosen on pooled URFD half-A over the 8 conf-0.30 phase caches (eval_candidate
     --pool), never on anything reported below;
  2. day: URFD half-B falls /28, half-B ADL clean /20, GMDCSA24-val ADL clean /16 (8-phase means);
  3. night: simulated IR (conf 0.30 caches, seeds 0/7/13) falls /60 and ADL false alarms /40;
  4. owner segments: replayed over the 8 cached phases at the chosen threshold, scored by
     score_incidents.py -- REPORTED, never used to choose.
Pose front-end fixed for every model: 320 px, crop 256 every 8, conf 0.30, 8 fps.

Usage: python training/measure/stage1_eval.py <name> <model_dir> [<fixed_threshold>]
Appends to training/data/stage1/results.jsonl.
"""
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
PY = sys.executable
POSE = os.path.join(ROOT, 'models', 'yolo26s-pose.pt')
OUT = os.path.join(ROOT, 'training', 'data', 'stage1')


def day_caches():
    v = {}
    for k in glob.glob('training/data/pose_cache/*/key.json'):
        key = json.load(open(k))
        if (key.get('roi_state') == 'reset-per-clip' and not key['simulate_ir']
                and key['pose_conf'] == 0.3 and key['roi_imgsz'] == 256):
            v[key['roi_phase']] = os.path.dirname(k).replace(os.sep, '/')
    assert sorted(v) == list(range(8)), 'need 8 phase caches, have %s' % sorted(v)
    return [v[p] for p in range(8)]


def ir_caches():
    v = {}
    for k in glob.glob('training/data/pose_cache/*/key.json'):
        key = json.load(open(k))
        if (key.get('roi_state') == 'reset-per-clip' and key['simulate_ir']
                and key['pose_conf'] == 0.3 and key['roi_imgsz'] == 256):
            v[key['simulation_seed']] = os.path.dirname(k).replace(os.sep, '/')
    assert sorted(v) == [0, 7, 13], 'need IR seeds 0/7/13, have %s' % sorted(v)
    return [v[s] for s in (0, 7, 13)]


def owner_caches():
    v = {}
    for k in glob.glob('training/data/pose_cache_owner/*/key.json'):
        key = json.load(open(k))
        if (key['pose_conf'] == 0.3 and 'pose_model' not in key
                and os.path.exists(os.path.join(os.path.dirname(k), 'done'))):
            v[key['roi_phase']] = os.path.dirname(k)
    assert sorted(v) == list(range(8)), 'need 8 owner phase caches, have %s' % sorted(v)
    return [v[p] for p in range(8)]


def run(cmd, env=None):
    e = dict(os.environ, PYTHONIOENCODING='utf-8', V3_POSE_MODEL=POSE, **(env or {}))
    return subprocess.run(cmd, capture_output=True, text=True, env=e, encoding='utf-8').stdout


def main(name, model_dir, fixed=None):
    os.makedirs(OUT, exist_ok=True)
    args = [PY, 'training/measure/eval_candidate.py', '--pool'] + day_caches() + ['--ir'] + ir_caches() + \
           ['--model', '%s=%s' % (name, model_dir)]
    if fixed:
        args += ['--fixed-threshold', str(fixed)]
    txt = run(args)
    open(os.path.join(OUT, '%s_pool.txt' % name), 'w', encoding='utf-8').write(txt)
    thr = float(re.search(r'threshold ([\d.]+) chosen', txt).group(1))
    if fixed:
        thr = float(fixed)
    mean = lambda what: float(re.search(r'mean %s\s+([\d.]+)' % what, txt).group(1))
    night = [(int(a), int(b)) for a, b in re.findall(r'NIGHT \S+\s+falls (\d+)/60\s+ADL false alarms (\d+)/40', txt)]
    row = {'name': name, 'model_dir': model_dir, 'threshold': thr,
           'halfB_falls': mean('half B falls'), 'halfB_adl_clean': mean('half B ADL clean'),
           'val_adl_clean': mean('val ADL clean'),
           'night_falls': [a for a, _ in night], 'night_fa': [b for _, b in night]}
    files = []
    for p, c in enumerate(owner_caches()):
        f = os.path.join(OUT, '%s_owner_phase%d.json' % (name, p))
        run([PY, 'training/measure/replay_owner_segments.py', c, model_dir, f], {'V3_THRESHOLD': str(thr)})
        files.append(f)
    sc = run([PY, 'training/measure/score_incidents.py'] + files)
    caught, fa, multi = [], [], []
    for line in sc.splitlines()[1:]:
        m = re.search(r'(\d+)/75\s+(\d+)\s+(\d+)/25', line)
        if m:
            caught.append(int(m.group(1))); fa.append(int(m.group(2))); multi.append(int(m.group(3)))
    row.update(owner_caught=caught, owner_fa=fa, owner_multi=multi)
    with open(os.path.join(OUT, 'results.jsonl'), 'a') as fh:
        fh.write(json.dumps(row) + '\n')
    print(json.dumps(row))


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:4]))
