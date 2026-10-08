# -*- coding: utf-8 -*-
"""The ORIGINAL detector, exactly as first committed, on the surfaces the new one is judged on.

"Original" = commit ce401fa: MediaPipe PoseLandmarker lite (IMAGE mode), 30-frame window, stride
10, threshold 0.50, 2-of-3 smoothing, collapse rule. Its classifier is loaded from the git LFS
object whose sha256 is recorded in that commit (1ba72967...); the file refuses to run with any
other, because this project nearly compared against `fall_classifier_v3_mediapipe_backup.onnx`,
which has the same size and a different hash -- it is NOT the original.

Fed at FPS (default 14, what it sustains on the CPU server's 3.5-core quota at 640x360 --
training/measure/bench_old_vs_new_cpu.py), sampling source time as a camera loop would.

Surfaces: URFD falls / ADL (right half = colour), GMDCSA24 validation ADL, Test/13-17, and the
owner's 126 compilation segments. Single-person by construction: that is the system as it was.

Usage:   FPS=14 python training/measure/eval_original.py
         NIGHT_SEED=0 FPS=14 python training/measure/eval_original.py   (night gate, URFD only; seeds 0/7/13)

NIGHT_SEED (Codex, 8 Oct: "old vs new at night" was claimed without measuring the old system at night): feeds the
SAME simulated-infrared frames the new system's night caches were built from -- cache_pose_streams.darken with
SIMULATE_IR=1 and the per-clip seeded RNG, applied to the same sampled frames in the same order -- and scores the
night gate (all 60 URFD falls, all 40 ADL) exactly as eval_candidate.py's NIGHT lines do.
"""
import hashlib
import importlib.util
import json
import os
import sys

import cv2

NIGHT_SEED = os.environ.get('NIGHT_SEED')
if NIGHT_SEED is not None:   # must be set before cache_pose_streams is imported: it reads these at import
    os.environ.update(SIMULATE_IR='1', SIMULATE_DARK_SEED=NIGHT_SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'training'))
sys.path.insert(0, os.path.join(ROOT, 'training', 'measure'))
FPS = float(os.environ.get('FPS', 14))
MODEL_DIR = os.path.join(ROOT, 'training', 'data', 'orig_model')
ORIGINAL_SHA = '1ba72967448be63b011bb63eb4e5eea1ffd55357e303791e11d3514424a9a01e'
OUT = os.path.join(ROOT, 'training', 'data', 'exp_ir', 'original_%gfps%s.json'
                   % (FPS, '' if NIGHT_SEED is None else '_night_s%s' % NIGHT_SEED))

sha = hashlib.sha256(open(os.path.join(MODEL_DIR, 'fall_classifier_v3.onnx'), 'rb').read()).hexdigest()
if sha != ORIGINAL_SHA:
    raise SystemExit('refusing to run: classifier sha256 %s is not the ce401fa original' % sha)

spec = importlib.util.spec_from_file_location('v3o', os.path.join(ROOT, 'training/measure/v3_original.py'))
v3o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3o)
DET = v3o.V3PoseFallDetector(model_dir=MODEL_DIR)


def frames(path, start_s=0.0, end_s=None, right_half=False):
    night = None
    if NIGHT_SEED is not None:
        import cache_pose_streams as cache
        assert cache.SIMULATE_IR and cache.SIMULATION_SEED == int(NIGHT_SEED) and not start_s and end_s is None
        rng = cache.clip_rng(path)
        night = lambda f: cache.darken(f, rng=rng)
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_s:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * src))
    i, last_slot = int(start_s * src), -1
    stop = int(end_s * src) if end_s else None
    while stop is None or i < stop:
        ok, f = cap.read()
        if not ok:
            break
        slot = int(i * FPS / src)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        f = f[:, f.shape[1] // 2:] if right_half else f
        yield night(f) if night else f
    cap.release()


def run(it):
    st = v3o.V3FallDetectionState()
    alert, peak = False, 0.0
    for f in it:
        det, prob, _label, _ = v3o.detect_v3_fall(f, st, DET, None)
        peak = max(peak, float(prob))
        alert = alert or bool(det)
    return alert, round(peak, 3)


def main():
    import cache_pose_streams as cache
    out = {'fps': FPS, 'classifier_sha256': sha, 'clips': {}, 'segments': {}}
    for group, paths, right_half in cache.clip_groups():
        if group not in (('urfd_fall', 'urfd_adl') if NIGHT_SEED is not None else ('urfd_fall', 'urfd_adl', 'val_adl')):
            continue
        for p in paths:
            out['clips']['%s/%s' % (group, os.path.basename(p))] = run(frames(p, right_half=right_half))
        print('  %-10s %d clips' % (group, len(paths)), flush=True)
    if NIGHT_SEED is not None:
        c = out['clips']
        falls = [v[0] for k, v in c.items() if k.startswith('urfd_fall/')]
        adl = [v[0] for k, v in c.items() if k.startswith('urfd_adl/')]
        out['clips'] = {k: v for k, v in c.items() if not k.startswith('val_adl/')}
        out['night_seed'] = int(NIGHT_SEED)
        json.dump(out, open(OUT, 'w'), indent=1)
        print('  NIGHT seed %s  falls %d/%d  ADL false alarms %d/%d' % (NIGHT_SEED, sum(falls), len(falls), sum(adl), len(adl)))
        print('wrote', OUT)
        return 0
    for n in (13, 14, 15, 16, 17):
        out['clips']['test/%d.mp4' % n] = run(frames('Test/%d.mp4' % n))
    inc = json.load(open('test_result/incidents/incidents.json', encoding='utf-8'))
    for clip, rows in inc['clips'].items():
        for r in rows:
            out['segments']['%s#%d' % (clip, r['segment'])] = run(
                frames(os.path.join('Test', clip), r['start_s'], r['end_s']))
    print('  segments %d' % len(out['segments']), flush=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    sys.exit(main())
