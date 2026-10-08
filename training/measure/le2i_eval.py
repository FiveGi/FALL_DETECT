# -*- coding: utf-8 -*-
"""Le2i half-1 final test (plan round 3 D4; run ONCE per pre-registered config, protocol in AI_HANDOFF.md).

Full pipeline on every half-1 video at the deployed CPU profile (320 px, crop 256 every 8, conf 0.30, 8 fps by source
time, hip tracker, NUM_POSES=4, 4 threads), state reset per video, one ROI phase per run. Writes per-video alert times;
scoring is a separate step (--score) so the raw output is fixed before anyone looks at it.

Usage:
  python training/measure/le2i_eval.py run NAME POSE_MODEL|- MODEL_DIR THRESHOLD PHASE
  python training/measure/le2i_eval.py score NAME [NAME ...]
"""
import csv
import glob
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
LE2I = os.path.join(os.path.dirname(ROOT), 'datasets', 'le2i')
MANIFEST = 'training/data/le2i_holdout_manifest.json'
OUT = 'training/data/le2i_final'
LABELS = os.path.join(os.path.dirname(ROOT), 'datasets', 'omnifall_labels', 'labels', 'le2i.csv')


def video_path(v):
    scene, name = v.split('/')
    n = name.split('_')[1]
    for p in (os.path.join(LE2I, scene, 'Videos', 'video (%s).avi' % n), os.path.join(LE2I, scene, 'video (%s).avi' % n)):
        if os.path.exists(p):
            return p
    raise SystemExit('missing video for %s' % v)


def half1():
    m = json.load(open(MANIFEST))
    return m['half1_final_test']['videos']


def fall_labels():
    falls = {}
    for r in csv.DictReader(open(LABELS)):
        falls[r['path']] = falls.get(r['path'], False) or r['label'] == '1'
    return falls


def sha(path):
    import hashlib
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]


def run(name, pose, model_dir, thr, phase):
    # Codex P1: nothing inherited -- drop every V3_* variable, then pin the frozen profile explicitly.
    for k in [k for k in os.environ if k.startswith('V3_')]:
        del os.environ[k]
    for k, v in (('V3_IMGSZ', '320'), ('V3_ROI_IMGSZ', '256'), ('V3_ROI_FULL_EVERY', '8'), ('V3_POSE_CONF', '0.3'),
                 ('V3_DEVICE', 'cpu'), ('V3_TRACKER', 'hip'), ('V3_PREPROCESS', 'auto'), ('V3_ENSEMBLE', ''),
                 ('V3_COLLAPSE_ENABLED', '1'), ('CPU_THREADS', '4'), ('OMP_NUM_THREADS', '4'), ('V3_THRESHOLD', thr)):
        os.environ[k] = v
    pose_file = os.path.abspath(pose) if pose != '-' else os.path.join(ROOT, 'models', 'yolo26s-pose.pt')
    os.environ['V3_POSE_MODEL'] = pose_file
    import cv2
    spec = importlib.util.spec_from_file_location('v3', 'app/detection/v3_fall_detection.py')
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)
    assert abs(v3.THRESHOLD - float(thr)) < 1e-9 and v3.ROI_IMGSZ == 256 and v3.IMGSZ == 320
    det = v3.V3PoseFallDetector(model_dir=os.path.abspath(model_dir))
    os.makedirs(OUT, exist_ok=True)
    onnx = os.path.join(os.path.abspath(model_dir), 'fall_classifier_v3.onnx')
    out = {'name': name, 'pose': pose_file, 'pose_sha': sha(pose_file), 'model_dir': model_dir, 'onnx_sha': sha(onnx),
           'threshold': float(thr), 'phase': int(phase), 'env': {k: v for k, v in os.environ.items() if k.startswith('V3_')},
           'videos': {}}
    vids = half1()
    for i, v in enumerate(vids):
        cap = cv2.VideoCapture(video_path(v))
        src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        st = v3.V3MultiPersonFallState()
        det.reset_roi_state(int(phase))
        k, last, hits, peak = 0, -1, [], 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            slot = int(k * 8 / src)
            k += 1
            if slot == last:
                continue
            last = slot
            res = v3.detect_v3_fall_multi(frame, st, det, config=None)
            peak = max([peak] + [r[2] for r in res])
            if any(r[1] for r in res):
                hits.append(round(k / src, 2))
        n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if k == 0 or (n_meta > 0 and k < 0.98 * n_meta):   # Codex P1: a truncated/unreadable clip must stop the run
            raise SystemExit('decoded %d of %d frames: %s' % (k, n_meta, v))
        out['videos'][v] = {'alerts_at': hits, 'peak': round(float(peak), 3), 'frames': k}
        print('%2d/%d %s alerts %d' % (i + 1, len(vids), v, len(hits)), flush=True)
    if len(out['videos']) != len(vids):
        raise SystemExit('incomplete run')
    json.dump(out, open(os.path.join(OUT, '%s_p%s.json' % (name, phase)), 'w'), indent=1)


def score(names):
    falls = fall_labels()
    vids = half1()
    missing = [v for v in vids if v not in falls]
    if missing:
        raise SystemExit('no label for %s' % missing[:3])
    for name in names:
        fs = sorted(glob.glob(os.path.join(OUT, '%s_p*.json' % name)))
        phases = sorted(json.load(open(f))['phase'] for f in fs)
        if phases != list(range(8)):   # Codex P1: exactly the 8 pre-registered phases, or no score
            raise SystemExit('%s: phases %s, need 0-7' % (name, phases))
        for f in fs:
            d = json.load(open(f))
            if sorted(d['videos']) != sorted(vids):
                raise SystemExit('%s: video set differs from half 1' % f)
            caught = sum(1 for v in vids if falls[v] and d['videos'][v]['alerts_at'])
            fa = [v for v in vids if not falls[v] and d['videos'][v]['alerts_at']]
            nf, nn = sum(falls[v] for v in vids), sum(not falls[v] for v in vids)
            print('%-28s phase %d  falls caught %d/%d  non-fall videos alerting %d/%d  %s'
                  % (name, d['phase'], caught, nf, len(fa), nn, fa))


def clips(names):
    """One mp4 per NON-fall video that alerted in ANY config/phase, spanning first alert - 3 s to last alert + 3 s
    (all configs pooled, so judges cannot tell which config alerted). Owner policy (05 Oct): a deliberate kneel/lie-down
    on the floor is not a false alarm; ANY other alert in the span makes the video a false alarm (Codex P2)."""
    import cv2
    falls = fall_labels()
    span = {}
    for name in names:
        for f in sorted(glob.glob(os.path.join(OUT, '%s_p*.json' % name))):
            for v, r in json.load(open(f))['videos'].items():
                if not falls[v] and r['alerts_at']:
                    a, b = span.get(v, (1e9, -1e9))
                    span[v] = (min(a, r['alerts_at'][0]), max(b, r['alerts_at'][-1]))
    os.makedirs(os.path.join(OUT, 'adjudicate'), exist_ok=True)
    for v, (t0, t1) in sorted(span.items()):
        cap = cv2.VideoCapture(video_path(v)); src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((t0 - 3) * src)))
        fn = os.path.join(OUT, 'adjudicate', '%s.mp4' % v.replace('/', '_'))
        w = None
        for _ in range(int((t1 - t0 + 6) * src)):
            ok, fr = cap.read()
            if not ok:
                break
            if w is None:
                w = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), src, (fr.shape[1], fr.shape[0]))
            w.write(fr)
        if w is not None:
            w.release()
        print(fn, 'alerts %.1f-%.1fs' % (t0, t1))


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(*sys.argv[2:7])
    elif sys.argv[1] == 'clips':
        clips(sys.argv[2:])
    else:
        score(sys.argv[2:])
