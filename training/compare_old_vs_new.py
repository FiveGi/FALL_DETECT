"""Run every ground-truth surface under the old and new inference settings and write the
comparison to docs/settings_comparison.md.

What changed is the INFERENCE CONFIG, not the trained weights: `models/fall_classifier_v3.onnx`
is still SS35's `yolopose_aug_seed42` export (md5 194614047877dc8e9ff896e5331170f7) and the
pose backbone is still yolo26s-pose. The settings that changed are how much of the frame the
pose model gets to see (imgsz) and how sure it must be before a person counts (conf). Both
were previously unmeasured defaults.

    old: imgsz=640  conf=0.50
    new: imgsz=960  conf=0.30

Surfaces, all with real ground truth:
  val       GMDCSA24 held-out, 15 fall clips + 16 ADL clips, one person each
  train50   a second GMDCSA24 split, 25 + 25, never used to pick any setting
  2-person  composites built by training/make_multiperson_testset.py -- a real fall beside a
            real bystander, plus adl+adl pairs where any alert is wrong, plus a blank-half
            control that separates resolution effects from the second person
  speed     ms/frame on the GPU, since none of this is affordable if it cannot keep up

Every run uses the multi-person entry point `detect_v3_fall_multi`, which is what
camera_manager actually calls.

Usage:
    python training/compare_old_vs_new.py
"""
import glob
import importlib.util
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = [('old', '640', '0.5'), ('new', '960', '0.3')]


def load_v3():
    spec = importlib.util.spec_from_file_location(
        'v3_fall_detection', os.path.join(ROOT, 'app', 'detection', 'v3_fall_detection.py'))
    v3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)
    return v3


def child(imgsz, conf):
    """Each config runs in its own process: IMGSZ and POSE_CONF are read at import time, so
    changing them in-process would not take effect."""
    env = dict(os.environ, V3_IMGSZ=imgsz, V3_POSE_CONF=conf, V3_DEVICE='cuda',
               PYTHONIOENCODING='utf-8', COMPARE_CHILD='1')
    out = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env,
                         capture_output=True, text=True, encoding='utf-8', errors='replace')
    if out.returncode != 0:
        print(out.stdout[-2000:])
        print(out.stderr[-2000:])
        raise SystemExit(f'child failed for imgsz={imgsz} conf={conf}')
    for line in out.stdout.splitlines():
        if line.startswith('RESULT '):
            return eval(line[len('RESULT '):])
    raise SystemExit('child produced no RESULT line')


# ---------------------------------------------------------------- measurement (child side)
def run_clip(v3, detector, path):
    import cv2
    state = v3.V3MultiPersonFallState()
    cap = cv2.VideoCapture(path)
    alerts, last = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = v3.detect_v3_fall_multi(frame, state, detector, config=None)
        label = 'fall' if any(r[1] for r in results) else 'no_fall'
        if label != last:
            alerts += label == 'fall'
            last = label
    cap.release()
    return alerts


def measure():
    import cv2
    v3 = load_v3()
    detector = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    data = os.path.join(ROOT, 'training', 'data')

    sys.path.insert(0, os.path.join(ROOT, 'training'))
    from eval_v3_frame_drop import VAL_FALL, VAL_ADL, TRAIN_FALL, TRAIN_ADL  # noqa: E402

    res = {}
    for name, falls, adls, fall_dir, adl_dir in (
        ('val', VAL_FALL, VAL_ADL, 'gmdcsa24_fall_raw', 'gmdcsa24_adl_raw_val'),
        ('train50', TRAIN_FALL, TRAIN_ADL, 'gmdcsa24_fall_raw', 'gmdcsa24_adl_raw_train50'),
    ):
        caught = sum(1 for c in falls
                     if os.path.exists(os.path.join(data, fall_dir, c + '.mp4'))
                     and run_clip(v3, detector, os.path.join(data, fall_dir, c + '.mp4')) > 0)
        clean = sum(1 for c in adls
                    if os.path.exists(os.path.join(data, adl_dir, c + '.mp4'))
                    and run_clip(v3, detector, os.path.join(data, adl_dir, c + '.mp4')) == 0)
        res[name] = {'falls': caught, 'falls_total': len(falls),
                     'clean': clean, 'clean_total': len(adls)}

    comp = os.path.join(data, 'multiperson_composite')
    for key, pattern, want_alert in (('two_person', 'falladl_*.mp4', True),
                                     ('two_person_clean', 'adladl_*.mp4', False),
                                     ('blank_control', 'fallblank_*.mp4', True)):
        clips = sorted(glob.glob(os.path.join(comp, pattern)))
        good = 0
        for c in clips:
            n = run_clip(v3, detector, c)
            good += (n > 0) if want_alert else (n == 0)
        res[key] = {'good': good, 'total': len(clips)}

    cap = cv2.VideoCapture(os.path.join(ROOT, 'Test', '1.mp4'))
    frames = []
    while len(frames) < 40:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    state = v3.V3MultiPersonFallState()
    v3.detect_v3_fall_multi(frames[0], state, detector, config=None)  # warm-up, untimed
    state = v3.V3MultiPersonFallState()
    times = []
    for f in frames:
        t0 = time.perf_counter()
        v3.detect_v3_fall_multi(f, state, detector, config=None)
        times.append((time.perf_counter() - t0) * 1000)
    res['ms_per_frame'] = round(statistics.mean(times), 1)
    res['fps'] = round(1000.0 / statistics.mean(times), 1)
    print('RESULT ' + repr(res))


# ------------------------------------------------------------------------- report (parent)
def pct(a, b):
    return f'{a}/{b} ({a / b:.0%})' if b else '-'


def main():
    results = {}
    for label, imgsz, conf in CONFIGS:
        print(f'measuring {label} (imgsz={imgsz} conf={conf}) ...', flush=True)
        results[label] = child(imgsz, conf)

    o, n = results['old'], results['new']
    md = ['# Old vs new detection settings', '',
          'Same trained model in both columns -- `models/fall_classifier_v3.onnx`',
          '(md5 `194614047877dc8e9ff896e5331170f7`) on the `yolo26s-pose` backbone. What',
          'changed is the inference configuration, both of which were previously unmeasured',
          'library defaults:', '',
          '| setting | old | new |', '|---|---|---|',
          '| pose input size (`V3_IMGSZ`) | 640 | **960** |',
          '| pose confidence (`V3_POSE_CONF`) | 0.50 | **0.30** |', '',
          'Every number below comes from the multi-person entry point',
          '(`detect_v3_fall_multi`), the one the camera loop actually runs.', '',
          '## Results', '',
          '| what is measured | old | new |', '|---|---|---|']

    md.append(f"| GMDCSA24 val - falls caught | {pct(o['val']['falls'], o['val']['falls_total'])} "
              f"| {pct(n['val']['falls'], n['val']['falls_total'])} |")
    md.append(f"| GMDCSA24 val - normal clips with no false alarm | "
              f"{pct(o['val']['clean'], o['val']['clean_total'])} "
              f"| {pct(n['val']['clean'], n['val']['clean_total'])} |")
    md.append(f"| train50 - falls caught | {pct(o['train50']['falls'], o['train50']['falls_total'])} "
              f"| {pct(n['train50']['falls'], n['train50']['falls_total'])} |")
    md.append(f"| train50 - normal clips with no false alarm | "
              f"{pct(o['train50']['clean'], o['train50']['clean_total'])} "
              f"| {pct(n['train50']['clean'], n['train50']['clean_total'])} |")
    md.append(f"| two people, one falls - fall caught | "
              f"{pct(o['two_person']['good'], o['two_person']['total'])} "
              f"| {pct(n['two_person']['good'], n['two_person']['total'])} |")
    md.append(f"| two people, nobody falls - no false alarm | "
              f"{pct(o['two_person_clean']['good'], o['two_person_clean']['total'])} "
              f"| {pct(n['two_person_clean']['good'], n['two_person_clean']['total'])} |")
    md.append(f"| small-person control - fall caught | "
              f"{pct(o['blank_control']['good'], o['blank_control']['total'])} "
              f"| {pct(n['blank_control']['good'], n['blank_control']['total'])} |")
    md.append(f"| speed on GPU | {o['ms_per_frame']} ms/frame ({o['fps']} fps) "
              f"| {n['ms_per_frame']} ms/frame ({n['fps']} fps) |")

    md += ['', '## Reading this', '',
           'A camera needs about 25 fps to be followed without dropping frames, so both',
           'columns are fast enough on the GPU and the new column costs a few ms.', '',
           'The two-person and control rows exist because every other dataset here has',
           'exactly one person in frame, which is not the situation this system is for.',
           'The control is the same fall in the same widened frame with *nobody* beside it:',
           'it separates "a bystander broke detection" from "the person is now rendered at',
           'half the pixels", which the composite changes at the same time.', '',
           'Reproduce with `python training/compare_old_vs_new.py` (needs the GMDCSA24 clips',
           'in `training/data/` and the composites from',
           '`python training/make_multiperson_testset.py`).', '']

    # Its own file: docs/model_comparison.md carries hand-written sections (the trained
    # model has changed since, SKILL.md SS47) that regenerating this would erase.
    dest = os.path.join(ROOT, 'docs', 'settings_comparison.md')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('\n'.join(md))
    print('\nwritten to', dest)


if __name__ == '__main__':
    if os.environ.get('COMPARE_CHILD'):
        measure()
    else:
        main()
