# -*- coding: utf-8 -*-
"""Run the seventeen `Test/` clips through one or more configurations, into `test_result/`.

These are the only clips here that are not a lab dataset -- real footage, mostly of elderly
people, at the resolutions and framings a real camera gives. They are the last check before any
configuration is believed, and the reason the result belongs in the repository rather than in a
scratchpad is that this is the number a person is most likely to be shown.

How each clip is scored is defined once, in `training/test_clips.py`: four single real falls
(caught or missed), one clip with no fall in it at all (silence is correct), and twelve
compilations that are counted rather than scored. Reporting a compilation as accuracy would be
inventing a denominator.

Output, all in `test_result/` (override with `TEST_RESULT_DIR`):

    test_clips.json   raw per-clip alert count and peak score, resumable
    test_clips.md     the same thing as a Thai table, which is what actually gets read
    test_clips.csv    for a spreadsheet

Usage:
    python training/measure/testclips_current.py
    TEST_MODEL_DIR=<dir> TEST_LABEL="frame position" python training/measure/testclips_current.py
"""
import csv
import datetime
import importlib.util
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'training'))
import test_clips as tc  # noqa: E402

OUT_DIR = os.environ.get('TEST_RESULT_DIR', os.path.join(ROOT, 'test_result'))
CLIPS = tc.ALL

# TEST_MODEL_DIR swaps the model while keeping both profiles, which is how a candidate
# classifier is compared against the deployed one on real footage. TEST_LABEL names it in the
# report, so two runs can live in the same file side by side.
MODEL_DIR = os.environ.get('TEST_MODEL_DIR', os.path.join(ROOT, 'models'))
SUFFIX = ('  [' + os.environ['TEST_LABEL'] + ']') if os.environ.get('TEST_LABEL') else ''
CONFIGS = [
    ('deployed GPU  960 @ 20fps, partial 4' + SUFFIX, 960, 20, 4),
    ('deployed CPU  320 @ 8fps,  partial 4' + SUFFIX, 320, 8, 4),
]


def load(imgsz, partial):
    os.environ['V3_DEVICE'] = os.environ.get('V3_DEVICE', 'cuda')
    os.environ['V3_IMGSZ'] = str(imgsz)
    os.environ['V3_PARTIAL_MIN'] = str(partial)
    spec = importlib.util.spec_from_file_location(
        'v3_%d_%d' % (imgsz, partial), os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, m.V3PoseFallDetector(model_dir=MODEL_DIR)


def run(mod, det, clip, fps):
    """-> (number of separate alerts, highest score any window reached).

    Frames are taken at `fps`, the rate that profile runs at live, not every frame of the file:
    the window is a fixed number of frames, so reading a 30fps file end to end measures a
    detector nobody deploys.
    """
    st = mod.V3MultiPersonFallState()
    cap = cv2.VideoCapture(clip)
    src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, last_slot, peak, last, alerts = 0, -1, 0.0, None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        slot = int(i * fps / src)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        res = mod.detect_v3_fall_multi(frame, st, det, config=None)
        for r in res:
            peak = max(peak, float(r[2]))
        label = 'fall' if any(r[1] for r in res) else 'no_fall'
        if label != last:
            alerts += 1 if label == 'fall' else 0
            last = label
    cap.release()
    return alerts, peak


def verdict(clip, alerts):
    """What this clip's number means, given what is in the clip."""
    kind = tc.kind(clip)
    if kind == 'real':
        return u'จับได้' if alerts else u'พลาด'
    if kind == 'negative':
        return (u'เงียบ (ถูก)' if not alerts
                else u'แจ้งเตือนผิด')
    return u'%d ครั้ง' % alerts


HEADER = [
    u'# ผลทดสอบคลิปจริงใน `Test/`',
    u'',
    u'สร้างอัตโนมัติจาก `python training/measure/testclips_current.py` — อย่าแก้ไฟล์นี้ด้วยมือ',
    u'',
    u'`13`–`16` มีล้มจริงคลิปละ 1 ครั้ง จึงตัดสินว่าจับได้หรือพลาด `17` ไม่มีการล้มเลย '
    u'เงียบคือคำตอบที่ถูก `1`–`12` เป็นคลิปรวมหลายเหตุการณ์ จึงนับจำนวนครั้ง ไม่ใช่คะแนน',
    u'',
    u'ตัวเลขในวงเล็บคือคะแนนสูงสุดที่โมเดลให้ในคลิปนั้น เกณฑ์แจ้งเตือนคือ 0.65',
    u'',
]
T_REAL = u'## ล้มจริง คลิปละ 1 ครั้ง'
T_NEG = u'## ไม่มีการล้ม — เงียบคือถูก'
T_COMP = u'## คลิปรวมหลายเหตุการณ์ — นับจำนวนครั้ง ไม่ใช่คะแนน'
N_COMP = (u'ไม่มี ground truth ต่อเหตุการณ์ ตัวเลขนี้เทียบกันได้ระหว่างคอลัมน์เท่านั้น '
          u'ห้ามอ่านเป็นความแม่นยำ `10` กับ `11` เป็นเด็ก อยู่นอกขอบเขตที่ระบบนี้ทำ')
T_SUM = u'## สรุปเฉพาะส่วนที่ให้คะแนนได้'
H_CLIP = u'คลิป'
H_WHAT = u'สิ่งที่อยู่ในคลิป'
H_CONF = u'การตั้งค่า'
H_CAUGHT = u'ล้มจริงที่จับได้'
H_QUIET = u'คลิปไม่มีล้มที่เงียบ'


def config_labels(results):
    """Configuration columns only. Keys starting with '_' are bookkeeping, such as _measured."""
    return [k for k in results if not k.startswith('_')]


def write_report(results):
    desc = tc.descriptions()
    labels = config_labels(results)
    measured = results.get('_measured', {})
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = list(HEADER)
    if measured:
        lines += [u'วันที่วัด: '
                  + u', '.join(u'%s = %s' % (k, v) for k, v in sorted(measured.items())),
                  u'']

    def table(title, clips, note=None):
        out = [title, u'']
        if note:
            out += [note, u'']
        out.append(u'| ' + H_CLIP + u' | ' + H_WHAT + u' | ' + u' | '.join(labels) + u' |')
        out.append(u'| --- | --- | ' + u' | '.join(['---'] * len(labels)) + u' |')
        for clip in clips:
            name = os.path.basename(clip)
            cells = []
            for label in labels:
                row = results[label].get(clip)
                if row is None:
                    cells.append(u'—')
                    continue
                alerts, peak = row
                cells.append(u'%s (%.2f)' % (verdict(clip, alerts), peak))
            out.append(u'| `%s` | %s | %s |'
                       % (name, desc.get(name, u''), u' | '.join(cells)))
        out.append(u'')
        return out

    lines += table(T_REAL, tc.REAL)
    lines += table(T_NEG, tc.NEGATIVE)
    lines += table(T_COMP, tc.COMPILATION, N_COMP)

    lines += [T_SUM, u'',
              u'| ' + H_CONF + u' | ' + H_CAUGHT + u' | ' + H_QUIET + u' |',
              u'| --- | --- | --- |']
    for label in labels:
        caught = sum(1 for c in tc.REAL if results[label].get(c, (0, 0))[0])
        quiet = sum(1 for c in tc.NEGATIVE if not results[label].get(c, (0, 0))[0])
        lines.append(u'| %s | %d/%d | %d/%d |'
                     % (label, caught, len(tc.REAL), quiet, len(tc.NEGATIVE)))
    lines.append(u'')

    md = os.path.join(OUT_DIR, 'test_clips.md')
    with open(md, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(u'\n'.join(lines))

    # utf-8-sig, because Excel on a Thai Windows install reads a plain UTF-8 csv as mojibake
    # and this file exists to be opened in Excel.
    csv_path = os.path.join(OUT_DIR, 'test_clips.csv')
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['clip', 'kind', 'description']
                        + [c for label in labels
                           for c in ('%s alerts' % label, '%s peak' % label)])
        for clip in CLIPS:
            name = os.path.basename(clip)
            row = [name, tc.kind(clip), desc.get(name, u'')]
            for label in labels:
                alerts, peak = results[label].get(clip, ('', ''))
                row += [alerts, peak]
            writer.writerow(row)
    return md, csv_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, 'test_clips.json')
    results = {}
    if os.path.exists(out_json):
        with open(out_json, encoding='utf-8') as fh:
            results = json.load(fh)
    for label, imgsz, fps, partial in CONFIGS:
        if label in results and len(results[label]) == len(CLIPS):
            continue
        mod, det = load(imgsz, partial)
        # Dated, because a column in this file outlives the run that produced it and "which
        # model was this" has been an unanswerable question here before.
        results.setdefault('_measured', {})[label] = datetime.date.today().isoformat()
        row = results.get(label, {})
        for clip in CLIPS:
            if clip in row:
                continue
            row[clip] = run(mod, det, clip, fps)
            results[label] = row
            # Written after every clip: seventeen clips at input size 960 is not a short run,
            # and a machine going down mid-sweep has cost this project real time before.
            with open(out_json, 'w', encoding='utf-8') as fh:
                json.dump(results, fh, indent=1, ensure_ascii=False)
            print('  %-44s %-14s alerts=%2d peak=%.2f'
                  % (label, clip, row[clip][0], row[clip][1]), flush=True)
    md, csv_path = write_report(results)
    print('wrote %s' % out_json)
    print('      %s' % md)
    print('      %s' % csv_path)


if __name__ == '__main__':
    main()
