# -*- coding: utf-8 -*-
"""Per-clip accuracy on the owner's Test/ clips (8 Oct), production model vs the original system.

Production = test_result/incidents/alerts_cpu_320px_8fps_roi256_phase0..7.json (320 px, 8 fps, crop 256, conf 0.3,
thr 0.65 -- these 8 files ARE results_pinned 'parity_deployed': 33.75 / 1.75 / 10.875). Original = exp_ir/original_14fps.json.
Truth and exclusions come from training/measure/score_incidents.truth(), so the per-clip rows add up to the report's totals.
Run: python test_result/fulltest_8oct/make_by_clip.py
"""
import glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'training', 'measure'))
os.chdir(ROOT)
import score_incidents as si

t = si.truth()
phases = [json.load(open(p, encoding='utf-8'))['segments']
          for p in sorted(glob.glob('test_result/incidents/alerts_cpu_320px_8fps_roi256_phase[0-7].json'))]
assert len(phases) == 8
old = json.load(open('training/data/exp_ir/original_14fps.json', encoding='utf-8'))['segments']
clips = sorted({k.split('#')[0] for k in t}, key=lambda c: int(c.split('.')[0]))
rows, tot = [], [0, 0, 0, 0, 0, 0]
for c in clips:
    keys = [k for k in t if k.split('#')[0] == c]
    falls = [k for k in keys if t[k]]; nof = [k for k in keys if not t[k]]
    caught = [sum(si.alerted(ph[k]) for k in falls) for ph in phases]
    fa = [sum(si.alerted(ph[k]) for k in nof) for ph in phases]
    oc = sum(si.alerted(old[k]) for k in falls); ofa = sum(si.alerted(old[k]) for k in nof)
    rows.append((c, len(falls), sum(caught) / 8, min(caught), max(caught), oc, len(nof), sum(fa) / 8, ofa))
    for i, v in enumerate((len(falls), sum(caught) / 8, oc, len(nof), sum(fa) / 8, ofa)):
        tot[i] += v
web = [('13.mp4', 'ล้มจริง 1 ครั้ง (ล้มลงท่าคลาน คลิปยาว 5 วิ)', 'ไม่เตือน', '—', 'ไม่เตือน'),
       ('14.mp4', 'ล้มจริง 1 ครั้ง (ยายถือไม้เท้า)', 'เตือน', '~10 วิ', 'เตือน'),
       ('15.mp4', 'ล้มจริง 1 ครั้ง (ทางเดินโรงพยาบาล 2 คน)', 'เตือน', '~8 วิ', 'เตือน'),
       ('16.mp4', 'ล้มจริง 1 ครั้ง (ลานคนเยอะ 4 คน)', 'เตือน', '~10.5 วิ', 'เตือน'),
       ('17.mp4', 'ไม่มีคนล้ม (ออกกำลังกายเป็นกลุ่ม) — ควรเงียบ', 'เงียบ ✓', '—', 'เตือนผิด ✗'),
       ('10.mp4', 'เด็กตกบันได — นอกขอบเขตของระบบ', 'ไม่เตือน', '—', 'ไม่ได้วัด'),
       ('11.mp4', 'เด็กตกเตียง กล้องกลางคืน — นอกขอบเขต', 'ไม่เตือน', '—', 'ไม่ได้วัด')]
L = ['# ผลความแม่นรายคลิป — คลิปในโฟลเดอร์ Test/ (8 ต.ค. 2026)', '',
     'โมเดลตัวจริงที่ใช้วันที่ 9 (yolo26s-pose + fall_classifier_v3, เกณฑ์ 0.65, 8 ภาพ/วินาที, CPU) เทียบกับ **ระบบเดิม** (โค้ดชุดแรก, MediaPipe, 14 ภาพ/วินาที)', '',
     '## 1. คลิป 13–17 และ 10–11 — ทดสอบผ่านหน้าเว็บจริง วันนี้ (เหมือนที่ผู้ใช้เห็น)', '',
     '| คลิป | เนื้อหา | ตัวจริง (ผ่านหน้าเว็บ) | เตือนหลังล้ม | ระบบเดิม |', '|---|---|---|---|---|']
L += ['| %s | %s | %s | %s | %s |' % w for w in web]
L += ['', 'สรุป: ล้มจริง 4 คลิป ตัวจริงเตือน **3** (ระบบเดิม 3) · คลิปไม่มีล้ม ตัวจริง**เงียบ** (ระบบเดิมเตือนผิด) · ภาพหน้าจอ/ภาพแนบแจ้งเตือนอยู่ในโฟลเดอร์นี้', '',
      '## 2. คลิปรวมเหตุการณ์ 1–9, 12 — นับทีละช่วง (แต่ละคลิปมีหลายเหตุการณ์ ตัดเป็นช่วง มีทั้งล้มและไม่ล้ม)', '',
      'ตัวจริง = ค่าเฉลี่ย 8 รอบ (ช่วง ต่ำสุด–สูงสุด) · ระบบเดิม = 1 รอบ · ช่วงที่อยู่นอกขอบเขต (เช่น เด็ก) ไม่นับ', '',
      '| คลิป | ช่วงที่มีคนล้ม | ตัวจริง จับได้ | ระบบเดิม จับได้ | ช่วงไม่มีล้ม | ตัวจริง เตือนผิด | ระบบเดิม เตือนผิด |', '|---|---|---|---|---|---|---|']
for c, nf, cm, cmin, cmax, oc, nn, fm, ofa in rows:
    L.append('| %s | %d | %.1f (%d–%d) | %d | %d | %.1f | %d |' % (c, nf, cm, cmin, cmax, oc, nn, fm, ofa))
L.append('| **รวม** | **%d** | **%.2f** | **%d** | **%d** | **%.2f** | **%d** |' % tuple(tot))
L += ['', 'อ่านง่าย: ในคลิปรวมเหตุการณ์ ตัวจริงจับได้ราว %.0f%% ของการล้ม (ระบบเดิม %.0f%%) แต่เตือนผิด %.2f ครั้ง (ระบบเดิม %d ครั้ง)'
      % (100 * tot[1] / tot[0], 100 * tot[2] / tot[0], tot[4], tot[5]), '',
      'ที่มา: `test_result/incidents/alerts_cpu_320px_8fps_roi256_phase0-7.json` (ตัวจริง, วัด 30 ก.ย. ค่าตั้งเดียวกับวันนี้), '
      '`training/data/exp_ir/original_14fps.json` (ระบบเดิม), เฉลย `test_result/incidents/incidents.json`, '
      'ผลผ่านหน้าเว็บ `ui_runs.log` · สร้างด้วย `make_by_clip.py`']
open('test_result/fulltest_8oct/accuracy_by_clip.md', 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('wrote test_result/fulltest_8oct/accuracy_by_clip.md -- totals', tot)
