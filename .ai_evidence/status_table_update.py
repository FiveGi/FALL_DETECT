from pathlib import Path
import html, json, re, statistics

ROOT = Path(__file__).resolve().parents[1]
def read(f): return (ROOT / f).read_text(encoding='utf-8')
def write(f, s): (ROOT / f).write_text(s, encoding='utf-8', newline='')
report = 'report/report_day2_3.html'
before = read(report)
assert 'id="problem-status"' not in before, 'One-shot edit already applied; use check_status_table.py for verification.'
draft = read('report/status_table_draft.md')
groups, rows = [], []
for line in draft.splitlines():
    if line.startswith('## '): groups.append((line[3:], len(rows)))
    elif line.startswith('- '): rows.append(line[2:].split(' | '))
assert len(rows) == 22

# Corrections apply to the published copy; Claude's original draft stays intact.
rows[0][1] = 'ค่าการเห็นคนหลังล้มของครู x@1280: 0.857 → 0.504 บน URFD half A 16 คลิป'
rows[1][2] = 'Le2i runner ปฏิเสธผลเก่าก่อนเริ่ม และหยุดเมื่อผลไม่ครบ/งานล้ม; scorer ตรวจครบ 8 เฟส × 63 คลิป (ตรวจโค้ดและ fixture บางกรณีแล้ว ยังไม่ทดสอบ runtime ครบ)'
rows[2][2] = 'D3: Claude/Gemini มาร์ก 17 คลิปแยกกัน; ใช้ได้ 14 ตัดออก 3; ตรวจภาพตัวอย่าง 10/10 หลังทบทวนภาพซูม เป็นตัววัดเชิงตำแหน่ง ไม่ใช่ยืนยันตัวบุคคลทุกกรณี'
rows[3][2] = 'แก้การค้นหาคำนำหน้าในสคริปต์ที่เกี่ยวข้องแล้ว; ไม่ได้กู้ recipe.env เก่าที่ขาดข้อมูล'
rows[4][2] = 'แก้ให้เขียน DONE เมื่อคำสั่งสำเร็จและมีผล sustained ครบ 3 บรรทัด; ตรวจโค้ด ไม่ได้จำลองงานล้มรอบนี้'
rows[5][2] = 'เปลี่ยนเป็น detached ตามบันทึก Claude; รอบนี้ไม่ได้ทดสอบปิดแชตหรือความอยู่รอดของคิว'
rows[6][2] = 'เจ้าของอนุญาตหยุดและเทรนใหม่; ผลสรุป ctrlA s45c/s46c ตรงแถวเดิมทั้ง fixed/rule ไม่ได้พิสูจน์น้ำหนักตรงกัน; เพิ่มกฎตรวจ process ตอนพัก'
rows[7][2] = 'เปลี่ยนจากค้นข้อความเก่าเป็นไฟล์ t3marker ที่เขียนเมื่อ T3 สำเร็จ; ไม่ใช่หลักฐานว่า marker มีชื่อเฉพาะทุกรอบ'
rows[8][1] = 'ข้อความรายงานเกินหลักฐานและมีข้อผิดพลาดการเผยแพร่'
rows[8][2] = 'แก้สาระ 12 ข้อและจุดเผยแพร่ 2 จุดตามการทบทวน; การแก้ฉบับนี้ยังรอ Claude/Gemini ตรวจ ไม่อ้างว่าทุกงานผ่านทีมแล้ว'
rows[8][3] = 'กำลังทำ'
rows[9][2] = 'T2-full @0.65 เทียบตัวควบคุมสูตรเดียวกัน s45–47: จับล้ม owner/multi เพิ่ม 3/3; เลือก s45 @0.70 รอ Le2i; เกณฑ์ตัวเลข 5 ข้อผ่าน 2/6 seed ไม่ใช่ผ่านครบทุกเกณฑ์'
rows[9][3] = 'กำลังทำ'
rows[10][2] = 'เพิ่ม OF-Syn 582 คลิป (307 นอนลง + 275 ล้ม ตามบันทึก audit); ที่ 0.65 เตือนผิดลด แต่จับล้ม owner ลด 7.5/2.5/6.25 จาก 75 ใน s45–47'
rows[11][0] = 'ตัวหาท่าทางเห็นสิ่งของเป็นคน (nightaug s44: 141.2 ตัวต่อ 1,000 ภาพ)'
rows[11][1] = 'ตรวจพบคนที่ไม่มีจริง; ไม่ใช่จำนวนแจ้งเตือนล้ม'
rows[11][2] = 'T3 ทดสอบ COCO ไม่มีคน 1,176 ภาพ: ลด 47.5%/42.9%/57.2%; ผ่านเกณฑ์ลด ≥50% เพียง 1/3; s44 เคยเป็นทางเลือก B'
rows[11][3] = 'ลองแล้วไม่ได้ผล'
rows[12][2] = 'EMA 0.999 ผ่าน 3/6; 0.9999 ผ่าน 2/6 ต่ำกว่าเกณฑ์ ≥4/6 ทั้งคู่; ค่าเฉลี่ยดีขึ้นไม่แปลว่าดีขึ้นทุก seed จึงคงตัวเลือกเดิม'
rows[13][1] = 'ผลความแม่นขึ้นกับเกณฑ์ที่ใช้; เพิ่มเวลาประมวลผล'
rows[13][2] = 'ensemble 3 ตัวที่วัดบนเครื่องพัฒนา: median +52% เทียบ deployed (11 ช่วง owner, 252 เฟรม, 4 threads); ไม่ใช่เวลาของทุกชุด ensemble'
rows[14][1] = 'เพิ่มเวลาประมวลผล ยังไม่ยืนยันว่าความแม่นที่เพิ่มคุ้มต้นทุน'
rows[14][2] = 'median +13.5%/+16.5% เทียบ deployed ในรอบเดียวกัน (stock/s44 crop320; 11 ช่วง, 252 เฟรม, 4 threads); เกินเพดาน +5%'
rows[15][2] = 'พบความต่างระหว่างนโยบายกับป้าย ADL; ภาพนิ่งยังยืนยันเจตนา/สาเหตุของทุก alert ไม่ได้ เจ้าของยอมรับเตือนเมื่อตั้งใจคุกเข่า/นอนบนพื้น; ใช้กติกาใหม่กับ Le2i คงผลเกณฑ์เดิมไว้'
rows[15][3] = 'พบสาเหตุแล้ว'
rows[16][1] = 'ตัวเลือก A: wrong-person proxy เฉลี่ย 2.75/14 คลิป × 8 เฟส (ตัดออก 3 คลิป)'
rows[16][2] = 'ทดลองปิด collapse บน s44 + ตัวตัดสินเดิม: owner ลด 0.75/75 จึงคงไว้; กรองขอบภาพ phase 0 เสียการจับล้ม ไม่คุ้มตามบันทึก ไม่ใช่การทดลองปิดกฎบน A'
rows[17][2] = 'URFD จำลอง IR0/ALT0: bilateral เพิ่ม 1/3 จาก 60; IR0 POSE-IR s42/s43 ได้ 38/42 เทียบ stock 17 (noise draw เดียว ไม่ใช่ช่วงทุก draw); ตัวอย่าง 9#5 ยังพลาด ต้องวัด IR จริงเพิ่ม'
rows[17][3] = 'ยังเปิด'
rows[18][2] = 'D6 เต็มติดวิดีโอ OOPS ตามบันทึก Claude; pilot CAUCAFall 100 คลิปก็ยังไม่รัน เพราะทีมไม่เห็นตรงกัน ไม่ใช่ทุก pilot ทำไม่ได้เพราะไฟล์หาย'
rows[18][3] = 'ยังเปิด'
rows[19][0] = 'เลือกตัวที่จะทดสอบยืนยัน'
rows[19][2] = 'เจ้าของเลือก A (nightaug_s44 + T2-full s45 @0.70); ปิดเฉพาะการเลือก ยังไม่อนุมัติใช้งานจริง'
rows[20][2] = 'Le2i half 1: 63 คลิป (47 ล้ม/16 ไม่ล้ม), 2 ระบบ × 8 เฟส; เริ่มแล้วพักตามสั่ง Claude รายงานว่ายังไม่มีผลสรุป/ไม่ได้อ่านผล ไม่อ้างว่าไม่เคยเริ่มทดสอบ'
rows[20][3] = 'รอเจ้าของ'

# Source references for every row (saved evidence, not a fresh model run).
refs = [
('training/data/rerun_0310/teacher_presence_v2.txt',6), ('training/le2i_final.sh',6),
('AI_HANDOFF.md',3093), ('training/step1_controls.sh',16), ('training/step2_timing.sh',6),
('AI_HANDOFF.md',2996), ('training/data/stage1/results_pinned.jsonl',37), ('training/step1_controls.sh',8),
('AI_HANDOFF.md',725), ('training/data/stage1/results_pinned.jsonl',35),
('training/data/stage1/results_pinned.jsonl',15), ('training/data/multi_diag_v2/object_false_person_t3.txt',1),
('training/data/stage1/results_pinned.jsonl',62), ('training/data/multi_diag_v2/pipeline_timing.txt',5),
('training/data/multi_diag_v2/pipeline_timing_crop320.txt',6), ('AI_HANDOFF.md',3288),
('training/data/multi_diag_v2/track_metric_s44_t2fs45_070.txt',1), ('training/data/multi_diag_v2/pose_screen_0500.txt',14),
('AI_HANDOFF.md',3209), ('AI_HANDOFF.md',3288), ('AI_HANDOFF.md',3302), ('report/report_day3_outline.md',81),
]
assert len(rows) == len(refs), (len(rows),len(refs))

data = [json.loads(l) for l in read('training/data/stage1/results_pinned.jsonl').splitlines() if l.strip()]
byname = {r['name']: r for r in data}
mean = statistics.mean
passed = lambda r: (r['halfB_falls'] >= 21.8 and r['halfB_adl_clean'] >= 16.1 and r['val_adl_clean'] >= 6.1 and mean(r['night_fa']) <= 13/3 and mean(r['owner_fa']) <= 5)
checks = []
for prefix, expected in [('T2_truncfull',2),('T2FEMA',3),('T2FEMA4',2)]:
    rs = [byname[f'{prefix}_s{s}_rule'] for s in range(45,51)]
    k = sum(map(passed,rs)); assert k == expected
    checks.append(f'{prefix}: five numeric gates {k}/6; source results_pinned.jsonl rows ' + ','.join(str(data.index(r)+1) for r in rs))
for seed, delta in [(45,-7.5),(46,-2.5),(47,-6.25)]:
    a,b = [byname[f'T1_{arm}_s{seed}_065'] for arm in ['syn','ctrl']]
    assert mean(a['owner_caught']) - mean(b['owner_caught']) == delta
    assert mean(a['owner_fa']) < mean(b['owner_fa'])
    a = byname[f'T2_truncfull_s{seed}_065']
    assert mean(a['owner_caught']) > mean(b['owner_caught']) and mean(a['owner_multi']) > mean(b['owner_multi'])
checks.append('T1 fixed .65 owner deltas -7.5/-2.5/-6.25; FA lower all 3. T2-full fixed .65 owner/multi gain all 3 matched seeds.')
for seed in [45,46]:
    for suffix in ['065','rule']:
        a,b = [byname[f'ctrlA_s{seed}{c}_{suffix}'] for c in ['', 'c']]
        for key in ['halfB_falls','halfB_adl_clean','val_adl_clean','night_falls','night_fa','owner_caught','owner_fa','owner_multi']:
            assert a[key] == b[key]
checks.append('ctrlA s45/s46 clean reruns: all saved outcome fields identical, both thresholds; weights not compared.')
wrong = json.loads(read('training/data/multi_diag_v2/track_metric_s44_t2fs45_070.txt').splitlines()[0])
assert mean(wrong['wrong']) == 2.75 and wrong['n_scored'] == 14 and wrong['n_excluded'] == 3
checks.append('A D3 wrong mean 2.75/14, 8 phases, 3 excluded; source track_metric_s44_t2fs45_070.txt:1.')
checks.append('Timing from rounded saved medians: ensemble %.2f%%; crop stock %.2f%% / s44 %.2f%% vs deployed.' % ((86.8/57.1-1)*100,(68/59.9-1)*100,(69.8/59.9-1)*100))
checks.append('T3 normalized object rates: reductions '+ ', '.join(f'{(1-b/a)*100:.2f}%' for a,b in [(119.9,62.9),(119,68),(141.2,60.4)])+'; 1/3 reaches 50%.')

def pill(status):
    cls = {'แก้แล้ว':'ok','พบสาเหตุแล้ว':'ok','กำลังทำ':'wip','ลองแล้วไม่ได้ผล':'tried'}.get(status,'open')
    return f'<span class="pill {cls}">{html.escape(status)}</span>'

out = ['<section id="problem-status"><h2>ปัญหาที่พบ วิธีแก้ และสถานะ</h2>',
       '<p class="note">สถานะล่าสุดสำหรับอ่านคู่กับบันทึกย้อนหลัง; กำลังทำหมายถึงงานยังไม่จบ ไม่ได้หมายความว่ามีการรันอยู่</p>',
       '<p class="note">คำอธิบาย: '+ ' · '.join(pill(s) for s in ['แก้แล้ว','กำลังทำ','ยังเปิด','รอเจ้าของ','พบสาเหตุแล้ว','ลองแล้วไม่ได้ผล'])+' — พบสาเหตุแล้วไม่เท่ากับแก้ระบบแล้ว; ลองแล้วไม่ได้ผลหมายถึงไม่ผ่านเกณฑ์ที่ตั้งไว้</p>']
for g,(title,start) in enumerate(groups):
    end = groups[g+1][1] if g+1 < len(groups) else len(rows)
    out.append(f'<h3>{html.escape(title)}</h3><div class="tbl"><table><thead><tr><th scope="col">ปัญหา</th><th scope="col">ผลกระทบ</th><th scope="col">วิธีแก้</th><th scope="col">สถานะ</th></tr></thead><tbody>')
    for idx in range(start,end):
        row=rows[idx]; path,line=refs[idx]
        # Source line numbers are repaired after the handoff append below.
        link=f'<a href="../{path}#L{line}">หลักฐาน {idx+1}</a>'
        out.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in row[:2])+'<td>'+html.escape(row[2])+' <small>'+link+'</small></td><td>'+pill(row[3])+'</td></tr>')
    out.append('</tbody></table></div>')
out.append('<p class="note">ตรวจจากไฟล์ผลเดิม ไม่รันวิดีโอใหม่; <a href="../.ai_evidence/status_table_review.md">ขอบเขต ชุดข้อมูล คำสั่ง และรายการแก้ไขรายแถว</a></p></section>')
new = before.replace('<nav aria-label="สารบัญ"', '\n'.join(out)+'\n<nav aria-label="สารบัญ"',1)
new = new.replace('.open { color: var(--bad); background: var(--bad-bg); }','.open { color: var(--bad); background: var(--bad-bg); }\n.pill.tried { color: #4b5563; background: #e5e7eb; }\n[data-theme="dark"] .pill.tried { color: #e5e7eb; background: #374151; }\n@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .pill.tried { color: #e5e7eb; background: #374151; } }')
new = new.replace('<h2>สารบัญรวม</h2><ol>','<h2>สารบัญรวม</h2><ol><li><a href="#problem-status">ปัญหาที่พบ วิธีแก้ และสถานะ</a></li>',1)
# Historical day-2 headings retain their period's status, explicitly labelled.
states = ['ยังเปิด','ลองแล้วไม่ได้ผล','ยังเปิด','ลองแล้วไม่ได้ผล','ลองแล้วไม่ได้ผล','ลองแล้วไม่ได้ผล','ยังเปิด','รอเจ้าของ','ยังเปิด','ยังเปิด','รอเจ้าของ','ยังเปิด','ยังเปิด','ยังเปิด','ยังเปิด','ยังเปิด','ยังเปิด','ยังเปิด','ยังเปิด','ยังเปิด']
for n,status in enumerate(states,1):
    pattern = rf'(<section class="prob" id="d2-p{n}"><h3>.*?)(</h3>)'
    new,count = re.subn(pattern, lambda m:m[1]+' '+pill(status)+' <small>(ณ วันที่ 2)</small>'+m[2],new,count=1)
    assert count == 1
for sid,status in [('d3-s2','ลองแล้วไม่ได้ผล'),('d3-s3','กำลังทำ'),('d3-s4','ลองแล้วไม่ได้ผล'),('d3-s5','แก้แล้ว'),('d3-s6','ลองแล้วไม่ได้ผล'),('d3-s7','แก้แล้ว'),('d3-s8c','ลองแล้วไม่ได้ผล'),('d3-s8b','พบสาเหตุแล้ว')]:
    pattern = rf'(<section[^>]*id="{sid}"[^>]*>\s*<h2>.*?)(</h2>)'
    new,count = re.subn(pattern, lambda m:m[1]+' '+pill(status)+m[2],new,count=1)
    assert count == 1, sid
write(report,new)
write('.ai_evidence/status_table_checks.txt','Command: python .ai_evidence/status_table_update.py\n'+'\n'.join(checks)+'\n')
notes = [
'Teacher: corrected values reproduced in old teacher_presence.txt:4 and rerun_0310/teacher_presence_v2.txt:6; URFD half-A 16 clips, x@1280. Fix does not prove complete teacher coverage.',
'Evaluator: narrowed to Le2i stale-output rejection (not automatic deletion), 16 runs and scorer checks; runtime assurance remains limited (handoff bounded P1 confirmation).',
'D3: 17 marked, 14 scored/3 excluded, 8 phases; 10-frame agreement is historical team evidence (handoff), not an independent visual check this turn.',
'Recipe: prefix fix inspected in step1_controls.sh:16/t2_trunc.sh:12; historical missing provenance is not restored.',
'Timing DONE: step2_timing.sh:6–7 requires command success and three sustained lines; no runtime failure injection.',
'Detached: operational history attributed to Claude; no close-chat survival check. Not a guarantee against future stalls.',
'Stale runners: 4 processes/two overlapping training jobs are historical reports; saved clean/original summary rows match for both seeds/thresholds, not a weight identity test.',
'Early Step1: shared t3marker file replaces text matching; draft claim of a run-unique filename was too strong (step1_controls.sh:8).',
'Report review: removed unsupported single count of 14 factual overclaims; 12 substantive corrections plus 2 publication defects are documented separately; current edit awaits review.',
'T2: fixed .65 matched owner/multi gain 3/3; five numeric gates 2/6 under rule thresholds, not all-gates PASS. Near-specific deltas not independently recomputed, so omitted.',
'T1: 582 = 307+275 accepted synthetic clips is attributed audit history (handoff OF-Syn dual audit); saved fixed-.65 rows confirm all three owner regressions and FA reductions. Rule-threshold comparisons differ, so fixed threshold is explicit.',
'T3: 1176 actual person-free images; false detections normalized per 1000, not fall alerts. Rounded saved rates reproduce 47.54/42.86/57.22%; only 1/3 >=50%.',
'EMA: added missing .9999 arm (2/6); .999 is 3/6, both fail >=4/6. Cached arithmetic independently recomputed.',
'Ensemble: removed unqualified FA improvement; +52% is median for the measured three-classifier configuration, not proof of timing for the later T2-full ensemble.',
'Crop: 13.5/16.5% versus deployed (rounded medians), not the paired s44-only increase; 11 owner segments/252 frames/4 threads. No server timing.',
'URFD: disagreement with categorical intent/causality. Owner policy is recorded, but paired alert-time video adjudication would settle whether deliberate floor lowering explains individual alerts. Kept old gate failures.',
'Wrong person: 2.75/14 is A spatial-proxy mean over 8 phases, not 2.75 people. Collapse-off experiment used old classifier on s44, not A; handoff:3151–3157 records edge-filter tradeoff.',
'Night: IR0/ALT0 screen +1/+3 is bilateral only; 17 to 38/42 is one IR noise draw with training seeds42/43, not a cross-noise range. 60 falls/40 ADL. Real 9#5 example remains report media evidence, not a new visual assessment.',
'D6: full job blocked by OOPS availability per Claude; 100-clip CAUCAFall pilot not run because team did not agree. No filesystem census this turn.',
'Choice A: closed selection only; no deployment approval. Handoff owner decision13:32.',
'Le2i: 63 videos=47 fall+16 non-fall, 2 configs x8 phases. Started then paused; no-results/no-access is attributed Claude history, not independently verified. No rerun.',
'Owner requests: preserved as pending from current report/handoff; no email, download, deployment or policy action taken.'
]
audit=['# Status-table review — Codex, 2026-10-05','',
'Command actually run: `python .ai_evidence/status_table_update.py` (saved-result arithmetic and report edit); then HTMLParser structural/link checks. Output: [status_table_checks.txt](status_table_checks.txt). No model/video evaluation or workload process control. No SKILL/archive read.',
'Dataset context: stage1_eval_pinned.py cached URFD half-B 28 falls/20 ADL, GMDCSA val16 ADL over 8 day phases; simulated-night60 falls/40 ADL x3 noise draws; owner75 falls/13 negatives and multi25 x8 phases. D3:14 scored+3 excluded x8. Teacher half-A16. COCO1176 images. CPU11 owner segments/252 frames/4 threads. Zero clips rerun.',
'Measurement producers: training/measure/stage1_eval_pinned.py, training/measure/pipeline_timing.py; original invocation for CPU in training/step2_timing.sh:6. Historical command/result provenance remains in the linked report sections/handoff. Runtime events are explicitly attributed, not independently reproduced.','']
for i,(note,(path,line)) in enumerate(zip(notes,refs),1):audit.append(f'{i}. {note} Evidence: [{path}:{line}](../{path}#L{line}).')
audit += ['', 'Heading pills: day-2 headings explicitly say “ณ วันที่ 2” and summarize their existing status, preserving chronology; day-3 experimental/measurement headings show their respective outcome. “พบสาเหตุแล้ว” does not assert a fix. No browser visual check performed.']
write('.ai_evidence/status_table_review.md','\n'.join(audit)+'\n')
print('\n'.join(checks))
print('Inserted 22 rows / 4 groups, 28 heading pills. Original draft preserved.')
