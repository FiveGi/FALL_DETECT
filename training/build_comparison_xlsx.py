# -*- coding: utf-8 -*-
"""Build the original-versus-current comparison workbook, in Thai, for presenting.

Four columns of results, every one of them measured over the same clips at the frame rate that
column would really run at:

  ต้นฉบับ @15fps   the first commit of this repository (ce401fa) -- MediaPipe pose, a 30-frame
                   window, threshold 0.50, 2-of-3 smoothing -- fed the rate a camera delivers
  ต้นฉบับ @30fps   the same system fed every frame of the file, which is how its own numbers
                   were originally reported and a rate it cannot actually sustain
  ปัจจุบัน GPU     today's deployment on a machine with a GPU
  ปัจจุบัน CPU     today's deployment on four CPU cores, which is what the server has

Every count on the summary sheets is a SUMIFS/COUNTIFS over the per-clip sheets, so the
workbook recalculates if a per-clip row is corrected, and a reader can follow any number back
to the clips it came from.
"""
import json
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# Per-clip results live in the session scratchpad, which is not part of the repository --
# RESULTS_DIR points at whatever holds them. The workbook they produce IS committed.
HERE = os.environ.get('RESULTS_DIR', os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get('XLSX_OUT', os.path.join(
    r'D:\project\PROJECT\Backend-Elderly-Surveillance-main', 'docs',
    'เปรียบเทียบ_ต้นฉบับ_กับ_ปัจจุบัน.xlsx'))

FONT = 'Arial'
HDR_FILL = PatternFill('solid', fgColor='1F3864')
SUB_FILL = PatternFill('solid', fgColor='D9E2F3')
GOOD_FILL = PatternFill('solid', fgColor='E2EFDA')
BAD_FILL = PatternFill('solid', fgColor='FCE4D6')
NOTE_FILL = PatternFill('solid', fgColor='FFF2CC')
THIN = Side(style='thin', color='B4C6E7')
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

COLS = [
    ('ต้นฉบับ @15fps', 'orig15'),
    ('ต้นฉบับ @30fps', 'orig30'),
    ('ปัจจุบัน GPU', 'gpu'),
    ('ปัจจุบัน CPU', 'cpu'),
]

GROUPS = [
    ('urfd_fall', 'URFD', 'ล้ม', 'คลิปล้ม 60 คลิป จากชุดข้อมูลที่ไม่เคยใช้ปรับจูนเลย'),
    ('urfd_adl', 'URFD', 'ปกติ', 'กิจกรรมปกติ 40 คลิป จากชุดเดียวกัน'),
    ('val_adl', 'GMDCSA24 val', 'ปกติ', 'กิจกรรมปกติ 16 คลิป กันไว้ไม่ได้ใช้ฝึก'),
    ('gmdcsa_fall', 'GMDCSA24', 'ล้ม', 'คลิปล้ม 79 คลิป ส่วนใหญ่อยู่ในชุดฝึกของโมเดลปัจจุบัน'),
    ('train50_adl', 'GMDCSA24 train50', 'ปกติ', 'กิจกรรมปกติ 25 คลิป อยู่ในชุดฝึก'),
]


def load(path):
    d = json.load(open(path))
    return d['meta'], {(r[0], r[1]): bool(r[2]) for r in d['rows']}


meta, data = {}, {}
for key, path in [('orig15', 'original/original_perclip.json'),
                  ('orig30', 'original/original_perclip_fps30.json'),
                  ('gpu', 'perclip/partial4_gpu_fps20.json'),
                  ('cpu', 'perclip/partial4_imgsz320_fps8.json')]:
    meta[key], data[key] = load(os.path.join(HERE, path))

test_orig = json.load(open(os.path.join(HERE, 'original/all17_original.json')))
test_now = json.load(open(os.path.join(HERE, 'testclips_current.json')))

wb = Workbook()


def style_header(ws, row, ncols, height=22):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(name=FONT, bold=True, color='FFFFFF', size=11)
        cell.fill = HDR_FILL
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = BOX
    ws.row_dimensions[row].height = height


def put(ws, row, col, value, bold=False, fill=None, fmt=None, align=None, wrap=False, size=11):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = Font(name=FONT, bold=bold, size=size)
    if fill:
        cell.fill = fill
    if fmt:
        cell.number_format = fmt
    cell.alignment = Alignment(horizontal=align or 'left', vertical='center', wrap_text=wrap)
    return cell


def widths(ws, spec):
    for col, w in spec.items():
        ws.column_dimensions[col].width = w


# ---------------------------------------------------------------- per-clip: lab datasets
det = wb.active
det.title = 'ผลรายคลิป'
put(det, 1, 1, 'ผลการทดสอบรายคลิป — 1 = ระบบแจ้งเตือน, 0 = ไม่แจ้งเตือน', bold=True, size=13)
put(det, 2, 1, 'ช่อง "ผลที่ถูกต้อง" คือสิ่งที่ควรเกิดขึ้น: คลิปล้มควรได้ 1, คลิปปกติควรได้ 0',
    fill=NOTE_FILL, wrap=True)
det.merge_cells('A1:H1')
det.merge_cells('A2:H2')

hdr = ['ไฟล์คลิป', 'ชุดข้อมูล', 'ประเภท', 'ผลที่ถูกต้อง'] + [c[0] for c in COLS]
for i, h in enumerate(hdr, start=1):
    put(det, 4, i, h)
style_header(det, 4, len(hdr), height=30)

row = 5
group_rows = {}
for gkey, dataset, kind, _desc in GROUPS:
    names = sorted(k[1] for k in data['orig15'] if k[0] == gkey)
    start = row
    for name in names:
        put(det, row, 1, name)
        put(det, row, 2, dataset)
        put(det, row, 3, kind, align='center')
        put(det, row, 4, 1 if kind == 'ล้ม' else 0, align='center')
        for j, (_label, key) in enumerate(COLS):
            v = 1 if data[key].get((gkey, name)) else 0
            correct = (v == 1) if kind == 'ล้ม' else (v == 0)
            put(det, row, 5 + j, v, align='center',
                fill=GOOD_FILL if correct else BAD_FILL)
        row += 1
    group_rows[gkey] = (start, row - 1)
det.freeze_panes = 'A5'
widths(det, {'A': 30, 'B': 18, 'C': 10, 'D': 13, 'E': 16, 'F': 16, 'G': 15, 'H': 15})
det.auto_filter.ref = 'A4:H%d' % (row - 1)

# ---------------------------------------------------------------- summary
s = wb.create_sheet('สรุปผล', 0)
put(s, 1, 1, 'เปรียบเทียบระบบตรวจจับการล้ม: ต้นฉบับ (MediaPipe) กับ ระบบปัจจุบัน', bold=True, size=15)
s.merge_cells('A1:F1')
put(s, 2, 1, 'ทุกคอลัมน์คือ “ทั้งระบบ” ไม่ใช่แค่ไฟล์โมเดล และถูกป้อนคลิปชุดเดียวกัน '
             'ที่อัตราเฟรมที่คอนฟิกนั้นทำได้จริง', wrap=True, fill=NOTE_FILL)
s.merge_cells('A2:F2')
s.row_dimensions[2].height = 30

put(s, 4, 1, 'คอนฟิกที่นำมาเทียบ', bold=True, size=12)
spec_rows = [
    ('ตัวสกัดท่าทาง', 'MediaPipe BlazePose', 'MediaPipe BlazePose', 'yolo26s-pose @960', 'yolo26s-pose @320'),
    ('ขนาดหน้าต่าง', '30 เฟรม', '30 เฟรม', '15 เฟรม', '15 เฟรม'),
    ('ค่า Threshold', '0.50', '0.50', '0.65', '0.65'),
    ('กฎแจ้งเตือน', '2 ใน 3 หน้าต่าง', '2 ใน 3 หน้าต่าง', '1 ใน 3 หน้าต่าง', '1 ใน 3 หน้าต่าง'),
    ('อัตราเฟรมที่ป้อน', '15 fps', '30 fps', '20 fps', '8 fps'),
    ('ให้คะแนนหน้าต่างไม่เต็ม', 'ไม่ได้', 'ไม่ได้', 'ได้ (ตั้งแต่ 4 เฟรม)', 'ได้ (ตั้งแต่ 4 เฟรม)'),
    ('ฮาร์ดแวร์', 'CPU', 'CPU (เกินกำลังจริง)', 'GPU', 'CPU 4 คอร์'),
]
put(s, 5, 1, 'รายการ')
for j, (label, _k) in enumerate(COLS):
    put(s, 5, 2 + j, label)
style_header(s, 5, 5, height=28)
r = 6
for spec in spec_rows:
    put(s, r, 1, spec[0], bold=True)
    for j in range(4):
        put(s, r, 2 + j, spec[1 + j], align='center')
    r += 1

# headline results, as formulas over the per-clip sheet
r += 1
put(s, r, 1, 'ผลการตรวจจับ', bold=True, size=12)
r += 1
put(s, r, 1, 'ตัวชี้วัด')
for j, (label, _k) in enumerate(COLS):
    put(s, r, 2 + j, label)
style_header(s, r, 5, height=28)
head_row = r
r += 1


def countifs_correct(gkey, kind, col_letter):
    """Cells in `col_letter` on the detail sheet that got this group right."""
    a, b = group_rows[gkey]
    want = 1 if kind == 'ล้ม' else 0
    return "=COUNTIFS('ผลรายคลิป'!%s%d:%s%d,%d)" % (col_letter, a, col_letter, b, want)


metrics = []
for gkey, dataset, kind, desc in GROUPS:
    a, b = group_rows[gkey]
    n = b - a + 1
    label = ('%s — จับการล้มได้ (จาก %d คลิป)' % (dataset, n) if kind == 'ล้ม'
             else '%s — ไม่แจ้งเตือนผิด (จาก %d คลิป)' % (dataset, n))
    metrics.append((label, gkey, kind, n, desc))

for label, gkey, kind, n, desc in metrics:
    put(s, r, 1, label, bold=(gkey in ('urfd_fall', 'urfd_adl')))
    for j in range(4):
        col_letter = get_column_letter(5 + j)
        put(s, r, 2 + j, countifs_correct(gkey, kind, col_letter), align='center')
    r += 1

# the two totals that matter: everything never trained on
held_fall = ['urfd_fall']
held_clean = ['urfd_adl', 'val_adl']
r += 1
put(s, r, 1, 'รวมเฉพาะข้อมูลที่ไม่เคยใช้ฝึกเลย — จับการล้มได้ (60 คลิป)', bold=True, fill=SUB_FILL)
for j in range(4):
    cl = get_column_letter(5 + j)
    parts = '+'.join(countifs_correct(g, 'ล้ม', cl)[1:] for g in held_fall)
    put(s, r, 2 + j, '=' + parts, bold=True, align='center', fill=SUB_FILL)
fall_total_row = r
r += 1
put(s, r, 1, 'รวมเฉพาะข้อมูลที่ไม่เคยใช้ฝึกเลย — ไม่แจ้งเตือนผิด (56 คลิป)', bold=True, fill=SUB_FILL)
for j in range(4):
    cl = get_column_letter(5 + j)
    parts = '+'.join(countifs_correct(g, 'ปกติ', cl)[1:] for g in held_clean)
    put(s, r, 2 + j, '=' + parts, bold=True, align='center', fill=SUB_FILL)
clean_total_row = r
r += 2

put(s, r, 1, 'คิดเป็นเปอร์เซ็นต์ — จับการล้มได้', bold=True)
for j in range(4):
    put(s, r, 2 + j, '=%s%d/60' % (get_column_letter(2 + j), fall_total_row),
        bold=True, align='center', fmt='0.0%')
r += 1
put(s, r, 1, 'คิดเป็นเปอร์เซ็นต์ — ไม่แจ้งเตือนผิด', bold=True)
for j in range(4):
    put(s, r, 2 + j, '=%s%d/56' % (get_column_letter(2 + j), clean_total_row),
        bold=True, align='center', fmt='0.0%')
r += 2

notes = [
    'อ่านตารางนี้อย่างไร',
    '• URFD คือชุดข้อมูลสาธารณะที่ไม่เคยถูกใช้ปรับจูนอะไรในโครงงานนี้เลย จึงเป็นตัวเลขที่ควรนำไปอ้างอิง',
    '• GMDCSA24 ถูกใช้เลือกค่าต่าง ๆ มาหลายรอบ และคลิปล้มส่วนใหญ่อยู่ในชุดฝึกของโมเดลปัจจุบัน '
    'ตัวเลขจึงสูงเกินจริงสำหรับทั้งสองฝ่าย ใส่ไว้เพื่อดูว่าไม่มีอะไรพังเท่านั้น',
    '• คอลัมน์ “ต้นฉบับ @30fps” คือระบบเดิมที่ป้อนทุกเฟรมของไฟล์ ซึ่งเป็นวิธีที่ตัวเลขเดิมถูกรายงาน '
    'แต่เป็นอัตราที่เครื่องจริงทำไม่ได้ — ดูแผ่น “ความเร็ว”',
    '• ช่องสีเขียวในแผ่น “ผลรายคลิป” คือตอบถูก สีส้มคือตอบผิด',
]
for i, line in enumerate(notes):
    put(s, r + i, 1, line, bold=(i == 0), wrap=True, fill=NOTE_FILL if i else None)
    s.merge_cells(start_row=r + i, start_column=1, end_row=r + i, end_column=5)
    if i:
        s.row_dimensions[r + i].height = 30
widths(s, {'A': 52, 'B': 20, 'C': 20, 'D': 18, 'E': 18})

# ---------------------------------------------------------------- real footage
t = wb.create_sheet('คลิปจริง Test')
put(t, 1, 1, 'คลิปถ่ายจริง 17 คลิป — จำนวนครั้งที่ระบบแจ้งเตือน', bold=True, size=13)
t.merge_cells('A1:G1')
put(t, 2, 1, 'คลิป 13-16 มีการล้มคลิปละ 1 ครั้ง • คลิป 17 ไม่มีการล้มเลย (กลุ่มคนเต้นออกกำลังกาย) '
             'การเงียบคือคำตอบที่ถูก • คลิป 1-12 เป็นคลิปตัดต่อหลายเหตุการณ์ จำนวนครั้งจึงไม่ใช่คะแนน',
    wrap=True, fill=NOTE_FILL)
t.merge_cells('A2:G2')
t.row_dimensions[2].height = 45

thdr = ['คลิป', 'มีการล้มกี่ครั้ง', 'ต้นฉบับ @15fps', 'ต้นฉบับ @30fps', 'ปัจจุบัน GPU', 'ปัจจุบัน CPU', 'หมายเหตุ']
for i, h in enumerate(thdr, start=1):
    put(t, 4, i, h)
style_header(t, 4, len(thdr), height=30)

TEST_NOTE = {
    10: 'เด็กเล็กตกบันได — ภาพเล็กมาก อยู่นอกขอบเขตที่ระบบออกแบบมา',
    11: 'เด็กตกเตียงสองชั้น กล้องกลางคืน — อยู่นอกขอบเขต',
    13: 'ล้มจริง แต่กล้องเคลื่อนและซูม (พื้นหลังไหล 4.8 px/เฟรม) ระบบออกแบบมาสำหรับกล้องนิ่ง',
    14: 'ล้มจริง กล้องนิ่ง',
    15: 'ล้มจริง กล้องนิ่ง',
    16: 'ล้มจริง',
    17: 'ไม่มีการล้ม — ยืนยันด้วยการดูเองและให้ Gemini ตรวจซ้ำ',
}
ORIG15 = 'original MediaPipe 30-frame 0.50 @15fps'
ORIG30 = 'original MediaPipe 30-frame 0.50 @30fps'
NOW_GPU = 'deployed GPU  960 @ 20fps, partial 4'
NOW_CPU = 'deployed CPU  320 @ 8fps,  partial 4'

tr = 5
for n in range(1, 18):
    clip = 'Test/%d.mp4' % n
    put(t, tr, 1, '%d.mp4' % n)
    truth = ('1 ครั้ง' if 13 <= n <= 16 else ('ไม่มี' if n == 17 else 'หลายเหตุการณ์'))
    put(t, tr, 2, truth, align='center')
    for j, src in enumerate([test_orig.get(ORIG15, {}), test_orig.get(ORIG30, {}),
                             test_now.get(NOW_GPU, {}), test_now.get(NOW_CPU, {})]):
        v = src.get(clip)
        alerts = v[0] if v else None
        cell = put(t, tr, 3 + j, alerts if alerts is not None else '-', align='center')
        if 13 <= n <= 16 and alerts is not None:
            cell.fill = GOOD_FILL if alerts else BAD_FILL
        elif n == 17 and alerts is not None:
            cell.fill = BAD_FILL if alerts else GOOD_FILL
    put(t, tr, 7, TEST_NOTE.get(n, 'คลิปตัดต่อหลายเหตุการณ์'), wrap=True)
    tr += 1

tr += 1
put(t, tr, 1, 'สรุปเฉพาะคลิปล้มจริง (13-16)', bold=True, fill=SUB_FILL)
for j in range(4):
    cl = get_column_letter(3 + j)
    put(t, tr, 3 + j, '=COUNTIFS(%s17:%s20,">0")&"/4"' % (cl, cl),
        bold=True, align='center', fill=SUB_FILL)
put(t, tr, 7, 'นับว่ามีการแจ้งเตือนอย่างน้อยหนึ่งครั้งในคลิปหรือไม่', wrap=True, fill=SUB_FILL)
widths(t, {'A': 12, 'B': 18, 'C': 16, 'D': 16, 'E': 14, 'F': 14, 'G': 52})
t.freeze_panes = 'A5'

# ---------------------------------------------------------------- speed
sp = wb.create_sheet('ความเร็ว')
put(sp, 1, 1, 'ความเร็วที่วัดได้จริง — ทำไมคอลัมน์ “ต้นฉบับ @30fps” ถึงไม่ใช่ตัวเลือกจริง', bold=True, size=13)
sp.merge_cells('A1:D1')
put(sp, 2, 1, 'หน้าต่างของตัวจำแนกมีขนาดเป็น “จำนวนเฟรม” คงที่ อัตราเฟรมจึงกำหนดว่ามันเห็นการล้มมากแค่ไหน '
              'ความเร็วกับความแม่นยำจึงเป็นเรื่องเดียวกันบนเครื่องที่ไม่มี GPU', wrap=True, fill=NOTE_FILL)
sp.merge_cells('A2:D2')
sp.row_dimensions[2].height = 32

speed = [
    ('การวัด', 'ต้นฉบับ (MediaPipe)', 'ปัจจุบัน (YOLO-pose)', 'หมายเหตุ'),
    ('เวลาต่อเฟรม (เฟรม 1080p ชุดเดียวกัน, Process เดียวกัน)', '70.3 ms', '54.4 ms', 'MediaPipe ช้ากว่า 1.3 เท่า'),
    ('อัตราเฟรมที่ลูปกล้องจริงทำได้ (เครื่อง GPU)', '≈18 fps (ประมาณการ)', '23.4 fps (วัดจริง)', 'ต้นฉบับวัดใน Container ปัจจุบันไม่ได้ เพราะ Image ไม่มี MediaPipe ที่โหลดได้แล้ว'),
    ('อัตราเฟรมบนเซิร์ฟเวอร์ CPU 4 คอร์', '-', '8.1 fps', 'คอนฟิกปัจจุบันของฝั่ง CPU'),
    ('อัตราเฟรมที่คอลัมน์ “@30fps” ต้องการ', '30 fps', '-', 'ไม่เคยมีเครื่องไหนทำได้'),
]
for i, rowvals in enumerate(speed):
    for j, v in enumerate(rowvals):
        put(sp, 4 + i, 1 + j, v, bold=(i == 0), wrap=True)
    if i == 0:
        style_header(sp, 4, 4, height=26)
widths(sp, {'A': 48, 'B': 24, 'C': 24, 'D': 56})
for i in range(1, len(speed)):
    sp.row_dimensions[4 + i].height = 32

put(sp, 4 + len(speed) + 1, 1,
    'สรุป: ตัวเลข 54/60 ของคอลัมน์ “ต้นฉบับ @30fps” เป็นของจริง แต่ต้องการอัตราเฟรมที่ระบบเดิมทำไม่ได้ '
    'ที่อัตราเฟรมซึ่งมันทำได้จริง (15 fps) มันจับได้ 27/60 — ความแตกต่างหลักของโครงงานนี้จึงคือ '
    '“ระบบทำงานได้จริงที่ความเร็วที่เครื่องทำได้” ไม่ใช่ว่าโมเดลเก่งขึ้นในทุกกรณี',
    wrap=True, fill=NOTE_FILL, bold=True)
sp.merge_cells(start_row=4 + len(speed) + 1, start_column=1,
               end_row=4 + len(speed) + 1, end_column=4)
sp.row_dimensions[4 + len(speed) + 1].height = 58

# ---------------------------------------------------------------- methodology
m = wb.create_sheet('วิธีวัดและข้อควรระวัง')
put(m, 1, 1, 'วิธีวัดและข้อควรระวัง', bold=True, size=13)
m.merge_cells('A1:B1')
items = [
    ('ต้นฉบับหมายถึงอะไร',
     'โค้ดและโมเดลตาม Commit แรกของโครงงาน (ce401fa) ดึงออกมาจากประวัติ Git โดยตรง '
     'ไม่ใช่การเอาโมเดลเก่ามาใส่ใน Pipeline ปัจจุบัน จึงเป็นการเทียบ “ระบบกับระบบ”'),
    ('ทำไมต้องระบุอัตราเฟรมเสมอ',
     'หน้าต่างของตัวจำแนกมีขนาดเป็นจำนวนเฟรมคงที่ ที่ 30 fps หน้าต่าง 30 เฟรมกินเวลา 1 วินาที '
     'แต่ที่ 15 fps กินเวลา 2 วินาที โมเดลเดียวกันจึงให้ผลต่างกันมาก ตัวเลขที่ไม่บอกอัตราเฟรมไม่มีความหมาย'),
    ('การแบ่งครึ่ง URFD',
     'เลือกค่าต่าง ๆ บนครึ่งหนึ่ง แล้วยืนยันบนอีกครึ่งที่ไม่เคยเปิดดู การแบ่งใช้วิธีหยิบทีละสองลำดับ '
     'ไม่ใช่เลขคู่/คี่ เพราะ URFD สลับประเภทการล้มตามเลขลำดับ (คี่ = ล้มจากท่ายืน, คู่ = ล้มจากเก้าอี้) '
     'การแบ่งแบบคู่/คี่จึงเป็นการเทียบคนละโจทย์'),
    ('ข้อจำกัดของ URFD',
     'ภาพที่ใช้ได้มีขนาดเพียง 320x240 (ไฟล์วางภาพความลึกกับภาพสีไว้ข้างกัน) จึงตอบคำถามเรื่อง '
     '“ลดขนาดภาพแล้วเสียอะไร” สำหรับกล้องจริงไม่ได้ ต้องอ่านคู่กับ GMDCSA24 ซึ่งเป็น 720p'),
    ('หน้าต่างที่เติมไม่ทัน',
     'ระบบเดิมจะไม่ให้คะแนนใด ๆ จนกว่าหน้าต่างจะเต็มด้วยเฟรมที่มีคนอยู่ ระบบต้นฉบับที่ 15 fps '
     'ต้องการเวลาถึง 2 วินาที ทำให้ 24 จาก 60 คลิปล้มของ URFD ไม่ได้รับคะแนนเลย '
     'คะแนน 27/60 ของมันจึงเป็น “ให้คะแนนไม่ได้ 24 คลิป” ไม่ใช่ “ตอบผิด 33 คลิป”'),
    ('ระบบปัจจุบันแก้เรื่องนี้อย่างไร',
     'ให้คะแนนหน้าต่างที่ยังไม่เต็มได้ตั้งแต่มีเฟรมจริง 4 เฟรม โดยเติมด้านหน้าด้วยเฟรมแรกที่เห็น '
     'ผลคือแจ้งเตือนคนที่นอนอยู่กับพื้นแล้วตั้งแต่กล้องเห็นครั้งแรกได้ ไม่ใช่เฉพาะตอนเห็นจังหวะล้ม'),
    ('คลิป Test/17',
     'เคยถูกจัดว่าเป็นคลิปล้มที่ทุกคอนฟิกพลาดมาหลายวัน ความจริงคือไม่มีการล้มอยู่ในคลิปเลย '
     'ยืนยันด้วยการดูเองและให้ Gemini ตรวจซ้ำ การที่ระบบเงียบจึงเป็นคำตอบที่ถูก'),
    ('เครื่องที่ใช้วัด',
     'เครื่องพัฒนา: RTX 4070 Ti Super • ฝั่ง CPU: Container จำกัด 4 คอร์ ปิด CUDA '
     'ซึ่งเป็นรูปร่างของเซิร์ฟเวอร์จริง แต่เซิร์ฟเวอร์จริงเป็น VM ที่คอร์ช้ากว่า '
     'จึงควรวัดซ้ำบนเครื่องนั้นก่อนนำตัวเลขฝั่ง CPU ไปอ้างอิง'),
]
mr = 3
put(m, 2, 1, 'หัวข้อ')
put(m, 2, 2, 'รายละเอียด')
style_header(m, 2, 2, height=24)
for title, body in items:
    put(m, mr, 1, title, bold=True, wrap=True)
    put(m, mr, 2, body, wrap=True)
    m.row_dimensions[mr].height = 62
    mr += 1
widths(m, {'A': 32, 'B': 108})

# ---------------------------------------------------------------- threshold evidence
th = wb.create_sheet('การเลือกค่า Threshold')
put(th, 1, 1, 'หลักฐานว่าค่า Threshold ปัจจุบัน (0.65) ถูกเลือกมาแล้ว ไม่ใช่ค่าที่ตั้งไว้เฉย ๆ',
    bold=True, size=13)
th.merge_cells('A1:E1')
put(th, 2, 1, 'วัดครบทั้งสองโปรไฟล์ ที่คอนฟิกใช้งานจริง บนคลิปชุดเดียวกัน 220 คลิป '
              '"ครึ่งที่ยืนยัน" คือครึ่งของ URFD ที่ไม่ถูกใช้เลือกค่าใด ๆ',
    wrap=True, fill=NOTE_FILL)
th.merge_cells('A2:E2')
th.row_dimensions[2].height = 30

THR = {
    'gpu': [('0.55', 57, 38, 27, 16), ('0.60', 57, 39, 27, 16), ('0.65', 56, 40, 26, 16),
            ('0.70', 54, 41, 24, 16), ('0.75', None, None, None, None)],
    'cpu': [('0.55', 48, 38, 24, 15), ('0.60', 46, 38, 22, 15), ('0.65', 45, 41, 21, 16),
            ('0.70', 44, 42, 20, 16), ('0.75', 41, 42, 19, 16)],
}
try:
    _m, _d = load(os.path.join(HERE, 'perclip/thr0.75_gpu.json'))
    if len(_d) >= 220:
        f = sum(1 for k in _d if k[0] == 'urfd_fall' and _d[k])
        c = sum(1 for k in _d if k[0] in ('urfd_adl', 'val_adl') and not _d[k])
        fb = sum(1 for k in _d if k[0] == 'urfd_fall' and _d[k]
                 and ((int(k[1].split('-')[1]) - 1) // 2) % 2 == 1)
        cb = sum(1 for k in _d if k[0] == 'urfd_adl' and not _d[k]
                 and ((int(k[1].split('-')[1]) - 1) // 2) % 2 == 1)
        THR['gpu'][-1] = ('0.75', f, c, fb, cb)
except Exception:
    pass

trow = 4
for prof, title in (('gpu', 'โปรไฟล์ GPU (imgsz 960, 20 fps)'), ('cpu', 'โปรไฟล์ CPU (imgsz 320, 8 fps)')):
    put(th, trow, 1, title, bold=True, size=12)
    trow += 1
    for i, h in enumerate(['Threshold', 'URFD จับล้มได้ (60)', 'ไม่แจ้งผิด (56)',
                           'ครึ่งที่ยืนยัน: จับล้ม (28)', 'ครึ่งที่ยืนยัน: ไม่แจ้งผิด (20)'], start=1):
        put(th, trow, i, h)
    style_header(th, trow, 5, height=32)
    trow += 1
    for t_, f_, c_, fb_, cb_ in THR[prof]:
        fill = SUB_FILL if t_ == '0.65' else None
        put(th, trow, 1, t_ + ('  ← ใช้อยู่' if t_ == '0.65' else ''), bold=(t_ == '0.65'), fill=fill)
        for j, v in enumerate((f_, c_, fb_, cb_)):
            put(th, trow, 2 + j, v if v is not None else '-', align='center',
                bold=(t_ == '0.65'), fill=fill)
        trow += 1
    trow += 1

put(th, trow, 1,
    'อ่านผล: ลด Threshold ลงหนึ่งขั้นจาก 0.65 ได้คลิปล้มเพิ่มพอ ๆ กับจำนวนคลิปที่แจ้งผิดเพิ่ม '
    '(ฝั่ง CPU +3 แลก −3, ฝั่ง GPU +1 แลก −1) อัตราแลกเปลี่ยน 1:1 แบบนี้คือลักษณะของค่าที่อยู่จุดเหมาะสมแล้ว '
    'การวัดครั้งนี้จึงเป็นการยืนยันค่าเดิม ไม่ใช่การเปลี่ยน',
    wrap=True, fill=NOTE_FILL, bold=True)
th.merge_cells(start_row=trow, start_column=1, end_row=trow, end_column=5)
th.row_dimensions[trow].height = 58
widths(th, {'A': 22, 'B': 22, 'C': 20, 'D': 26, 'E': 28})

# ---------------------------------------------------------------- chart on the summary
# A grouped bar: four configurations, two measures, both percentages on one 0-100% axis --
# so there is one scale and no second y-axis. Two series get a legend, and the bars carry
# their own value labels, so identity never rests on colour alone. The two hues are slots 1
# and 2 of the reference categorical palette, checked with the skill's validator (worst
# adjacent pair, protan, delta-E 24.7 -- well clear of the 8 floor).
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.drawing.line import LineProperties
from openpyxl.chart.shapes import GraphicalProperties

chart_row = r + len(notes) + 2
put(s, chart_row - 1, 1, 'ภาพรวม — เฉพาะข้อมูลที่ไม่เคยใช้ฝึกเลย', bold=True, size=12)

# The chart reads its own small block of cells, which reference the totals above, so the
# picture and the table can never disagree.
base = chart_row
put(s, base, 1, 'คอนฟิก')
put(s, base, 2, 'จับการล้มได้')
put(s, base, 3, 'ไม่แจ้งเตือนผิด')
style_header(s, base, 3, height=24)
for j, (label, _k) in enumerate(COLS):
    put(s, base + 1 + j, 1, label)
    cl = get_column_letter(2 + j)
    put(s, base + 1 + j, 2, '=%s%d/60' % (cl, fall_total_row), fmt='0%', align='center')
    put(s, base + 1 + j, 3, '=%s%d/56' % (cl, clean_total_row), fmt='0%', align='center')

chart = BarChart()
chart.type = 'col'
chart.grouping = 'clustered'
chart.title = 'ผลบนข้อมูลที่ไม่เคยใช้ฝึกเลย (%)'
chart.y_axis.numFmt = '0%'
chart.y_axis.scaling.min = 0
chart.y_axis.scaling.max = 1.1    # headroom so the tallest bar's label clears the title
chart.y_axis.majorGridlines = None      # the value labels carry the numbers; a grid is noise
chart.x_axis.delete = False
chart.y_axis.delete = False
chart.gapWidth = 80
chart.overlap = -12
chart.height = 12    # tall enough that the legend gets its own band below the axis labels
chart.width = 22
chart.legend.position = 'r'   # its own column; at the bottom Excel overlaps it with the axis labels
chart.legend.overlay = False  # without this Excel floats the legend on top of the bars
data = Reference(s, min_col=2, max_col=3, min_row=base, max_row=base + 4)
cats = Reference(s, min_col=1, min_row=base + 1, max_row=base + 4)
chart.add_data(data, titles_from_data=True)
chart.set_categories(cats)
for series, hexcolor in zip(chart.series, ('2A78D6', 'EB6834')):
    series.graphicalProperties = GraphicalProperties(solidFill=hexcolor)
    series.graphicalProperties.line = LineProperties(noFill=True)
# Value only. Left to its defaults openpyxl lets Excel print the series name and the category
# name beside every bar as well, which on eight bars is unreadable overlap.
labels = DataLabelList()
labels.showVal = True
labels.showSerName = False
labels.showCatName = False
labels.showLegendKey = False
labels.showPercent = False
labels.showBubbleSize = False
labels.numFmt = '0%'
chart.dataLabels = labels
s.add_chart(chart, 'A%d' % (base + 6))

wb.save(OUT)
print('wrote', OUT)
