"""Build the owner-requested report and check preservation; no inference or services."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import hashlib
import html
import re
from bs4 import BeautifulSoup as Soup

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'report'
read = lambda p: p.read_text(encoding='utf-8')
sources = [Soup(read(R / f'report_day{d}.html'), 'html.parser') for d in (2, 3)]
ref = Soup(read(R / 'report_first_style_ref.html'), 'html.parser')
out = Soup('<!doctype html>\n<html lang="th"><head></head><body><main class="wrap"></main></body></html>', 'html.parser')
for tag in sources[0].head.find_all(['meta', 'link']):
    out.head.append(deepcopy(tag))
title = out.new_tag('title')
title.string = 'รายงานวันที่ 2–3 — การตรวจจับการล้ม'
out.head.append(title)
out.head.append(deepcopy(ref.style))
for s in sources:
    for style in s.find_all('style')[1:]:
        out.head.append(deepcopy(style))

def add(parent, markup):
    fragment = Soup(markup, 'html.parser')
    for child in list(fragment.contents):
        parent.append(child)

add(out.head, '''<style>
#day3 .prob { background: transparent; border: 0; padding: 0; margin: 0; }
#day3 .prob h3 { display: block; margin: 30px 0 8px; }
.day-heading { margin-top: 56px; }
.media-list { columns: 2 300px; }
</style>''')
main = out.main
add(main, '''<header class="doc"><div class="toolbar"><span class="eyebrow">รายงานรวมวันที่ 2–3 · ระบบตรวจจับการล้มของผู้สูงอายุ</span><button id="theme" type="button" aria-label="สลับธีมสว่างและมืด">สลับสว่าง / มืด</button></div>
<h1>สิ่งที่พบ สิ่งที่ลองแก้ และทางเลือกก่อนทดสอบยืนยัน</h1>
<p class="col">เรียงรายงานวันที่ 2 แล้ววันที่ 3 ตามเวลาของแต่ละฉบับ พร้อมเทคนิคปรับภาพกลางคืน บันทึกเก่ายังคงอยู่เพื่อแสดงพัฒนาการ ให้ใช้อัปเดตวันที่ 3 อ่านสถานะล่าสุด</p></header>
<section id="merged-summary" class="summary col"><h2>สรุปรวมวันที่ 2–3</h2><ul>
<li>วันที่ 2 พบว่าท่าทางคนที่ล้มหายไป โดยเฉพาะกลางคืนจำลอง การปรับภาพช่วยได้จำกัด ส่วน POSE-IR ช่วยจับล้มได้มากขึ้น แต่ผลเต็มยังไม่สม่ำเสมอพอจะเปลี่ยนระบบจริง</li>
<li>วันที่ 3 เปรียบเทียบการสอนตัวตัดสิน T1/T2 และตัวหาท่าทาง T3 พร้อมตัววัดว่าเตือนถูกคนหรือไม่ ผลของ T2-full ยังแกว่งตามรอบสุ่ม และการเฉลี่ยน้ำหนักยังไม่แก้ข้อจำกัดครบ</li>
<li>ตามอัปเดตล่าสุดในรายงานวันที่ 3 เจ้าของเลือก A แล้ว: เน้นจับล้มได้มากกว่า โดยยังต้องทดสอบ Le2i ตามกติกาที่ล็อกไว้ งานทดสอบถูกพักและยังไม่มีผล จึงยังไม่ใช่อนุมัติใช้งานจริง</li>
<li>หัวข้อใหม่แยกผลทดสอบกลางคืนจำลองออกจากภาพตัวอย่างจริง เทคนิคช่างภาพไม่ได้รับประกันว่าจะทำให้ตัวหาท่าทางเห็นคนนอน และข้อมูล IR จริงยังต้องทดสอบก่อนสรุปประโยชน์</li>
</ul><p class="note">หลักฐานสรุป: <a href="#d2-summary">บริบทวันที่ 2</a> · <a href="#d3-s0">อัปเดตวันที่ 3</a> · <a href="#d3-s1">ผลสะสมวันที่ 3</a> · <a href="#night-tech">เทคนิคกลางคืน</a></p></section>
<nav id="merged-toc" class="summary" aria-label="สารบัญ"><h2>สารบัญรวม</h2><ol></ol></nav>''')

preserved = []
for day, source in zip((2, 3), sources):
    article = out.new_tag('article', id=f'day{day}')
    main.append(article)
    add(article, f'<h2 class="day-heading">วันที่ {day} — เนื้อหาต้นฉบับตามช่วงเวลาที่บันทึก</h2>')
    block = deepcopy(source.main)
    for nav in block.find_all('nav'):
        nav.decompose()
    for button in block.find_all('button'):
        button.decompose()
    # Keep all body content; consolidate only page controls/navigation.
    for h in block.find_all('h1'):
        h.name = 'h2'
    for node in block.select('[id]'):
        node['id'] = f'd{day}-' + node['id']
    for node in block.select('[href^="#"]'):
        node['href'] = f'#d{day}-' + node['href'][1:]
    for node in block.select('[href]'):
        node['href'] = node['href'].replace('../Backend-Elderly-Surveillance-main/', '../')
    old = block.select_one('#d2-summary' if day == 2 else '#d3-s1 h2')
    old.string = '1. บริบทและผลสะสม ณ วันที่ ' + str(day)
    preserved.append(deepcopy(block))
    for child in list(block.contents):
        article.append(child)

draft = read(R / 'night_tech_section.md')
night = out.new_tag('section', id='night-tech', attrs={'class': 'prob'})
main.append(night)
add(night, '<h2>เทคนิคตอนกลางคืนแบบช่างภาพ</h2>')
add(night, '''<p class="col">เจ้าของถามว่าใช้เทคนิคของช่างภาพปรับภาพให้อ่านง่ายขึ้นได้ไหม: ภาพอินฟราเรดขาวดำมี noise และแสงไม่สม่ำเสมอ ซึ่งอาจทำให้ตัวหาท่าทางพลาดคนที่ล้มนอนบนพื้น</p>
<p class="note">เรียบเรียงจาก <a href="night_tech_section.md">ร่างของ Claude</a> ตรวจตัวเลขโดย Codex; ภาพสองตัวอย่างตรวจจากภาพที่แนบ ไม่ใช่การรันวัดโมเดลใหม่ เทคนิคที่ทดลองสั่งผ่าน V3_PREPROCESS ใน app/detection/v3_fall_detection.py; stacking ไม่ได้ทดลอง</p>''')
rows = [line for line in draft.splitlines() if line.startswith('|')]
table = '<div class="tbl"><table><thead><tr>'
for cell in rows[0].strip('|').split('|'):
    table += '<th>' + html.escape(cell.strip()) + '</th>'
table += '</tr></thead><tbody>'
for row in rows[1:]:
    table += '<tr>'
    for cell in row.strip('|').split('|'):
        cell = cell.strip().replace('คล้าย bilateral แต่ขอบดีกว่า เร็วกว่า', 'ใช้ภาพนำทางช่วยรักษาขอบ; ความเร็วขึ้นกับการใช้งาน').replace('(แรงสุด ช้าสุด)', '(ช้าที่สุดใน cost screen นี้)')
        escaped = html.escape(cell)
        escaped = re.sub(r'https://[^\s<]+', lambda m: '<a href="' + m[0] + '">' + m[0] + '</a>', escaped)
        table += '<td>' + escaped + '</td>'
    table += '</tr>'
add(night, table + '</tbody></table></div>')
costs = [('B', 'auto: CLAHE + gamma', 12.9), ('V', 'vflat → auto', 27.1), ('U', 'auto → unsharp', 23.0), ('D1', 'bilateral → auto', 16.7), ('D2', 'guided → auto', 31.5), ('D3', 'NL-means → auto', 364.3), ('G', 'auto; งาน accuracy เพิ่ม grey gate', 12.0)]
add(night, '''<h3>ความเร็ว: วัดทั้งลำดับการปรับภาพ ไม่ใช่ตัวกรองเดี่ยว</h3><p class="col">CPU 1 เธรด, 60 เฟรมจาก URFD 640×240 และ Test/ 1920×1080, 2 ต.ค. เครื่องไม่ว่างจึงเป็นค่าประมาณ; cost screen บังคับเปิด gate รวมถึง G จึงไม่ใช่การวัดต้นทุน grey gate แยกต่างหาก เกณฑ์คือเพิ่มจาก B ไม่เกิน 5 ms/เฟรม</p>''')
add(night, '<div class="tbl"><table><thead><tr><th>แขน / ลำดับ</th><th>เฉลี่ย ms/เฟรม</th><th>ต่างจาก B</th></tr></thead><tbody>' + ''.join(f'<tr><td>{arm}: {label}</td><td class="n">{ms:.1f}</td><td class="n">{ms-12.9:+.1f}</td></tr>' for arm,label,ms in costs) + '</tbody></table></div>')
add(night, '''<p class="col">มีเพียง D1 (+3.8 ms) และ G (−0.9 ms) ที่ผ่านเกณฑ์เพิ่มเวลาใน screen นี้; G ในงาน accuracy เพิ่มการเปิด auto เมื่อภาพเกือบขาวดำ (chroma &lt; 6) ร่วมกับเงื่อนไขความมืดเดิม ไม่ได้เปิดเฉพาะภาพขาวดำเท่านั้น ค่า G ที่ต่ำกว่า B ไม่ใช่หลักฐานว่า gate ทำให้เร็วขึ้น</p>
<p class="note">หลักฐาน: AI_HANDOFF.md หัวข้อ 2 ต.ค. 00:00; <a href="../training/data/multi_diag_v2/preprocess_cost.txt">preprocess_cost.txt บรรทัด 1–8</a>; คำสั่งเดิม <code>python training/measure/preprocess_cost.py</code> (<a href="../training/measure/preprocess_cost.py">โค้ด บรรทัด 17–23, 30–38</a>; เลือก URFD fall-01..04-cam0 และ Test/13,14 รวม 6 เส้นทางคลิป เก็บคลิปละไม่เกิน 10 เฟรม; log ยืนยัน 60 เฟรม ไม่ได้แจกแจงผลอ่านรายคลิป); <a href="../training/night_noise_eval.sh">night_noise_eval.sh บรรทัด 3–15</a></p>''')
accuracy = [('stock', 'เดิม', 17,4,15,1), ('noise_D1', '+ bilateral → auto',18,4,18,1), ('noise_G', '+ G',17,4,14,1), ('nightaug_s42','POSE-IR s42',38,3,26,3), ('nightaug_s43','POSE-IR s43',42,3,24,2)]
add(night, '''<h3>ความแม่น: URFD กลางคืนจำลอง</h3><p class="col">แต่ละ cache ใช้คลิปล้ม 60 และคลิปไม่ล้ม (ADL) 40 คลิป; ตัวตัดสินเดิมคงเกณฑ์ 0.65, CPU profile ค่าต่อไปนี้เป็น screen ของ noise seed 0 บน IR และ NIGHT_ALT ไม่ใช่ผลครบทุก noise seed หรือผลกล้องจริง s42/s43 คือรอบสุ่มฝึกโมเดล</p>''')
add(night, '<div class="tbl"><table><thead><tr><th>วิธี</th><th>IR: ล้ม /60</th><th>IR: เตือนผิด /40</th><th>NIGHT_ALT: ล้ม /60</th><th>NIGHT_ALT: เตือนผิด /40</th></tr></thead><tbody>' + ''.join('<tr><td>' + label + '</td>' + ''.join(f'<td class="n">{v}</td>' for v in vals) + '</tr>' for tag,label,*vals in accuracy) + '</tbody></table></div>')
add(night, '''<p class="col">D1 ช่วย +1 คลิปบน IR และ +3 บน NIGHT_ALT; G ไม่เพิ่มบน IR และลดลง 1 บน NIGHT_ALT ส่วน POSE-IR มากกว่าสองเท่าเฉพาะ IR (17 → 38/42); NIGHT_ALT เพิ่มจาก 15 → 26/24 จึงยังไม่ถึงสองเท่า ข้อสรุปคือการฝึกด้วยภาพกลางคืนจำลองน่าติดตามกว่าการปรับภาพใน screen นี้ ไม่ใช่การอนุมัติใช้จริง</p>
<p class="note">หลักฐาน: AI_HANDOFF.md หัวข้อ 2 ต.ค. 05:25; <a href="../training/data/multi_diag_v2/pose_screen_0500.txt">pose_screen_0500.txt บรรทัด 1–8, 14–23, 33–42</a>; ตัวสร้างรายงาน <a href="../training/measure/pose_screen_report.py">pose_screen_report.py บรรทัด 2–7</a>, รูปแบบคำสั่งเดิม <code>python training/measure/pose_screen_report.py stock nightaug_s42 nightaug_s43 noise_D1 noise_G</code> (อ่านผลที่บันทึกไว้ ไม่ได้รันคำสั่งวัดนี้ซ้ำ)</p>
<h3>ภาพตัวอย่างจริงจากคลิปเจ้าของ</h3><p class="col">ภาพประกอบจัดทำ 5 ต.ค. แต่ละช่องแสดงภาพผ่านเทคนิคหนึ่งและโครงร่างสีเหลืองที่โมเดลเห็น ช่องสุดท้ายหัวเขียวใช้ POSE-IR บนภาพเดิม เป็นตัวอย่างเฟรม ไม่ใช่ผลตลอดคลิปหรือสถิติความแม่น</p>
<figure><img src="img/night_tech_4_16.jpg" alt="คลิป 4#16 เปรียบเทียบภาพเดิม เทคนิคปรับภาพ และ POSE-IR" loading="lazy"><figcaption><b>คลิป 4#16:</b> ในเฟรมตัวอย่าง ภาพเดิมและทุกเทคนิคปรับภาพแสดงโครงร่าง 1 คนที่ยืน; POSE-IR แสดง 2 คน รวมคนที่นอนอยู่บนพื้น</figcaption></figure>
<figure><img src="img/night_tech_9_5.jpg" alt="คลิป 9#5 ไม่มีวิธีใดแสดงโครงร่างคนที่นอนในเฟรมตัวอย่าง" loading="lazy"><figcaption><b>คลิป 9#5:</b> ฉากเดียวกันอีกมุมตามคำอธิบายร่าง ทุกช่องรวม POSE-IR แสดงโครงร่าง 1 คนที่ยืน และไม่แสดงโครงร่างคนที่นอน กลางคืนยังเป็นข้อจำกัด</figcaption></figure>
<p class="note">ร่างระบุว่าไม่นำคลิป 5#16 มาแสดง เพราะคนล้มจมในอ่างน้ำและมองไม่เห็นตั้งแต่แรก; รอบรวมรายงานไม่ได้เปิดตรวจคลิปที่ตัดออกนี้</p>
<p class="col">การทำภาพให้อ่านง่ายสำหรับคนไม่รับประกันว่าโมเดลจะเห็นคนนอน งานต่อไปยังต้องทดสอบการฝึกโมเดลและข้อมูล IR จริง เช่น MUVIM; ข้อมูลนี้เป็นแนวทางที่เสนอ ยังไม่มีผลในหลักฐานชุดนี้ยืนยันว่าจะดีขึ้น</p>''')

# One complete TOC, including headings omitted from either original navigation.
toc = out.select_one('#merged-toc ol')
for area in [out.select_one('#day2'), out.select_one('#day3'), night]:
    day = area.get('id')
    for index, heading in enumerate(area.find_all('h2')):
        target = heading.get('id') or (heading.parent.get('id') if heading.parent.name == 'section' else None)
        if not target:
            target = f'{day}-heading-{index}'
            heading['id'] = target
        label = ('วันที่ ' + day[-1] + ' · ') if day in ('day2','day3') else ''
        add(toc, '<li><a href="#' + target + '">' + html.escape(label + heading.get_text(' ', strip=True)) + '</a></li>')
media = sorted({n[a] for n in out.find_all(True) for a in ('src','poster','href') if n.get(a,'').startswith(('img/','videos/'))})
add(main, '<section id="media-manifest"><h2>รายการไฟล์สื่อทั้งหมดที่หน้านี้อ้างอิง</h2><ul class="media-list">' + ''.join('<li><a href="' + p + '">' + p + '</a></li>' for p in media) + '</ul></section>')
add(toc, '<li><a href="#media-manifest">รายการไฟล์สื่อทั้งหมด</a></li>')
add(main, '<p class="note">การรวมฉบับ: ไม่ลบเนื้อหารายงานซ้ำ เพราะคงบริบทของแต่ละวัน; รวมสารบัญและปุ่มธีมเป็นชุดเดียว เปลี่ยนชื่อหัวข้อสรุปเดิมเป็นบริบทและผลสะสม และคงข้อความเดิมทั้งหมดไว้ ตัวเลขที่เพิ่มในหัวข้อกลางคืนตรงกับหลักฐาน แก้คำว่า “มากกว่าสองเท่า” ให้ใช้เฉพาะ IR และระบุว่าค่าเวลาเป็นลำดับปรับภาพรวม auto</p>')
out.body.append(deepcopy(sources[1].script))
target = R / 'report_day2_3.html'
target.write_text(str(out), encoding='utf-8')

# Static checks are deliberately separate from browser playback / model evaluation.
check = Soup(read(target), 'html.parser')
assert check.style.string == ref.style.string
assert len(check.find_all('nav')) == 1
assert len(check.find_all('h1')) == 1
ids = [n['id'] for n in check.select('[id]')]
assert len(ids) == len(set(ids))
assert all(a['href'][1:] in ids for a in check.select('[href^="#"]'))
for day, expected in zip((2,3), preserved):
    actual = check.select_one(f'#day{day}')
    # Every normalized text node from the source survives with multiplicity.
    missing = Counter(expected.stripped_strings) - Counter(actual.stripped_strings)
    assert not missing, missing
    for name in ('img','video','source','table','figure','details'):
        assert len(actual.find_all(name)) == len(expected.find_all(name)), (day,name)
assert all((R / p).is_file() for p in media), [p for p in media if not (R/p).is_file()]
urls = re.findall(r'https://[^\s|]+', draft)
assert all(check.find('a', href=u) for u in urls)
cost_text = read(ROOT / 'training/data/multi_diag_v2/preprocess_cost.txt')
for arm, label, ms in costs:
    assert re.search(r'^' + arm + r'\s+.*mean\s+' + re.escape(f'{ms:.1f}') + r' ms', cost_text, re.M)
screen = read(ROOT / 'training/data/multi_diag_v2/pose_screen_0500.txt')
for tag,label,ir,ifa,alt,afa in accuracy:
    chunk = screen.split('=== ' + tag + ' ')[1].split('===')[0]
    assert f'ir0    falls {ir}/60  FA {ifa}/40' in chunk
    assert f'alt0   falls {alt}/60  FA {afa}/40' in chunk
from PIL import Image
for media_path in media:
    if media_path.startswith('img/'):
        with Image.open(R / media_path) as picture:
            picture.verify()
manifest = R / 'report_day2_3_media.txt'
manifest.write_text('\n'.join(media) + '\n', encoding='utf-8')
report = ['Command: python report/merge_day2_3.py',
          'PASS: first-reference base CSS identical; one TOC, one H1, unique IDs, all fragment targets resolve.',
          'PASS: both reports retain all body text nodes (multiplicity), tables, images, videos, sources, figures and details after documented heading/control changes.',
          'Content duplicates removed: none. Original navigation consolidated; day-3 theme button replaced by shared button.',
          f'PASS: all {len(media)} relative media files exist; all {len(urls)} draft web URLs remain clickable.',
          'PASS: 7 cost means and 20 night screen counts equal recorded artifacts (URFD 60 falls /40 ADL per night cache; cost 60 frames).',
          'Numerical corrections: no source scalar changed; narrowed >2x to IR, +1..+3 to D1 (G: 0/-1), clarified full preprocessing sequence costs and forced-open G timing.',
          'Not run: model measurements, browser layout/playback, remote URL availability. Two night images visually inspected with view_image.',
          'Evidence: AI_HANDOFF.md headings 2026-10-02 00:00 and 05:25; training/data/multi_diag_v2/preprocess_cost.txt:1-8; pose_screen_0500.txt:1-8,14-23,33-42.',
          'PASS: Pillow verify decoded all 19 image files; 15 copied assets have SHA256 provenance in report_day2_3_copied_media.txt.',
          'Source hashes:']
report += [f'{p.name} SHA256 {hashlib.sha256(p.read_bytes()).hexdigest()}' for p in [R/'report_day2.html',R/'report_day3.html',R/'report_first_style_ref.html']]
(R/'report_day2_3_checks.txt').write_text('\n'.join(report)+'\n', encoding='utf-8')
print('\n'.join(report[:8]))
