$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$outline = Get-Content "$root/report/report_day3_outline.md" -Encoding UTF8
$template = Get-Content "$root/../report/report_day2.html" -Raw -Encoding UTF8
$styles = ([regex]::Matches($template, '(?s)<style>.*?</style>') | ForEach-Object { $_.Value }) -join "`n"
function Inline([string]$s) {
    $s = [System.Net.WebUtility]::HtmlEncode($s)
    return [regex]::Replace($s, '\*\*(.*?)\*\*', '<strong>$1</strong>')
}
$sections = @{}
$key = ''
for ($i=0; $i -lt $outline.Count; $i++) {
    $line = $outline[$i]
    if ($line -match '^## (\d+b?)\. (.*)$') {
        $key = $Matches[1]
        $sections[$key] = @{ title=$Matches[2]; start=($i+1); lines=[System.Collections.Generic.List[string]]::new() }
    } elseif ($key) { $sections[$key].lines.Add($line) }
}
function Body($lines) {
    $out = [System.Text.StringBuilder]::new()
    $table = $false
    $paragraph = ''
    foreach ($line in $lines) {
        if ($line -match '^\|') {
            if ($paragraph) { [void]$out.AppendLine("<p>$(Inline $paragraph)</p>"); $paragraph='' }
            if (!$table) { [void]$out.AppendLine('<div class="tbl" role="region" aria-label="ตารางเปรียบเทียบ เลื่อนซ้ายขวาได้" tabindex="0"><table><caption>ผลเฉลี่ยของระบบเดิมและสองทางเลือก</caption>'); $table=$true; $head=$true }
            $cells=$line.Trim('|').Split('|')
            if (($cells -join '') -match '^[- :]+$') { continue }
            $tag = if ($head) {'th'} else {'td'}
            [void]$out.Append('<tr>')
            foreach ($cell in $cells) { [void]$out.Append("<$tag>$(Inline $cell.Trim())</$tag>") }
            [void]$out.AppendLine('</tr>'); $head=$false
            continue
        }
        if ($table) { [void]$out.AppendLine('</table></div>'); $table=$false }
        if ($line -match '^- (ภาพ/คลิป|ภาพ \(Codex\)|ภาพ:)' ) { continue }
        if ($line -match '^- ') {
            if ($paragraph) { [void]$out.AppendLine("<p>$(Inline $paragraph)</p>") }
            $paragraph=$line.Substring(2)
        } elseif ($line.Trim()) { $paragraph += ' ' + $line.Trim() }
    }
    if ($paragraph) { [void]$out.AppendLine("<p>$(Inline $paragraph)</p>") }
    if ($table) { [void]$out.AppendLine('</table></div>') }
    return $out.ToString()
}
$intro = @'
<!doctype html><html lang="th"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>รายงานวันที่ 3 — ผลทดลองและทางเลือกของเจ้าของ</title>
'@
$extra = @'
<style>
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;overflow-wrap:anywhere}button{font:inherit;color:var(--ink);background:var(--paper);border:1px solid var(--rule);border-radius:6px;padding:8px 14px;cursor:pointer;min-height:44px}.toolbar{display:flex;gap:12px;align-items:center;justify-content:space-between}.choice{border:2px solid var(--accent);border-top:6px solid var(--accent)}.choice h2{margin-top:0;border:0}.choice-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.choice-card{padding:16px;background:var(--accent-soft)}.choice-card h3{margin-top:0}.choice-card p:last-child{margin-bottom:0}caption{text-align:left;padding:12px;font-weight:600}nav a{display:inline-block;padding:5px 10px 5px 0}summary{min-height:44px}a:focus-visible,button:focus-visible,.tbl:focus-visible{outline:3px solid var(--accent);outline-offset:3px}.source{border-top:1px solid var(--rule);padding-top:12px}.status{font-weight:600;color:var(--accent)}.prob h2{margin-top:0}section{scroll-margin-top:16px}.glossary dt{font-weight:bold}.glossary dd{margin:0 0 14px}.note code{overflow-wrap:anywhere}video{aspect-ratio:16/9}footer{margin-top:40px;border-top:1px solid var(--rule);padding-top:15px}@media(max-width:560px){.wrap{padding:22px 14px 50px}.prob,.summary{padding:16px}.choice-grid{grid-template-columns:1fr}h2{font-size:1.25rem}table{min-width:630px}.toolbar{align-items:flex-start}}@media print{button,nav{display:none}.wrap{max-width:none}.prob{break-inside:avoid}body{background:white;color:black}}
</style></head><body><main class="wrap">
<header class="doc"><div class="toolbar"><span class="eyebrow">รายงานวันที่ 3 · ระบบตรวจจับการล้มของผู้สูงอายุ</span><button type="button" id="theme" aria-label="สลับธีมสว่างและมืด">สลับสว่าง / มืด</button></div>
<h1>จับล้มได้มากขึ้น หรือเตือนผิดคนน้อยลง</h1><p class="col">ผลทดลอง 3 ตุลาคม 2026 เวลา 21:00 ถึงเช้า 5 ตุลาคม 2026 · มีสองทางเลือกให้เจ้าของตัดสิน ก่อนทดสอบยืนยันครั้งสุดท้าย</p>
<div class="meta"><span>สถานะ: รอเจ้าของเลือก A หรือ B</span><span>ยังไม่ใช่การอนุมัติเปลี่ยนระบบจริง</span></div></header>
<nav class="summary" aria-label="สารบัญ"><a href="#s1">สรุป</a><a href="#s4b"><strong>4b · เลือก A / B</strong></a><a href="#terms">คำศัพท์และวิธีอ่าน</a><a href="#s2">T1</a><a href="#s3">T2</a><a href="#s4">T3</a><a href="#s5">เตือนถูกคน</a><a href="#s6">ความเร็ว</a><a href="#s7">เหตุขัดข้อง</a><a href="#s8">ทีม</a><a href="#s9">งานต่อไป</a></nav>
<p class="note">รายงานนี้เรียบเรียงจากโครงเนื้อหาสุดท้ายและผลตรวจเดิม ไม่ได้รันวัดตัวเลขใหม่ ภาพและคลิปเป็นตัวอย่างประกอบ ไม่ใช่หลักฐานว่าผลดีขึ้นกับทุกกล้อง</p>
'@
$glossary = @'
<section id="terms" class="summary"><h2>คำศัพท์และวิธีอ่านตัวเลข</h2>
<p><b>ระบบมีสองส่วน:</b> ตัวหาท่าทางหาตำแหน่งร่างกายในภาพ แล้วตัวตัดสินดูท่าทางต่อเนื่องว่าคล้ายการล้มหรือไม่ การเห็นคนจึงยังไม่เท่ากับจับการล้มได้</p>
<dl class="glossary">
<dt>T1 / T2 / T3 และ D3 / D5 / D6</dt><dd>ชื่อเรียกการทดลองและงานตรวจแต่ละเรื่อง ไม่ใช่คะแนน: T1 เพิ่มคลิปสร้างด้วยคอมพิวเตอร์; T2 ฝึกกับร่างกายที่เห็นไม่ครบ; T3 เพิ่มภาพไม่มีคน; D3 ตรวจว่าเตือนถูกคน; D5 วิเคราะห์การล้มที่พลาด; D6 สกัดท่าทางใหม่ทั้งชุด</dd>
<dt>POSE-IR, nightaug_s44, posneg_s44</dt><dd>ชื่อรุ่นตัวหาท่าทาง: nightaug ฝึกเพิ่มด้วยภาพกลางคืนจำลอง; posneg ฝึกเพิ่มให้รู้จักภาพไม่มีคนด้วย ส่วน IR คืออินฟราเรดที่กล้องใช้มองกลางคืน ขณะนี้ตัวเลขกลางคืนมาจากภาพจำลอง ยังไม่ใช่กล้องกลางคืนจริง</dd>
<dt>seed หรือ s45 / s46 / s47</dt><dd>หมายเลขกำหนดการสุ่มตอนฝึก เพื่อทดลองว่าวิธีเดียวกันยังช่วยเมื่อเปลี่ยนการสุ่มหรือไม่ ตัวควบคุมคือรุ่นที่ฝึกด้วยสูตรเดียวกันแต่ไม่เพิ่มวิธีที่กำลังทดสอบ การเทียบคู่คือเทียบด้วยหมายเลขสุ่มเดียวกัน</dd>
<dt>เกณฑ์ 0.65 / 0.70, TRUNC-full, ensemble</dt><dd>เกณฑ์คือคะแนนที่ใช้ประกอบการตัดสินเตือน ไม่ใช่เปอร์เซ็นต์ความแม่นยำ TRUNC-full หรือ T2-full คือชุดฝึกให้รู้จักคนที่เห็นร่างกายไม่ครบแบบเต็ม ส่วน ensemble คือใช้ตัวตัดสินหลายตัวร่วมกัน</dd>
<dt>URFD, val, Le2i, OF-Syn, COCO และ coco-pose</dt><dd>ชื่อชุดข้อมูล: URFD ใช้วัดการล้มและกิจกรรมไม่ล้ม; val คือชุดตรวจระหว่างพัฒนา (ที่นี่คือ GMDCSA 16 คลิปไม่ล้ม); Le2i half 1 คือครึ่งชุดที่กันไว้ยืนยันหลังเลือก; OF-Syn คือคลิปสร้างด้วยคอมพิวเตอร์; COCO คือชุดภาพทั่วไป และ coco-pose คือส่วนข้อมูลท่าทางคน</dd>
<dt>ตัวเลขมีทศนิยม และ /75 /25 /14</dt><dd>อ่านเป็นจำนวนที่จับได้เฉลี่ยจากจำนวนทั้งหมด ไม่ใช่เปอร์เซ็นต์ คลิปเจ้าของมี 75 ช่วงล้มและ 13 ช่วงไม่ล้ม; ฉากหลายคน 25 ช่วง; ระยะใกล้ 25 ช่วง แต่ละค่าเฉลี่ยจาก 8 รอบที่เลื่อนจังหวะเริ่มเก็บภาพ ไม่ใช่คลิปใหม่เพิ่ม 8 เท่า งานเตือนถูกคนให้คะแนน 14 จาก 17 ช่วง (ตัดออก 3 ช่วง)</dd>
<dt>จับล้ม / ไม่ล้มไม่เตือน / เตือนผิด / เตือนผิดคน / เห็นของเป็นคน</dt><dd>สองค่าแรกยิ่งมากยิ่งดี ส่วนสามค่าหลังยิ่งน้อยยิ่งดี และวัดคนละเรื่อง: เห็นของเป็นคนยังไม่จำเป็นต้องเกิดการแจ้งเตือนล้ม เครื่องหมาย — คือไม่มีค่าที่นำมาเทียบในตารางนี้</dd>
<dt>CPU, เฟรม, ms, p95, crop 320 และ COCO floor</dt><dd>CPU คือหน่วยประมวลผลหลัก; เฟรมคือภาพหนึ่งใบในวิดีโอ; ms คือมิลลิวินาที (หนึ่งในพันวินาที); p95 คือเวลาที่ 95% ของภาพใช้ไม่เกินค่านั้น; crop 320 คือตัดบริเวณคนแล้วป้อนภาพขนาด 320 ให้ระบบ; COCO floor ในงานนี้หมายถึงเกณฑ์จำกัดการเห็นของเป็นคน ซึ่งยังรอเจ้าของตัดสิน</dd>
<dt>process, deterministic, recipe.env, SYN_ / TRUNC_ และ scorecard</dt><dd>process คือโปรแกรมที่กำลังทำงาน; deterministic คือทำซ้ำด้วยเงื่อนไขเดิมแล้วได้ผลเดิม; recipe.env คือไฟล์บันทึกค่าฝึก และ SYN_ / TRUNC_ คือชื่อกลุ่มค่าที่เกี่ยวกับคลิปสังเคราะห์และร่างกายไม่ครบ; scorecard คือตารางสรุปผลเทียบเกณฑ์</dd>
</dl><p class="note">URFD ในตารางใช้ครึ่ง B: 28 คลิปล้ม / 20 คลิปไม่ล้ม; กลางคืนจำลอง 60 คลิปล้ม / 40 คลิปไม่ล้ม × การสุ่มสัญญาณรบกวน 3 แบบ ค่ากลางคืนเป็นค่าเฉลี่ย ไม่ใช่การวัดกล้องจริง</p></section>
'@
$notes = @{
'1'='คำว่า “หนึ่งตัว” หมายถึงตัวตัดสิน T2-full s45 ที่เลือกไว้ ส่วนตัวหาท่าทางยังมีสองทางเลือก A และ B; “เตือนผิดไม่เพิ่ม” ในสรุปหมายถึงคลิปเจ้าของ สำหรับกลางคืนดูค่าจริงแยกในตาราง 4b'
'2'='สถานะ: ไม่ใช้ T1 ตามผลทั้งสามรอบ ตัวเลขที่ลดลงเป็นจำนวนคลิปล้มที่จับได้ลดลงจากทั้งหมด 75 ช่วง'
'3'='สถานะ: เลือกตัวตัดสิน s45 ที่ 0.70 เป็นตัวเลือกสุดท้าย แต่ยังรอเลือกตัวหาท่าทางและทดสอบชุดที่กันไว้ กราฟเทียบคู่ด้านล่างใช้ 0.65 ทั้งสองฝั่ง ไม่ใช่ตาราง A/B ที่ 0.70'
'4'='สถานะ: วิธี T3 ไม่ผ่านเงื่อนไขว่าต้องลดการเห็นของเป็นคนอย่างน้อยครึ่งหนึ่งทุกรอบ แต่รุ่น s44 เมื่อนำมาจับคู่กับ T2-full ยังเป็นทางเลือก B ได้ ค่าการเห็นคนนอนพื้นเป็นสัดส่วน ยิ่งมากยิ่งดี'
'4b'='สถานะ: รอเจ้าของเลือก ทั้งคู่ผ่านเกณฑ์ตัวเลขห้าข้อ ไม่ได้แปลว่าผ่านทุกเงื่อนไขให้นำขึ้นใช้จริง ผลกลางคืนจำลองของ A จับได้ 36–40/60 และเตือนผิด 2–7/40; B จับได้ 30–35/60 และเตือนผิด 2–4/40 จากการสุ่มสามแบบ'
'5'='คำว่า “ระบบปัจจุบัน (ท่าทาง s44)” ในบันทึกส่วนนี้คือระบบทดลองที่ใช้ตัวตัดสินเดิมกับท่าทาง s44 ไม่ใช่ระบบที่ใช้งานจริงในตาราง 4b ค่า T2-full s45 ที่ 8.5 เป็นผลที่ 0.65; ตัวเลือก A ที่ 0.70 เตือนถูกคน 7.375/14 และ B 6.875/14 จึงไม่ควรนำ 8.5 ไปแทนค่าของตัวเลือก A'
'6'='สถานะ: ไม่ใช้การรวมสามตัว และไม่ใช้การตัดภาพ 320 ในตอนนี้ ค่า 57 ms คือเวลาประมวลผลต่อภาพที่เคยวัด ไม่ใช่คำรับรองความเร็วทุกเครื่อง กฎคนหายไปทันทีคงไว้'
'7'='สถานะ: ฝึกใหม่และได้ผลซ้ำเดิมแล้ว ข้อกำหนดตรวจโปรแกรมตอนพักเป็นกติกางานต่อไป รายงานนี้ไม่ได้เริ่มหรือหยุดงานทดลองใด'
'8'='สถานะ: ทีมตกลงให้นำ A และ B มาให้เจ้าของเลือก การตรวจอ่านรายงาน HTML ฉบับนี้โดย Claude และ Gemini ยังเป็นขั้นตอนถัดไป'
'9'='สถานะ: รอการตัดสินของเจ้าของ MUVIM คือชื่อชุดข้อมูลภาพอินฟราเรดจริงที่ต้องขอสิทธิ์เข้าถึง ส่วน server คือเครื่องที่จะนำระบบไปใช้งานจริง'
}
$media = @{
'3'=@'
<figure><img src="img/day3_t2_paired_deltas.png" alt="กราฟผลเพิ่มจาก T2-full เทียบตัวควบคุมเป็นคู่ 3 หมายเลขสุ่ม ที่เกณฑ์ 0.65" loading="lazy"><figcaption>จับล้มเจ้าของ / หลายคน / ระยะใกล้ / เตือนถูกคน ดีขึ้นทั้งสามคู่ แต่ไม่ได้ผ่านเกณฑ์ทุกข้อทุกคู่</figcaption></figure>
<figure><video controls playsinline preload="metadata" aria-label="เปรียบเทียบฉากใกล้กล้อง"><source src="videos/day3_compare_near.mp4" type="video/mp4"><a href="videos/day3_compare_near.mp4">เปิดคลิปฉากใกล้กล้อง</a></video><figcaption>ฉากใกล้กล้อง: Test/7.mp4 ช่วงที่ 2 เวลา 5.97–9.93 วินาที ระบบเดิมที่ 0.65 เทียบ A ที่ 0.70 · 32 ภาพ ที่ 8 ภาพต่อวินาที เลือกจังหวะเริ่มรอบแรก เป็นตัวอย่าง ไม่ใช่ผลรวม</figcaption></figure>
<figure><video controls playsinline preload="metadata" aria-label="เปรียบเทียบฉากหลายคน"><source src="videos/day3_compare_multi.mp4" type="video/mp4"><a href="videos/day3_compare_multi.mp4">เปิดคลิปฉากหลายคน</a></video><figcaption>ฉากหลายคน: Test/1.mp4 ช่วงที่ 15 เวลา 51.77–57.23 วินาที ระบบเดิมที่ 0.65 เทียบ A ที่ 0.70 · 44 ภาพ ที่ 8 ภาพต่อวินาที รอบแรกเช่นกัน</figcaption></figure>
'@
'4'=@'
<figure><img src="img/day3_coco_false_person_grid.jpg" alt="ตัวอย่าง COCO หกภาพไม่มีป้ายกำกับคน ที่ nightaug ตรวจเป็นคนแต่ posneg ไม่ตรวจเป็นคน" loading="lazy"><figcaption>เลือกตัวอย่าง 6 ภาพจาก 66 ภาพที่ไล่ดู ในกลุ่มที่เข้าเงื่อนไข 1,176 ภาพ ภาพเหล่านี้อธิบายอาการเท่านั้น ไม่ใช่การวัดซ้ำผลรวม 1,000 ภาพในตาราง และกรอบคนไม่ได้แปลว่ามีการเตือนล้มแล้ว</figcaption></figure>
'@
'5'=@'
<figure><img src="img/day3_audit_sheet1.jpg" alt="แผ่นภาพตรวจว่ากรอบคนที่ระบบเตือนตรงกับคนล้มจริงหรือไม่" loading="lazy"><figcaption>ภาพตรวจคนที่ถูกเตือน ใช้อ่านควบคู่กับผลตรวจ 10/10 ในบันทึก ไม่ใช่ผลทดสอบใหม่จากรายงานนี้</figcaption></figure>
<figure><img src="img/day3_audit_9_17_zoom.jpg" alt="ภาพขยายการตรวจคนที่ถูกเตือนในตัวอย่าง 9_17" loading="lazy"><figcaption>ภาพขยายตัวอย่าง 9_17 เพื่อให้แยกคนที่ระบบเลือกได้ง่ายขึ้น</figcaption></figure>
'@
}
$html = [System.Text.StringBuilder]::new()
[void]$html.AppendLine($intro + $styles + $extra)
foreach ($id in @('1','4b','2','3','4','5','6','7','8','9')) {
    if ($id -eq '2') { [void]$html.AppendLine($glossary) }
    $s=$sections[$id]
    $class=if($id -eq '4b'){'prob choice'}else{'prob'}
    [void]$html.AppendLine("<section class=`"$class`" id=`"s$id`"><h2>$id. $(Inline $s.title)</h2>")
    if($id -eq '4b') { [void]$html.AppendLine('<div class="choice-grid"><div class="choice-card"><h3>A · จับล้มได้มากกว่า</h3><p>ทีมแนะนำตามความสำคัญที่เจ้าของให้กับการล้มกลางวันและฉากหลายคน แต่เห็นสิ่งของเป็นคนมากกว่าเกณฑ์ที่ยังรอตัดสิน</p></div><div class="choice-card"><h3>B · เตือนผิดคนน้อยกว่า</h3><p>เห็นสิ่งของเป็นคนน้อยกว่า แต่จับล้มคลิปเจ้าของได้น้อยกว่า A และต่ำกว่าระบบเดิมเล็กน้อย</p></div></div>') }
    [void]$html.AppendLine((Body $s.lines))
    [void]$html.AppendLine("<div class=`"plain`"><p>$(Inline $notes[$id])</p></div>")
    if($media.ContainsKey($id)){[void]$html.AppendLine($media[$id])}
    [void]$html.AppendLine("<p class=`"note source`">ที่มาเนื้อหา: <a href=`"report_day3_outline.md#L$($s.start)`">โครงเนื้อหาสุดท้าย บรรทัด $($s.start)</a> · <a href=`"#evidence`">ไฟล์หลักฐานและคำสั่งเดิม</a></p></section>")
}
$end = @'
<section id="evidence" class="summary"><h2>หลักฐานและขอบเขตของรายงาน</h2>
<p>ตัวเลขยกมาจากผลเดิมที่ตรวจแล้วตามคำสั่งเจ้าของ รอบนี้สร้างหน้าเว็บและคัดลอกรูปเท่านั้น ไม่ได้ตรวจตัวเลขซ้ำ ไม่ได้ทดสอบ Le2i และไม่ได้วัดความเร็วใหม่</p>
<ul><li>ตาราง A/B: <a href="../.ai_evidence/verify_pose_choice_codex.txt">ผลตรวจเดิมพร้อมจำนวนคลิปและบรรทัดต้นทาง</a>; คำสั่งเดิม <code>&amp; ([scriptblock]::Create((Get-Content -Raw .ai_evidence/verify_pose_choice_codex.ps1)))</code>; แถวระบบจริง / A / B อยู่ใน <code>training/data/stage1/results_pinned.jsonl:10,36,55</code></li>
<li>เกณฑ์ T2: <a href="../.ai_evidence/verify_t2full_revised_codex.txt">ผลตรวจเดิม</a>; คำสั่งเดิม <code>&amp; ([scriptblock]::Create((Get-Content -Raw .ai_evidence/verify_t2full_revised_codex.ps1)))</code>; จำนวนคลิปใช้ตามคำอธิบายวิธีอ่านด้านบน</li>
<li>ภาพและคลิป: <a href="../.ai_evidence/day3_media/README.md">คำสั่งสร้าง สถานที่มา และจำนวนภาพ</a>; คำสั่งเดิม <code>python tools/day3_media.py chart</code>, <code>grid</code>, <code>clips</code>, <code>verify</code> (สั่งแยก); <a href="../.ai_evidence/day3_media/chart.json">กราฟและแถวต้นทาง</a>, <a href="../.ai_evidence/day3_media/grid.json">ภาพ COCO</a>, <a href="../.ai_evidence/day3_media/clips.json">คลิป</a>, <a href="../.ai_evidence/day3_media/verification.json">ผลตรวจไฟล์จากรอบก่อน</a></li>
<li>เหตุขัดข้องและมติทีม: <a href="../AI_HANDOFF.md#L3021">บันทึกทีมตั้งแต่บรรทัด 3021</a>; เรื่องที่ไม่ได้มีคำสั่งวัดในโครงเนื้อหาเป็นคำอธิบายตามบันทึก Claude ไม่ใช่การรับรองผลทดลองใหม่จากผู้จัดหน้า</li>
<li>รูปตรวจคนที่เตือน: คัดลอกจาก <code>../.ai_evidence/track_audit/sheet1.jpg</code> และ <code>9_17_zoom.jpg</code> เข้ามาในโฟลเดอร์ภาพของรายงาน โดยไม่ปรับรูป</li></ul></section>
<footer class="note">เนื้อหา: Claude · จัดหน้าและสื่อประกอบ: Codex · ขั้นถัดไป: Claude ตรวจรายงาน และ Gemini ตรวจความอ่านง่าย · <a href="report_day3_outline.md">อ่านโครงเนื้อหาต้นฉบับ</a></footer></main>
<script>document.getElementById('theme').addEventListener('click',function(){const r=document.documentElement;const dark=r.dataset.theme?r.dataset.theme==='dark':window.matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=dark?'light':'dark';this.textContent=dark?'เปลี่ยนเป็นธีมมืด':'เปลี่ยนเป็นธีมสว่าง';});</script></body></html>
'@
[void]$html.AppendLine($end)
[IO.File]::WriteAllText("$root/report/report_day3.html",$html.ToString(),[Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath "$root/../.ai_evidence/track_audit/sheet1.jpg" -Destination "$root/report/img/day3_audit_sheet1.jpg"
Copy-Item -LiteralPath "$root/../.ai_evidence/track_audit/9_17_zoom.jpg" -Destination "$root/report/img/day3_audit_9_17_zoom.jpg"
