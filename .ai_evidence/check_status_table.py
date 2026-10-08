from pathlib import Path
from html.parser import HTMLParser
import re

root=Path(__file__).resolve().parents[1]
text=(root/'report/report_day2_3.html').read_text(encoding='utf-8')
class Parser(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=[]; self.headings=0; self.heading=False; self.heading_pills=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag in ['h2','h3']:self.heading=True
        if self.heading and tag=='span' and 'pill' in a.get('class','').split():self.heading_pills+=1
    def handle_endtag(self,tag):
        if tag in ['h2','h3']:self.heading=False
p=Parser();p.feed(text)
assert len(p.ids)==len(set(p.ids)), 'duplicate ids'
section=re.search(r'<section id="problem-status">(.*?)</section>',text,re.S)[1]
assert section.count('<tbody>')==4
assert section.count('<tr>')==26
assert section.count('<td>')==88
assert p.heading_pills==28
assert text.index('id="merged-summary"')<text.index('id="problem-status"')<text.index('id="merged-toc"')
for link in re.findall(r'href="([^"]+)"',section):
    path=link.split('#')[0]
    target=(root/'report'/path).resolve()
    assert target.exists(),link
    if '#L' in link:
        line=int(link.split('#L')[1]); assert line <= len(target.read_text(encoding='utf-8').splitlines()),link
reference=(root/'report/report_first_style_ref.html').read_text(encoding='utf-8')
for cls in ['pill','ok','wip','open']:
    pat=r'\.'+cls+r' \{[^}]+\}'
    assert re.search(pat,text)[0]==re.search(pat,reference)[0],cls
assert '.pill.tried' in text
assert not any(ord(c)<32 and c not in '\t\r\n' for c in text)
result='PASS: 22 rows / 4 groups / 4 columns; 28 heading pills; summary-table-TOC order; unique IDs; all new evidence paths/line bounds; first-report pill CSS identical; neutral tried CSS; no invalid control characters. HTMLParser only; no browser render or runtime/model tests.'
with (root/'.ai_evidence/status_table_checks.txt').open('a',encoding='utf-8') as f:f.write('Command: python .ai_evidence/check_status_table.py\n'+result+'\n')
print(result)
