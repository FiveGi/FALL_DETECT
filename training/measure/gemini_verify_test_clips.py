# -*- coding: utf-8 -*-
"""Ask Gemini what happens in a Test/ clip, for the compilations nobody had watched.

`Test/10` and `Test/11` produce zero alerts in every configuration tried, and the open-questions
list has carried "correct silence or a shared miss is unknown" for days because neither had ever
been looked at. Contact sheets answered that by eye; this is the second opinion the standing
rule asks for.

Unlike URFD these are social-media re-uploads: letterboxed, watermarked, with sticker overlays
and a TikTok end card of several seconds. The prompt therefore asks what the footage IS as well
as whether anybody falls, because "a fall the system should have caught" and "footage outside
what this system is for" are different answers and only one of them is a defect.

Usage:
    CLIPS="Test/10.mp4 Test/11.mp4" python gemini_verify_test_clips.py
"""
import json
import os
import sys
import time

from dotenv import load_dotenv

ROOT = r'D:\project\PROJECT\Backend-Elderly-Surveillance-main'
load_dotenv(os.path.join(ROOT, '.env'))
os.chdir(ROOT)
from google import genai  # noqa: E402

PROMPT = (
    "This is a short video. Judge the motion across the whole clip, not any single frame.\n"
    "Answer ONLY compact JSON with these keys:\n"
    '  "falls": how many times a person actually falls, collapses or loses balance to the '
    'ground (a number),\n'
    '  "seconds": when each fall happens,\n'
    '  "who": who falls - "elderly adult", "adult", "child", "toddler" or "nobody",\n'
    '  "setting": one short phrase for what the footage is (home camera, CCTV, phone video, '
    'news clip, compilation of several scenes...),\n'
    '  "quality": anything that would make a person hard to see - low resolution, night '
    "vision, letterboxing, overlays, subject very far away - or \"clear\",\n"
    '  "scenes": how many distinct scenes are cut together (a number).'
)


def main():
    names = os.environ.get('CLIPS', '').split()
    if not names:
        print('set CLIPS="Test/10.mp4 Test/11.mp4"')
        return
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    out = {}
    for name in names:
        up = client.files.upload(file=name)
        while up.state.name == 'PROCESSING':
            time.sleep(2)
            up = client.files.get(name=up.name)
        for attempt in range(6):
            try:
                r = client.models.generate_content(model='gemini-2.5-flash', contents=[up, PROMPT])
                break
            except Exception as e:
                if attempt == 5:
                    raise
                print('   %s: retry %d after %s' % (name, attempt + 1, type(e).__name__))
                time.sleep(15 * (attempt + 1))
        txt = r.text.strip().removeprefix('```json').removeprefix('```').removesuffix('```')
        try:
            v = json.loads(txt)
        except Exception:
            v = {'parse_fail': txt[:200]}
        out[name] = v
        print('\n%s' % name)
        for k, val in v.items():
            print('   %-9s %s' % (k, val))
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_watch',
                        'gemini_verdicts.json')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\nwrote', dest)


main()
