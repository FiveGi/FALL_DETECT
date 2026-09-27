# -*- coding: utf-8 -*-
"""Ask Gemini what is in each `Test/` incident segment, so 112 segments can become ground truth.

This is the first of two passes. Gemini proposes; a person confirms against the contact sheet
already written beside each segment. Nothing here writes a `label` -- it writes `gemini`, and
`label` stays null until somebody looks. On this project's own record that is not ceremony:
`Test/17` was carried as "a fall every configuration misses" for days before anyone watched it
and found a crowd doing an exercise routine, and a clip a verification pass called a caught
fall turned out to be aftermath, with the fall itself outside the window.

Each segment is cut out and uploaded on its own. That matters: these are compilations that cut
between unrelated incidents every few seconds, and SS49 found Gemini answering about somebody
else's fall when a window spanned three scenes. A segment is one shot by construction.

**Resumable, because the free tier is 20 requests a day.** Every answer is written immediately
and an existing answer is never asked again, so this can be run once a day for a week, or in
one go on a paid key. Segments where the detector alerted are asked first -- they are the half
that gives precision, which is the more useful half to have early.

Usage:
    python training/measure/label_incidents.py            # as many as the quota allows
    LIMIT=5 python training/measure/label_incidents.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
load_dotenv(os.path.join(ROOT, '.env'))
from google import genai  # noqa: E402

INCIDENTS = os.path.join(ROOT, 'test_result', 'incidents', 'incidents.json')
LIMIT = int(os.environ.get('LIMIT', 20))
MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')

PROMPT = (
    "This is a few seconds of real footage, one continuous shot, from a CCTV or phone camera.\n"
    "Answer about THIS clip only.\n"
    '- "fall": true only if a person actually falls to the ground during the clip — loses '
    'their balance and goes down. A person who is ALREADY on the ground when the clip starts '
    'and does not fall during it is NOT a fall here; use "already_down" for that.\n'
    '- "already_down": true if somebody is on the floor for most of the clip without falling '
    'in it.\n'
    '- "people": how many people are visible (a number).\n'
    '- "elderly": true if the person who falls, or the main person, looks elderly.\n'
    '- "out_of_domain": true if this is not the kind of footage a home fall-detection camera '
    'would see — a title card, a cartoon, sport, stunts, a crowd scene, a camera that pans or '
    'zooms, or a child rather than an adult.\n'
    '- "description": one short phrase of what happens.\n'
    'Answer ONLY compact JSON with exactly those keys.'
)


def cut(src, start_s, end_s, dest):
    """One segment as its own file. ffmpeg if it is there, OpenCV otherwise."""
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(start_s), '-i', src,
             '-t', str(max(0.4, end_s - start_s)), '-c:v', 'libx264', '-preset', 'veryfast',
             '-an', dest],
            check=True, capture_output=True)
        return os.path.exists(dest) and os.path.getsize(dest) > 0
    except Exception:
        pass
    import cv2
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for _ in range(int((end_s - start_s) * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        out.write(frame)
    cap.release()
    out.release()
    return os.path.exists(dest) and os.path.getsize(dest) > 0


def ask(client, path):
    up = client.files.upload(file=path)
    while up.state.name == 'PROCESSING':
        time.sleep(2)
        up = client.files.get(name=up.name)
    for attempt in range(4):
        try:
            r = client.models.generate_content(model=MODEL, contents=[up, PROMPT])
            break
        except Exception as exc:
            name = type(exc).__name__
            # A quota refusal is not a transient error and retrying it burns the rest of the
            # run's time for nothing. Stop and let the next day's run continue.
            if 'RESOURCE_EXHAUSTED' in str(exc) or '429' in str(exc):
                raise SystemExit('quota exhausted -- run again tomorrow, progress is saved')
            if attempt == 3:
                raise
            print('      retry %d after %s' % (attempt + 1, name), flush=True)
            time.sleep(15 * (attempt + 1))
    txt = r.text.strip().removeprefix('```json').removeprefix('```').removesuffix('```')
    try:
        return json.loads(txt)
    except Exception:
        return {'parse_fail': txt[:200]}


def main():
    with open(INCIDENTS, encoding='utf-8') as fh:
        data = json.load(fh)

    todo = []
    for clip, rows in data['clips'].items():
        for row in rows:
            if row.get('gemini') is None:
                todo.append((clip, row))
    # Alerted segments first: they are the half that gives precision.
    todo.sort(key=lambda cr: (not cr[1]['alerts_at'], cr[0], cr[1]['segment']))
    if not todo:
        print('every segment already has an answer')
        return

    print('%d segments without an answer; asking up to %d' % (len(todo), LIMIT))
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    asked = 0
    with tempfile.TemporaryDirectory() as tmp:
        for clip, row in todo[:LIMIT]:
            src = os.path.join('Test', clip)
            dest = os.path.join(tmp, '%s_%02d.mp4' % (clip.replace('.mp4', ''), row['segment']))
            if not cut(src, row['start_s'], row['end_s'], dest):
                print('   %s seg %d: could not be cut, skipped' % (clip, row['segment']))
                continue
            row['gemini'] = ask(client, dest)
            asked += 1
            with open(INCIDENTS, 'w', encoding='utf-8') as fh:
                json.dump(data, fh, indent=1, ensure_ascii=False)
            g = row['gemini']
            print('   %-9s seg %-3d %5.1fs  %-7s  fall=%-5s down=%-5s ood=%-5s  %s'
                  % (clip, row['segment'], row['seconds'],
                     'ALERTED' if row['alerts_at'] else 'silent',
                     g.get('fall'), g.get('already_down'), g.get('out_of_domain'),
                     str(g.get('description'))[:46]), flush=True)

    done = sum(1 for rows in data['clips'].values() for r in rows if r.get('gemini'))
    total = sum(len(rows) for rows in data['clips'].values())
    print()
    print('asked %d this run. %d of %d segments now have a Gemini answer.' % (asked, done, total))
    print('Every "label" is still null. Gemini proposes; the contact sheet beside each segment')
    print('is what confirms it, and this project has been wrong about unwatched clips before.')


if __name__ == '__main__':
    main()
