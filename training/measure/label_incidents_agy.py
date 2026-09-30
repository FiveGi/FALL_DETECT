# -*- coding: utf-8 -*-
"""Propose labels for the incident segments with Antigravity CLI, reading the contact sheets.

The owner's Google AI Pro subscription is refused by Gemini Code Assist and Gemini CLI, but
accepted by the Antigravity CLI (`agy`), which also has a non-interactive `--print` mode and
`--json-schema` for structured answers. That makes it the one route by which a Pro-quota model
can label these segments without a person pasting images into a chat.

It reads `sheets/*.jpg` -- the frame contact sheets already rendered for every segment -- not
video, because the CLI's file reading is documented for images, not video.

Answers go into row['antigravity'], never into row['label']. A model's answer is a proposal;
`label` is set by a person, and this project has been wrong about unwatched footage before.
The definitions are those of label_incidents.py's PROMPT, word for word, so the two sources
can be compared rather than merely placed side by side.

Runs in `--mode plan` so the agent cannot edit the workspace. Resumable: segments that already
carry an answer from the same model are skipped.

Usage:
    LIMIT=10 AGY_MODEL=gemini-3.1-pro-high python training/measure/label_incidents_agy.py
    ONLY_WITH_GEMINI=1 ...   # restrict to segments the API pass already answered, for a pilot
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_incidents import PROMPT as CLIP_PROMPT  # noqa: E402  single source of the definitions

INCIDENTS = os.path.join(ROOT, 'test_result', 'incidents', 'incidents.json')
AGY = os.environ.get('AGY_BIN') or os.path.join(os.environ.get('LOCALAPPDATA', ''),
                                                 'agy', 'bin', 'agy.exe')
MODEL = os.environ.get('AGY_MODEL', 'gemini-3.1-pro-high')
LIMIT = int(os.environ.get('LIMIT', 10))
ONLY_WITH_GEMINI = os.environ.get('ONLY_WITH_GEMINI') == '1'
KEY = os.environ.get('AGY_KEY', 'antigravity')   # lets a second model be stored beside the first

SCHEMA = json.dumps({
    'type': 'object',
    'properties': {
        'can_see_image': {'type': 'boolean'},
        'fall': {'type': 'boolean'},
        'already_down': {'type': 'boolean'},
        'people': {'type': 'integer'},
        'elderly': {'type': 'boolean'},
        'out_of_domain': {'type': 'boolean'},
        'description': {'type': 'string'},
    },
    'required': ['can_see_image', 'fall', 'already_down', 'people', 'elderly',
                 'out_of_domain', 'description'],
})

PROMPT = (
    'Read the image file {sheet} in this workspace. It is a contact sheet: frames sampled in '
    'time order, left to right then top to bottom, from one short continuous video segment. '
    'Treat the frames as that clip. Do not create, edit or delete any file.\n'
    'Set "can_see_image" to false if you could not actually view the image contents.\n\n'
    + CLIP_PROMPT.replace('Answer ONLY compact JSON with exactly those keys.', '')
)


def ask(sheet):
    cmd = [AGY, '--mode', 'plan', '--model', MODEL, '--print-timeout', '240s',
           '--output-format', 'json', '--json-schema', SCHEMA,
           '-p', PROMPT.format(sheet=sheet)]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                         errors='replace', timeout=300)
    try:
        body = json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {'error': (out.stderr or out.stdout).strip()[-300:]}
    if body.get('status') != 'SUCCESS' or not body.get('structured_output'):
        return {'error': str(body.get('status')), 'raw': str(body.get('response'))[:300]}
    ans = dict(body['structured_output'])
    ans['model'] = MODEL
    ans['tokens'] = (body.get('usage') or {}).get('total_tokens')
    ans['seconds'] = round(body.get('duration_seconds') or 0, 1)
    return ans


def main():
    if not os.path.exists(AGY):
        raise SystemExit('agy not found at %s -- set AGY_BIN' % AGY)
    with open(INCIDENTS, encoding='utf-8') as fh:
        data = json.load(fh)

    todo = []
    for clip, rows in data['clips'].items():
        for row in rows:
            done = row.get(KEY)
            if done and not done.get('error') and done.get('model') == MODEL:
                continue
            if ONLY_WITH_GEMINI and not row.get('gemini'):
                continue
            todo.append((clip, row))
    print('%d segment(s) to ask, limit %d, model %s, stored as %r'
          % (len(todo), LIMIT, MODEL, KEY), flush=True)

    for clip, row in todo[:LIMIT]:
        sheet = row['sheet'].replace('\\', '/')
        row[KEY] = ask('test_result/incidents/' + sheet)
        with open(INCIDENTS, 'w', encoding='utf-8') as fh:     # after every answer: resumable
            json.dump(data, fh, indent=1, ensure_ascii=False)
        a = row[KEY]
        if a.get('error'):
            print('   %-8s seg %-3d ERROR %s' % (clip, row['segment'], a['error'][:80]), flush=True)
            continue
        print('   %-8s seg %-3d see=%-5s fall=%-5s down=%-5s ood=%-5s %5ss %6s tok  %s'
              % (clip, row['segment'], a['can_see_image'], a['fall'], a['already_down'],
                 a['out_of_domain'], a['seconds'], a['tokens'], a['description'][:44]),
              flush=True)


if __name__ == '__main__':
    main()
