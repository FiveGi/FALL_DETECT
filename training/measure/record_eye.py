# -*- coding: utf-8 -*-
"""Record a by-eye verdict on an incident segment, and set `label` only when two sources agree.

Gemini proposes and a person confirms. This writes the person's half into `by_eye` and then
decides `label` by a rule rather than by whoever wrote last:

    by_eye and gemini agree            -> label is that verdict
    they disagree                      -> label stays null, `conflict` records both
    by_eye says unclear                -> label stays null
    only one source has an opinion     -> label stays null

**"unclear" is a first-class answer.** Several of these segments are night footage where six
thumbnails cannot settle whether somebody fell, and a guess there is worse than a gap: a gap
gets revisited, a wrong label gets measured against forever. `Test/17` sat in this project's
records as a fall nobody could catch until somebody finally watched it.

Usage:
    python training/measure/record_eye.py 5.mp4 3 fall
    python training/measure/record_eye.py 9.mp4 14 unclear "too dark to judge from stills"
    python training/measure/record_eye.py --status

Verdicts: fall | no_fall | already_down | out_of_domain | unclear
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INCIDENTS = os.path.join(ROOT, 'test_result', 'incidents', 'incidents.json')
VERDICTS = ('fall', 'no_fall', 'already_down', 'out_of_domain', 'unclear')


def gemini_verdict(g):
    """Gemini's answer reduced to the same vocabulary, so the two can be compared at all."""
    if not g or 'parse_fail' in g:
        return None
    if g.get('out_of_domain'):
        return 'out_of_domain'
    if g.get('fall'):
        return 'fall'
    if g.get('already_down'):
        return 'already_down'
    return 'no_fall'


def load():
    with open(INCIDENTS, encoding='utf-8') as fh:
        return json.load(fh)


def save(data):
    with open(INCIDENTS, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)


def status(data):
    rows = [r for v in data['clips'].values() for r in v]
    eye = [r for r in rows if r.get('by_eye')]
    gem = [r for r in rows if r.get('gemini')]
    lab = [r for r in rows if r.get('label')]
    con = [r for r in rows if r.get('conflict')]
    unclear = [r for r in eye if r['by_eye']['verdict'] == 'unclear']
    print('%d segments   %d alerted' % (len(rows), sum(1 for r in rows if r['alerts_at'])))
    print('  Gemini answered      %d' % len(gem))
    print('  looked at by eye     %d  (of which unclear: %d)' % (len(eye), len(unclear)))
    print('  AGREED -> label set  %d' % len(lab))
    print('  disagreed            %d' % len(con))
    if con:
        print()
        for r in con:
            print('   conflict: %s' % r['conflict'])


def main():
    if '--status' in sys.argv:
        status(load())
        return 0
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    clip, segment, verdict = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    note = sys.argv[4] if len(sys.argv) > 4 else ''
    if verdict not in VERDICTS:
        print('verdict must be one of: %s' % ', '.join(VERDICTS))
        return 1

    data = load()
    rows = data['clips'].get(clip)
    if rows is None:
        print('no such clip: %s' % clip)
        return 1
    row = next((r for r in rows if r['segment'] == segment), None)
    if row is None:
        print('no segment %d in %s' % (segment, clip))
        return 1

    row['by_eye'] = {'verdict': verdict, 'note': note}
    other = gemini_verdict(row.get('gemini'))
    row.pop('conflict', None)
    if verdict == 'unclear' or other is None:
        row['label'] = None
    elif other == verdict:
        row['label'] = verdict
    else:
        row['label'] = None
        row['conflict'] = '%s seg %d: by eye %s, Gemini %s' % (clip, segment, verdict, other)
    save(data)
    print('%s seg %-3d  by eye: %-14s Gemini: %-14s label: %s'
          % (clip, segment, verdict, other or '(not asked)', row['label'] or 'still null'))
    if row.get('conflict'):
        print('   they disagree, so the label stays null and the segment needs watching')
    return 0


if __name__ == '__main__':
    sys.exit(main())
