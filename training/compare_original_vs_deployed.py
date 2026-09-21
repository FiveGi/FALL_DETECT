"""Compare detector configurations side by side, and write docs/original_vs_deployed.md.

"Original" means the first commit of this repository (`ce401fa`): MediaPipe pose extraction, a
30-frame window, threshold 0.50, 2-of-3 smoothing. Not the old classifier dropped into today's
pipeline -- the whole detector as it was, so the comparison is of two systems rather than two
weight files.

Every column is fed the same clips at the frame rate that column would actually run at. That is
the part most easily got wrong: every number this project reported before SS50 came from
reading every frame of a video file, which a live camera never does, and the classifier's
window is a fixed number of frames, so the rate changes how much real time the model sees.

Surfaces, and why each is here:

  URFD            100 clips nothing here was ever trained or tuned on. The number to quote.
  GMDCSA24 val    held out of training, but used to pick settings many times over, so it reads
                  high for every column and mainly shows nothing collapsed.
  GMDCSA24 falls  all 79, most of which are training clips for the deployed model -- included
                  because a large drop would matter, not because a high score means much.
  Test/13-16      four real falls, not acted, one fall each, so caught or missed. Confirmed
                  by watching them and by a second opinion from Gemini (SS60); this used to be
                  five clips, and `Test/17` was counted as a fall nothing could catch.
  Test/17         a crowd doing an outdoor exercise routine, and nobody falls in it. Silence
                  is the right answer, so it is scored as a clip that must NOT alert.
  Test/1-12       social-media compilations, several incidents per file. A count, not a score.

URFD is also split in half, because choosing an operating point by reading URFD would burn it
as an independent measure: one half is what a decision may be made on, the other only ever
confirms it afterwards. The split is by PAIRS of sequences, not by odd and even, and that
correction matters. URFD alternates its two fall types -- odd-numbered sequences are falls from
standing, even-numbered ones are falls out of a chair -- so an odd/even split, which is what
this project used until SS60, put every standing fall in one half and every chair fall in the
other and compared two different tasks. The deployed detector reads 28/30 on one and 13/30 on
the other for that reason alone. Taking sequences two at a time keeps both fall types, and both
cameras, in both halves.

Usage:

    python training/compare_original_vs_deployed.py ORIGINAL.json DEPLOYED.json

    python training/compare_original_vs_deployed.py "original=a.json" "deployed=b.json"
        "30-frame @24fps=c.json" --all17 all17_results.json --window-fill fill.json

A bare path takes its name from its position (first is the original, second the deployed);
`name=path` sets the column heading explicitly. `--all17` may be repeated, and each file
contributes its configurations to the Test/ tables.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROUPS = [
    ('urfd_fall', 'URFD — falls caught', True),
    ('urfd_adl', 'URFD — normal clips with no false alarm', False),
    ('val_adl', 'GMDCSA24 val — normal clips with no false alarm', False),
    ('gmdcsa_fall', 'GMDCSA24 — falls caught (mostly training clips)', True),
    ('train50_adl', 'GMDCSA24 train50 — normal clips clean (training clips)', False),
]
# Test/13-16 each contain exactly one real fall; Test/17 contains none. It sat in the "real
# elderly falls" group for days as a clip every configuration missed, which made the deployed
# detector look worse than it is and made a 4/5 configuration look better. Watching it settles
# it: it is a crowd of people doing an outdoor exercise routine, filmed from across a
# courtyard, with a lot of deep-squat motion and nobody falling. Gemini reads it the same way.
REAL = ['Test/%d.mp4' % n for n in range(13, 17)]
NEGATIVE = ['Test/17.mp4']
COMPILATION = ['Test/%d.mp4' % n for n in range(1, 13)]


def load(path):
    d = json.load(open(path))
    return d['meta'], {(r[0], r[1]): r[2] for r in d['rows']}


def score(rows, group, want_alert, keep=None):
    keys = [k for k in rows if k[0] == group and (keep is None or keep(k[1]))]
    return sum(1 for k in keys if rows[k] == want_alert), len(keys)


def clip_index(name):
    """`fall-07-cam0-rgb.mp4` -> 7. URFD's split-half key."""
    digits = ''.join(c if c.isdigit() else ' ' for c in name).split()
    return int(digits[0]) if digits else 0


def row(cells):
    return '| ' + ' | '.join(str(c) for c in cells) + ' |'


def describe(meta):
    return [meta.get('pose', 'yolo26s-pose, input 960'),
            '%s frames' % meta.get('window'),
            meta.get('threshold'),
            '%s of %s windows' % (meta.get('need'), meta.get('of')),
            '%.0f fps' % meta.get('fps', 0)]


def lab_section(cols):
    """The 220-clip lab surface: one column per configuration, plus held-out totals."""
    names = [n for n, _, _ in cols]
    out = ['## Results on the lab datasets', '',
           row(['what is measured'] + names), row(['---'] * (len(names) + 1))]
    held = [0] * len(cols)
    clean = [0] * len(cols)
    held_n = clean_n = 0
    for group, label, want in GROUPS:
        hits, n = [], 0
        for _, _, rows in cols:
            h, n = score(rows, group, want)
            hits.append(h)
        if n == 0:
            continue
        out.append(row([label] + ['%d/%d' % (h, n) for h in hits]))
        if group == 'urfd_fall':
            held = [a + b for a, b in zip(held, hits)]
            held_n += n
        if group in ('urfd_adl', 'val_adl'):
            clean = [a + b for a, b in zip(clean, hits)]
            clean_n += n
    out.append(row(['**falls caught, data never trained on**']
                   + ['**%d/%d** (%.0f%%)' % (h, held_n, 100.0 * h / held_n) for h in held]))
    out.append(row(['**clean, data never trained on**']
                   + ['**%d/%d** (%.0f%%)' % (c, clean_n, 100.0 * c / clean_n) for c in clean]))
    out.append('')
    return out


def split_half_section(cols):
    """URFD split by PAIRS of sequences -- choose on half A, confirm on half B, never reversed.

    Pairs rather than parity: URFD alternates fall type by sequence number, so an odd/even
    split separates falls from standing from falls out of a chair and the two halves are not
    the same task. Sequences taken two at a time put both types in both halves.
    """
    names = [n for n, _, _ in cols]
    out = ['### URFD split in half', '',
           'Half A is what a decision may be made on; half B only confirms it. A configuration',
           'that wins on A but not on B won on noise.', '',
           'The split takes sequences two at a time rather than by odd and even, because URFD',
           'alternates its fall types: odd-numbered sequences are falls from standing and',
           'even-numbered ones are falls out of a chair, so an odd/even split compares two',
           'different tasks rather than two samples of one.', '',
           row(['half'] + names), row(['---'] * (len(names) + 1))]
    for label, keep in [('A — may choose on', lambda n: ((clip_index(n) - 1) // 2) % 2 == 0),
                        ('B — confirms only', lambda n: ((clip_index(n) - 1) // 2) % 2 == 1)]:
        for group, what, want in [('urfd_fall', 'falls caught', True),
                                  ('urfd_adl', 'clean', False)]:
            cells, n = [], 0
            for _, _, rows in cols:
                h, n = score(rows, group, want, keep)
                cells.append(h)
            out.append(row(['%s, %s' % (label, what)] + ['%d/%d' % (c, n) for c in cells]))
    out.append('')
    return out


def window_fill_section(cols, fill):
    """How many URFD falls a column cannot score at all, because the window never fills.

    `fill` maps clip name -> [source fps, one character per frame: was anybody detected]. The
    pose pass does not depend on the sampling rate, so one bitmap answers the question for
    every (window, rate) pair by arithmetic.
    """
    if not fill:
        return []
    def fills(name, window, target_fps):
        src, bits = fill[name]
        kept, last_slot = [], -1
        for i, b in enumerate(bits):
            slot = int(i * target_fps / src)
            if slot == last_slot:
                continue
            last_slot = slot
            kept.append(b)
        return kept.count('1') >= window

    falls = sorted(n for n in fill if n.startswith('fall-'))
    names = [n for n, _, _ in cols]
    out = ['### How many URFD falls each column can score at all', '',
           'The classifier cannot produce a number until its window holds a full set of frames',
           'with a person in them, and some URFD clips end before that happens. The ceiling',
           'camera on the standing falls is the clear case: the room is empty for two thirds of',
           'the clip and the person walks into view as they land, so at 15 fps there are 43',
           'sampled frames, a person in 14 of them, and a window that needs 15. Those clips',
           'score exactly 0.00 — not a fall the model rejected, a fall it was never shown.',
           '',
           'A longer window at a lower rate needs more real time of visible person before it can',
           'say anything at all, so this is part of what the recall column is measuring.', '',
           row(['column', 'window must cover', 'falls it cannot score', 'recall over the rest']),
           row(['---'] * 4)]
    for (name, meta, rows) in cols:
        window, fps = int(meta.get('window')), float(meta.get('fps'))
        unscorable = [n for n in falls if not fills(n, window, fps)]
        rest = [n for n in falls if n not in unscorable]
        caught = sum(1 for n in rest if rows.get(('urfd_fall', n)))
        out.append(row([name, '%.2f s' % (window / fps), '%d/%d' % (len(unscorable), len(falls)),
                        '%d/%d (%.0f%%)' % (caught, len(rest), 100.0 * caught / max(1, len(rest)))]))
    out.append('')
    return out


def fall_type_section(cols):
    """URFD's two fall types, which the odd/even split used to hide inside one number."""
    names = [n for n, _, _ in cols]
    out = ['### URFD by fall type', '',
           'URFD recorded two kinds of fall and alternates them by sequence number. They are',
           'not equally hard, and reporting only the total hides which one a change helped.',
           'Falls from standing are also the ones the ceiling camera barely sees: the room is',
           'empty for two thirds of those clips and the person walks into view as they land.',
           '',
           row(['fall type'] + names), row(['---'] * (len(names) + 1))]
    for label, parity in [('from standing (odd sequences)', 1),
                          ('out of a chair (even sequences)', 0)]:
        cells, n = [], 0
        for _, _, rows in cols:
            h, n = score(rows, 'urfd_fall', True,
                         keep=lambda x, p=parity: clip_index(x) % 2 == p)
            cells.append(h)
        out.append(row([label] + ['%d/%d' % (c, n) for c in cells]))
    out.append('')
    return out


def all17_section(all17):
    """Test/ clips. Real falls are pass/fail; compilations are counts and are labelled as such."""
    if not all17:
        return []
    names = list(all17.keys())
    out = ['## Results on the Test/ clips', '',
           'These are the only footage here that is not a lab dataset. `13`-`16` are real falls,',
           'one each, so a column is simply caught or missed. `17` has no fall in it at all — a',
           'crowd doing an outdoor exercise routine — so the right answer there is silence, and',
           'it is scored that way rather than counted as a miss. `1`-`12` are compilations cut',
           'from social media with several incidents and long stretches of ordinary activity',
           'between them, so an alert count there is neither right nor wrong on its own and is',
           'reported as a count.', '',
           '### Real falls (`Test/13`-`16`), and one clip that must stay silent', '',
           row(['configuration'] + [os.path.basename(c) for c in REAL] + ['caught']
               + [os.path.basename(c) + ' (no fall)' for c in NEGATIVE]),
           row(['---'] * (len(REAL) + len(NEGATIVE) + 2))]
    for name in names:
        r = all17[name]
        cells = []
        for c in REAL:
            a, p = r.get(c, [0, 0.0])
            cells.append(('**caught** %.2f' % p) if a else ('missed %.2f' % p))
        cells.append('%d/%d' % (sum(1 for c in REAL if r.get(c, [0])[0]), len(REAL)))
        for c in NEGATIVE:
            a, p = r.get(c, [0, 0.0])
            cells.append(('**false alarm** %.2f' % p) if a else ('silent %.2f' % p))
        out.append(row([name] + cells))
    out += ['', 'The number after each verdict is the highest score that configuration reached on',
            'the clip, because missed-by-a-little and never-came-close are different problems.',
            '', '### Compilations (`Test/1`-`12`) — alerts raised, not a score', '',
            row(['configuration'] + [os.path.basename(c).replace('.mp4', '') for c in COMPILATION]
                + ['total']),
            row(['---'] * (len(COMPILATION) + 2))]
    for name in names:
        r = all17[name]
        counts = [r.get(c, [0])[0] for c in COMPILATION]
        out.append(row([name] + counts + [sum(counts)]))
    out.append('')
    return out


def changed_section(cols):
    """Clip-by-clip movement between the first column and each later one."""
    base_name, _, base = cols[0]
    out = ['## What changed, clip by clip', '']
    for name, _, rows in cols[1:]:
        caught = sorted(k[1] for k in base if k[0].endswith('_fall') and not base[k] and rows.get(k))
        missed = sorted(k[1] for k in base if k[0].endswith('_fall') and base[k] and not rows.get(k))
        fixed = sorted(k[1] for k in base if k[0].endswith('_adl') and base[k] and not rows.get(k))
        added = sorted(k[1] for k in base if k[0].endswith('_adl') and not base[k] and rows.get(k))
        out += ['**%s** against %s:' % (name, base_name), '',
                '- falls %s missed and this catches: **%d**' % (base_name, len(caught)),
                '- falls %s caught and this misses: **%d**' % (base_name, len(missed))
                + ((' (%s)' % ', '.join(missed)) if missed and len(missed) <= 12 else ''),
                '- false alarms removed: **%d**' % len(fixed),
                '- false alarms introduced: **%d**' % len(added)
                + ((' (%s)' % ', '.join(added)) if added and len(added) <= 12 else ''),
                '']
    return out


def verdict_section():
    """What the tables add up to.

    The speed figures come from `scratchpad/bench_original_vs_deployed_speed.py`, which times
    both detectors on the same 1080p frames on an otherwise idle machine, and the collapse-rule
    A/B from `scratchpad/original/original_no_collapse.py`. They are quoted here because the
    per-clip JSON files cannot contain them, and both scripts are re-runnable.
    """
    return [
        '## What this adds up to', '',
        'The deployed detector is the right one to run **on this machine**, and the reason is',
        'not that its model is better everywhere.', '',
        '**Fed every frame, the original is better on URFD than anything deployed since** —',
        '54/60 falls (90%) against 41/60 (68%). That is not a trick of its alerting rule: its',
        '"the person vanished" collapse rule was switched off and re-run, and URFD recall was',
        '**54/60 either way**, so the recall is the classifier. Nor is it a threshold artefact:',
        'at a threshold loose enough to give a *better* clean rate than the original (0.50,',
        '36/56 clean against 33/56), the deployed model still catches only 43/60.', '',
        '**It cannot be fed every frame.** Timed on the same 1080p frames on an idle machine,',
        'the original needs **70.3 ms per frame (14.2 fps)** against the deployed detector’s',
        '**54.4 ms (18.4 fps)** — MediaPipe on the CPU is slower than YOLO26s-pose on the GPU,',
        'before decoding or anything else in the loop. 30 fps was never available to it. At the',
        'rate it can actually sustain, its own numbers are **27/60 (45%)**.', '',
        'So the honest summary of the rewrite is: **the system now works at the speed it',
        'really runs**, which is worth +14 falls and +6 clean clips at 15 fps, not that the',
        'model learned more. The 90% figure is what a faster machine would be worth, and it is',
        'the strongest argument in this document for spending effort on frame rate rather than',
        'on the classifier.', '',
        'The 30-frame model at 24 fps was measured to settle whether to move back to it. It is',
        'behind on URFD falls (39/60), behind on URFD clean (30/40), behind or level on both',
        'halves of the split, and has more clips it cannot score (9 against 7). It wins one',
        'real-footage clip, `Test/13` — a moving, zooming camera on a young adult who catches',
        'himself on his hands, measured at 4.8 px/frame of background motion against 0.03 for',
        'the fixed-camera clips. Nothing in this pipeline is built for a moving camera, and',
        'seed variance alone moves one to three clips per surface. **The deployed',
        'configuration stays.**', '',
    ]


def main():
    args = list(sys.argv[1:])
    all17_paths = []
    fill_path = None
    if '--window-fill' in args:
        i = args.index('--window-fill')
        fill_path = args[i + 1]
        del args[i:i + 2]
    while '--all17' in args:
        i = args.index('--all17')
        all17_paths.append(args[i + 1])
        del args[i:i + 2]
    if len(args) < 2:
        print(__doc__)
        return 1

    cols = []
    for i, a in enumerate(args):
        name, _, path = a.partition('=')
        if not path:
            path = a
            name = ('original', 'deployed')[i] if i < 2 else 'column %d' % (i + 1)
        meta, rows = load(path)
        cols.append((name, meta, rows))

    all17 = {}
    for p in all17_paths:
        all17.update(json.load(open(p)))
    fill = json.load(open(fill_path)) if fill_path else None

    names = [n for n, _, _ in cols]
    lines = ['# Detector configurations compared', '',
             'Every column is the **whole detector**, not a weight file: the original column is',
             "this repository's first commit, with its own pose extractor, window and alerting",
             'rule. Each is fed the same clips at the frame rate it would actually run at,',
             'because the window is a fixed number of frames and the rate decides how much real',
             'time it covers — every number published here before that was understood came from',
             'reading every frame of a file, which a live camera never does.', '',
             row([''] + names), row(['---'] * (len(names) + 1))]
    fields = ['pose extractor', 'window', 'threshold', 'alerting rule', 'fed at']
    described = [describe(m) for _, m, _ in cols]
    for j, field in enumerate(fields):
        lines.append(row([field] + [d[j] for d in described]))
    lines.append('')

    lines += lab_section(cols)
    lines += window_fill_section(cols, fill)
    lines += fall_type_section(cols)
    lines += split_half_section(cols)
    lines += all17_section(all17)
    lines += changed_section(cols)
    lines += verdict_section()
    lines += ['Reproduce the columns with the runners named in SKILL.md SS55 and SS60, then',
              'regenerate this file with',
              '`python training/compare_original_vs_deployed.py "name=a.json" "name=b.json" ...`.',
              '']

    dest = os.path.join(ROOT, 'docs', 'original_vs_deployed.md')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print('\nwritten to %s' % dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
