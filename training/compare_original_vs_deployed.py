"""Compare the ORIGINAL detector with the deployed one, and write docs/original_vs_deployed.md.

"Original" means the first commit of this repository (`ce401fa`): MediaPipe pose extraction, a
30-frame window, threshold 0.50, 2-of-3 smoothing. Not the old classifier dropped into today's
pipeline -- the whole detector as it was, so the comparison is of two systems rather than two
weight files.

Both are fed the same clips at the same frame rate. That is the part most easily got wrong:
every number this project reported before SS50 came from reading every frame of a video file,
which a live camera never does, and the classifier's window is a fixed number of frames, so the
rate changes what the model sees. 15 fps is what `V3_TARGET_FPS` pins the camera loop to.

Surfaces, and why each is here:

  URFD           100 clips nothing here was ever trained or tuned on. The number to quote.
  GMDCSA24 val   held out of training, but used to pick settings many times over, so it reads
                 high for both columns and mainly shows neither model collapsed.
  GMDCSA24 falls all 79, most of which are training clips for the deployed model -- included
                 because a large drop would matter, not because a high score means much.

Inputs are the two per-clip JSON files produced by the runners (see the scratchpad scripts
referenced in SKILL.md SS55). Usage:

    python training/compare_original_vs_deployed.py ORIGINAL.json DEPLOYED.json
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


def load(path):
    d = json.load(open(path))
    return d['meta'], {(r[0], r[1]): r[2] for r in d['rows']}


def score(rows, group, want_alert):
    keys = [k for k in rows if k[0] == group]
    hit = sum(1 for k in keys if rows[k] == want_alert)
    return hit, len(keys)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    om, orig = load(sys.argv[1])
    dm, dep = load(sys.argv[2])

    lines = ['# The original detector versus the deployed one', '']
    lines += [
        'Both columns are the **whole detector**, not two weight files: the original is this',
        f"repository's first commit ({om.get('model', 'original')}) with its own pose extractor,",
        'window and alerting rule. Both are fed the same clips at the same frame rate, because',
        'the window is a fixed number of frames and the rate decides how much real time it',
        'covers — every number this project published before that was understood came from',
        'reading every frame of a file, which a live camera never does.',
        '',
        '| | original | deployed |',
        '|---|---|---|',
        f"| pose extractor | {om.get('pose', 'mediapipe')} | yolo26s-pose, input 960 |",
        f"| window | {om.get('window')} frames | {dm.get('window')} frames |",
        f"| threshold | {om.get('threshold')} | {dm.get('threshold')} |",
        f"| alerting rule | {om.get('need')} of {om.get('of')} windows | {dm.get('need')} of {dm.get('of')} windows |",
        f"| fed at | {om.get('fps'):.0f} fps | {dm.get('fps'):.0f} fps |",
        '',
        '## Results',
        '',
        '| what is measured | original | deployed | change |',
        '|---|---|---|---|',
    ]

    held_o = held_d = held_n = 0
    clean_o = clean_d = clean_n = 0
    for group, label, want_alert in GROUPS:
        o_hit, n = score(orig, group, want_alert)
        d_hit, _ = score(dep, group, want_alert)
        if n == 0:
            continue
        delta = d_hit - o_hit
        arrow = f'**{delta:+d}**' if delta else '0'
        lines.append(f'| {label} | {o_hit}/{n} | {d_hit}/{n} | {arrow} |')
        # "Never trained on" totals: URFD entirely, plus GMDCSA24's held-out ADL split.
        if group == 'urfd_fall':
            held_o, held_d, held_n = held_o + o_hit, held_d + d_hit, held_n + n
        if group in ('urfd_adl', 'val_adl'):
            clean_o, clean_d, clean_n = clean_o + o_hit, clean_d + d_hit, clean_n + n

    lines += [
        f'| **falls caught, data never trained on** | **{held_o}/{held_n}** '
        f'({held_o / held_n:.0%}) | **{held_d}/{held_n}** ({held_d / held_n:.0%}) | '
        f'**{held_d - held_o:+d}** |',
        f'| **clean, data never trained on** | **{clean_o}/{clean_n}** '
        f'({clean_o / clean_n:.0%}) | **{clean_d}/{clean_n}** ({clean_d / clean_n:.0%}) | '
        f'**{clean_d - clean_o:+d}** |',
        '',
    ]

    changed_caught = sorted(k[1] for k in orig
                            if k[0].endswith('_fall') and not orig[k] and dep.get(k))
    changed_missed = sorted(k[1] for k in orig
                            if k[0].endswith('_fall') and orig[k] and not dep.get(k))
    fp_fixed = sorted(k[1] for k in orig
                      if k[0].endswith('_adl') and orig[k] and not dep.get(k))
    fp_added = sorted(k[1] for k in orig
                      if k[0].endswith('_adl') and not orig[k] and dep.get(k))
    lines += [
        '## What changed, clip by clip',
        '',
        f'- falls the original missed and the deployed detector catches: **{len(changed_caught)}**',
        f'- falls the original caught and the deployed detector misses: **{len(changed_missed)}**'
        + (f' ({", ".join(changed_missed)})' if changed_missed and len(changed_missed) <= 12 else ''),
        f'- false alarms removed: **{len(fp_fixed)}**',
        f'- false alarms introduced: **{len(fp_added)}**'
        + (f' ({", ".join(fp_added)})' if fp_added and len(fp_added) <= 12 else ''),
        '',
        'Reproduce both columns with the runners named in SKILL.md SS55, then regenerate this',
        'file with `python training/compare_original_vs_deployed.py <original.json> <deployed.json>`.',
        '',
    ]

    dest = os.path.join(ROOT, 'docs', 'original_vs_deployed.md')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'\nwritten to {dest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
