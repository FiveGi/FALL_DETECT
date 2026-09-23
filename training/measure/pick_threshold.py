# -*- coding: utf-8 -*-
"""Give each classifier its own threshold before comparing it with another one.

**Why this exists.** Two classifiers trained by the same recipe with different seeds do not
produce the same numbers for the same window -- one of them is simply more confident than the
other across the board. Comparing them at a fixed threshold therefore measures where each
model's score distribution happens to sit, not which one separates falls from everything else.
Measured on the CPU profile: the deployed model at 0.65 catches 45 of URFD's 60 falls and the
same recipe retrained with the same seed catches 36 -- a gap of nine clips, three times the
"seed variance is 1-3 clips" this project had been working from. Give the second model its own
threshold and the two land on exactly the same place on the confirming half. The nine clips
were an operating point, not a detector.

**The rule, declared before the numbers are read.** For each model, choose the threshold that
maximises balanced accuracy on URFD half A -- `falls_A / n_falls_A + clean_A / n_clean_A` --
and then report half B at that threshold, never re-choosing on B. **Ties go to the lower
threshold**: a tie means half A cannot tell the two settings apart, and this system is built on
the position that missing a fall is worse than a false alarm -- `train.py` weights the loss 1.5
to 1 that way -- so the tie is broken towards catching more of them.

Input is one directory of per-clip JSON files per model, or a glob per model; each file is the
same model at a different threshold, as written by `replay_classifiers.py`.

Usage:
    python pick_threshold.py "baseline=perclip/cpu_fp0_seed42*.json" \
                             "frame position=perclip/cpu_fp1_seed42*.json"
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from urfd_split import in_half_a, in_half_b  # noqa: E402

EXPECTED_CLIPS = 220


def score(path):
    d = json.load(open(path))
    rows = d['rows']
    if len(rows) < EXPECTED_CLIPS:
        return None
    out = {'threshold': d['meta']['threshold'], 'path': path}
    for tag, keep in (('a', in_half_a), ('b', in_half_b)):
        f = [r for r in rows if r[0] == 'urfd_fall' and keep(r[1])]
        c = [r for r in rows if r[0] == 'urfd_adl' and keep(r[1])]
        out[tag] = (sum(1 for r in f if r[2]), len(f),
                    sum(1 for r in c if not r[2]), len(c))
    uf = [r for r in rows if r[0] == 'urfd_fall']
    held = [r for r in rows if r[0] in ('urfd_adl', 'val_adl')]
    val = [r for r in rows if r[0] == 'val_adl']
    out['urfd_falls'] = (sum(1 for r in uf if r[2]), len(uf))
    out['held_clean'] = (sum(1 for r in held if not r[2]), len(held))
    out['val_clean'] = (sum(1 for r in val if not r[2]), len(val))
    return out


def balanced(half):
    caught, n_f, clean, n_c = half
    return caught / n_f + clean / n_c


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cols = []
    for arg in sys.argv[1:]:
        name, _, pattern = arg.partition('=')
        if not pattern:
            name, pattern = os.path.basename(arg), arg
        results = [s for s in (score(p) for p in sorted(glob.glob(pattern))) if s]
        if not results:
            print('no complete result files matched %r' % pattern)
            return 1
        # Ties to the lower threshold -- see the rule above.
        best = max(results, key=lambda s: (balanced(s['a']), -s['threshold']))
        cols.append((name, best, len(results)))

    print('%-22s %6s | %-13s | %-13s | %-9s %-9s %-7s'
          % ('model', 'thr', 'half A (chosen)', 'half B (confirms)',
             'URFD all', 'held clean', 'val'))
    print('%-22s %6s | %-6s %-6s | %-6s %-6s | %-9s %-9s %-7s'
          % ('', '', 'falls', 'clean', 'falls', 'clean', '', '', ''))
    print('-' * 96)
    for name, s, n in cols:
        fa, nfa, ca, nca = s['a']
        fb, nfb, cb, ncb = s['b']
        print('%-22s %6.2f | %-6s %-6s | %-6s %-6s | %-9s %-9s %-7s'
              % (name[:22], s['threshold'],
                 '%d/%d' % (fa, nfa), '%d/%d' % (ca, nca),
                 '%d/%d' % (fb, nfb), '%d/%d' % (cb, ncb),
                 '%d/%d' % s['urfd_falls'], '%d/%d' % s['held_clean'],
                 '%d/%d' % s['val_clean']))
    print()
    print('Threshold chosen on half A by balanced accuracy; half B is only ever read after.')
    print('half B balanced accuracy: ' + ', '.join(
        '%s %.3f' % (name, balanced(s['b'])) for name, s, _n in cols))
    return 0


if __name__ == '__main__':
    sys.exit(main())
