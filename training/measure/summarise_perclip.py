# -*- coding: utf-8 -*-
"""Summarise per-clip result files, with URFD kept split so a choice cannot spend it.

Reads whatever `rule_sweep_perclip.py` or `replay_classifiers.py` wrote and prints one row per
configuration. URFD is shown as two columns, A and B: **A is the only half a decision may be
made on, B is read afterwards to confirm it.** A configuration that wins on A and not on B won
on noise -- that has happened here, and reading all hundred clips to pick a setting is how the
project lost URFD's independence once already.

GMDCSA24 is printed in full because it has been tuned against many times over and has no
independence left to protect; it is here to show that nothing collapsed, not to choose with.

Usage:
    python summarise_perclip.py perclip/*.json
    python summarise_perclip.py --sort urfd_a perclip/cpu_thr*.json
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from urfd_split import in_half_a, in_half_b  # noqa: E402


# The five groups rule_sweep_perclip.py and replay_classifiers.py walk, totalled.
EXPECTED_CLIPS = 220


def summarise(path):
    d = json.load(open(path))
    rows = d['rows']
    out = {}
    for tag, keep in (('a', in_half_a), ('b', in_half_b)):
        f = [r for r in rows if r[0] == 'urfd_fall' and keep(r[1])]
        a = [r for r in rows if r[0] == 'urfd_adl' and keep(r[1])]
        out['urfd_%s_falls' % tag] = (sum(1 for r in f if r[2]), len(f))
        out['urfd_%s_clean' % tag] = (sum(1 for r in a if not r[2]), len(a))
    for key, group, want in (('gmdcsa_falls', 'gmdcsa_fall', True),
                             ('val_clean', 'val_adl', False),
                             ('train50_clean', 'train50_adl', False)):
        g = [r for r in rows if r[0] == group]
        out[key] = (sum(1 for r in g if r[2] == want), len(g))
    # The headline pair: every fall on the dataset nothing here was tuned against, and every
    # clean clip that was held out of training, which is the false-alarm surface that matters.
    # These files are written after every group so a long sweep can resume, which means a
    # run still in progress is a valid JSON file holding a partial answer. Summarising one
    # silently would read as a configuration that lost badly.
    out['n_rows'] = len(rows)
    uf = [r for r in rows if r[0] == 'urfd_fall']
    held = [r for r in rows if r[0] in ('urfd_adl', 'val_adl')]
    out['urfd_falls'] = (sum(1 for r in uf if r[2]), len(uf))
    out['held_clean'] = (sum(1 for r in held if not r[2]), len(held))
    return d['meta'], out


def fmt(t):
    return '%d/%d' % t


def main():
    paths = []
    args = sys.argv[1:]
    sort_key = None
    if args and args[0] == '--sort':
        sort_key = args[1]
        args = args[2:]
    for a in args:
        paths.extend(sorted(glob.glob(a)) or [a])
    if not paths:
        print(__doc__)
        return 1

    table = []
    for p in paths:
        meta, s = summarise(p)
        table.append((os.path.basename(p).replace('.json', ''), meta, s))
    if sort_key:
        table.sort(key=lambda r: -r[2][sort_key][0])

    head = ('%-30s | %-6s %-6s | %-6s %-6s | %-7s %-7s | %-7s %-6s %-6s'
            % ('configuration', 'A fall', 'A clean', 'B fall', 'B clean',
               'URFD all', 'held cln', 'GMDCSA', 'val', 'train50'))
    print('%-30s | %-13s | %-13s | %-15s | %s'
          % ('', 'URFD half A', 'URFD half B', 'URFD + held out', 'tuned against, shown for collapse'))
    print('%-30s | %-13s | %-13s | %-15s |'
          % ('', 'choose on this', 'confirms only', ''))
    print(head)
    print('-' * len(head))
    partial = [n for n, _m, s in table if s['n_rows'] < EXPECTED_CLIPS]
    for name, meta, s in table:
        if s['n_rows'] < EXPECTED_CLIPS:
            name = name + ' (incomplete)'
        print('%-30s | %-6s %-6s | %-6s %-6s | %-7s %-7s | %-7s %-6s %-6s'
              % (name[:30], fmt(s['urfd_a_falls']), fmt(s['urfd_a_clean']),
                 fmt(s['urfd_b_falls']), fmt(s['urfd_b_clean']),
                 fmt(s['urfd_falls']), fmt(s['held_clean']),
                 fmt(s['gmdcsa_falls']), fmt(s['val_clean']), fmt(s['train50_clean'])))
    print()
    print('A is the half a decision may be made on. B only confirms it afterwards.')
    if partial:
        print('%d file(s) hold fewer than %d clips and are still being written -- their rows '
              'are not a result.' % (len(partial), EXPECTED_CLIPS))
    return 0


if __name__ == '__main__':
    sys.exit(main())
