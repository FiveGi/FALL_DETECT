# -*- coding: utf-8 -*-
"""Score any alert file on the owner's segments by ONE rule, so old and new are comparable.

Two hand-assembled tables disagreed on the multi-person count (14/8 vs 16/10) because each was
tallied separately. This is the only scorer: truth is a person's label first, Gemini's otherwise;
out-of-domain is skipped; "caught" = any alert inside a fall segment (eval_incidents_cpu.py's rule).

Accepts eval_incidents_cpu.py output ({'segments': {k: {'alerts_at': [...]}}}) and
eval_original.py output ({'segments': {k: [alert, peak]}}).

Usage: python training/measure/score_incidents.py FILE [FILE ...]
       python training/measure/score_incidents.py --paired 'BASE_*_phase*.json' 'CAND_*_phase*.json'
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INCIDENTS = os.path.join(ROOT, 'test_result', 'incidents', 'incidents.json')
MULTI = os.path.join(os.path.dirname(ROOT), '.ai_evidence', 'multi_person_genuine.txt')


def alerted(v):
    return bool(v['alerts_at']) if isinstance(v, dict) else bool(v[0])


def truth():
    data = json.load(open(INCIDENTS, encoding='utf-8'))
    out = {}
    for clip, rows in data['clips'].items():
        for r in rows:
            g = r.get('antigravity') or {}
            if r.get('label'):
                ood, fall = r['label'] == 'out_of_domain', r['label'] == 'fall'
            elif g and not g.get('error'):
                ood, fall = bool(g.get('out_of_domain')), bool(g.get('fall'))
            else:
                continue
            if not ood:
                out['%s#%d' % (clip, r['segment'])] = fall
    return out


def main(paths):
    t = truth()
    multi = [l.strip() for l in open(MULTI) if l.strip() and not l.startswith('#')] \
        if os.path.exists(MULTI) else []
    print('%-48s %7s %5s %9s' % ('file', 'caught', 'FA', 'multi'))
    for p in paths:
        seg = json.load(open(p, encoding='utf-8'))['segments']
        caught = sum(1 for k, f in t.items() if f and alerted(seg[k]))
        falls = sum(1 for f in t.values() if f)
        fa = sum(1 for k, f in t.items() if not f and alerted(seg[k]))
        mc = sum(1 for k in multi if alerted(seg[k]))
        print('%-48s %3d/%-3d %5d %4d/%-4d' % (os.path.basename(p)[:48], caught, falls, fa, mc, len(multi)))
    return 0


def paired(base_glob, cand_glob):
    """Phase-paired comparison (Codex, crop-state review): the crop schedule's phase is
    arbitrary in a live camera, so each setting is a DISTRIBUTION over phases. Pairs files by
    their `_phaseN` suffix and reports per-phase counts, mean/range, and which segments flip."""
    import glob
    import re
    t = truth()
    multi = [l.strip() for l in open(MULTI) if l.strip() and not l.startswith('#')] \
        if os.path.exists(MULTI) else []
    ph =lambda p: int(re.search(r'_phase(\d+)', p).group(1))
    base = {ph(p): p for p in glob.glob(base_glob)}
    cand = {ph(p): p for p in glob.glob(cand_glob)}
    rows, flips = [], {}
    for k in sorted(set(base) & set(cand)):
        r = []
        for p in (base[k], cand[k]):
            seg = json.load(open(p, encoding='utf-8'))['segments']
            r.append((sum(1 for s, f in t.items() if f and alerted(seg[s])),
                      sum(1 for s, f in t.items() if not f and alerted(seg[s])),
                      sum(1 for s in multi if alerted(seg[s])), seg))
        rows.append((k, r))
        for s, f in t.items():
            a, b = alerted(r[0][3][s]), alerted(r[1][3][s])
            if a != b:
                flips.setdefault(s, [f, 0, 0])[1 if b else 2] += 1
    print('phase   base caught/FA/multi   cand caught/FA/multi')
    for k, r in rows:
        print('%5d   %6d %3d %5d        %6d %3d %5d' % ((k,) + r[0][:3] + r[1][:3]))
    for i, name in ((0, 'base'), (1, 'cand')):
        for j, what in ((0, 'caught'), (1, 'FA'), (2, 'multi')):
            v = [r[i][j] for _k, r in rows]
            print('%s %-6s mean %.1f  range %d-%d' % (name, what, sum(v) / len(v), min(v), max(v)))
    print()
    print('segments whose answer differs (n phases cand gains / cand loses):')
    for s, (f, g, l) in sorted(flips.items()):
        print('  %-10s %-6s +%d -%d' % (s, 'fall' if f else 'NOFALL', g, l))
    return 0


if __name__ == '__main__':
    if sys.argv[1:2] == ['--paired']:
        sys.exit(paired(sys.argv[2], sys.argv[3]))
    sys.exit(main(sys.argv[1:]))
