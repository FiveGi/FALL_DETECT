"""Turn Gemini's per-frame verdicts into a precision comparison between two alert sets.

Caught/clean counts from dataset labels only say whether a clip contains a fall. They cannot
say whether the system fired at the right moment or for the right reason -- which is the
difference between "more alerts" and "better detection". This reads the verdict files written
by verify_production_alerts.py and reports, per alert set, how many alerts Gemini judged to be
real falls.

Usage:
    python training/summarize_gemini_verification.py old new
"""
import json
import os
import re
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def verdicts(tag):
    path = os.path.join(DATA, f'prod_{tag}_verify_results.json')
    if not os.path.exists(path):
        raise SystemExit(f'missing {path} -- run verify_production_alerts.py with VERIFY_TAG={tag}')
    raw = json.load(open(path, encoding='utf-8'))
    out = {}
    for key, text in raw.items():
        m = re.search(r'"verdict"\s*:\s*"([A-Z_]+)"', text or '')
        reason = re.search(r'"reason"\s*:\s*"([^"]*)"', text or '')
        out[key] = (m.group(1) if m else 'UNPARSED', reason.group(1) if reason else '')
    return out


def main():
    tags = sys.argv[1:] or ['old', 'new']
    summaries = {}
    for tag in tags:
        v = verdicts(tag)
        real = sum(1 for verdict, _ in v.values() if verdict == 'FALL')
        not_fall = sum(1 for verdict, _ in v.values() if verdict == 'NOT_A_FALL')
        other = len(v) - real - not_fall
        summaries[tag] = (len(v), real, not_fall, other, v)
        print(f'{tag}: {len(v)} alerts  ->  Gemini says FALL {real}, NOT_A_FALL {not_fall}'
              + (f', unparsed {other}' if other else ''))
        if len(v):
            print(f'   verified precision: {real / len(v):.1%}')

    if len(tags) == 2:
        a, b = tags
        na, ra = summaries[a][0], summaries[a][1]
        nb, rb = summaries[b][0], summaries[b][1]
        print()
        print(f'{a} -> {b}: alerts {na} -> {nb}, Gemini-confirmed real falls {ra} -> {rb}, '
              f'false alarms {na - ra} -> {nb - rb}')
        # The question a raw alert count cannot answer: did the extra alerts find real falls
        # or just add noise?
        if nb > na:
            extra_real = rb - ra
            extra_total = nb - na
            print(f'the {extra_total} extra alerts in {b} contain {extra_real} Gemini-confirmed '
                  f'real falls')

    print()
    print('false alarms in the newer set, for reading by eye:')
    last = tags[-1]
    for key, (verdict, reason) in sorted(summaries[last][4].items()):
        if verdict == 'NOT_A_FALL':
            print(f'   {key:28} {reason}')


if __name__ == '__main__':
    main()
