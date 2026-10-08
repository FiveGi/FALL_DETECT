"""Authoritative scorer for tools/cap_latency.sh (pre-registered T2 latency protocol, AI_HANDOFF.md 8 Oct ~00:37).

Codex (8 Oct): the summary inside cap_latency.sh takes a median of whatever rate values a run produced, so a run that
logged fewer than three minutes (or none -> 0.0) would still be scored. Here every run must have exactly three per-minute
values (minutes 2-4) and every clip x cap exactly three runs; anything else makes the result INVALID, not PASS or FAIL.
Usage: python tools/cap_latency_score.py [training/data/system_test/cap_latency.txt]
"""
import statistics as st
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'training/data/system_test/cap_latency.txt'
runs, problems = {}, []
for line in open(path):
    if not line.startswith('clip ') or 'rates(min2-4)' not in line:   # skip the script's own summary lines
        continue
    r = line.split()
    clip, cap = r[1], r[3]
    rates = r[r.index('rates(min2-4)') + 1:r.index('falls')]
    if len(rates) != 3:
        problems.append('clip %s cap %s: %d rate values, need 3' % (clip, cap, len(rates)))
        continue
    runs.setdefault((clip, cap), []).append((st.median(float(x) for x in rates), int(r[-1])))
for key in [(c, k) for c in ('17', '14') for k in ('4', '8')]:
    if len(runs.get(key, [])) != 3:
        problems.append('clip %s cap %s: %d complete runs, need 3' % (key[0], key[1], len(runs.get(key, []))))
if problems:
    print('INVALID:', '; '.join(problems))
    sys.exit(2)
ok = True
for clip in ('17', '14'):
    a = st.median(m for m, _ in runs[(clip, '4')])
    b = st.median(m for m, _ in runs[(clip, '8')])
    p = b >= 0.97 * a
    ok &= p
    print('clip %s: cap4 %.2f fps, cap8 %.2f fps (%+.1f%%) -> %s  runs cap4 %s cap8 %s'
          % (clip, a, b, 100 * (b / a - 1), 'PASS' if p else 'FAIL',
             [m for m, _ in runs[(clip, '4')]], [m for m, _ in runs[(clip, '8')]]))
f17 = sum(f for _, f in runs[('17', '8')])
ok &= f17 == 0
print('clip 17 cap 8 fall alerts over 3 runs: %d -> %s' % (f17, 'PASS' if f17 == 0 else 'FAIL'))
print('LATENCY+FA GATE:', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
