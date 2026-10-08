#!/usr/bin/env bash
# plan_v5 T2 latency, protocol PRE-REGISTERED in AI_HANDOFF.md (8 Oct ~00:37) before any measurement:
# one camera at a time on the dev stack (4 pinned cores, option A), clips 17 and 14, caps in the order 4,8,4,8,4,8.
# Each run: recreate celery_worker with V3_NUM_POSES=<cap> (scratch compose override, worker only), start the clip camera,
# 4 min; per-minute 'Detection rate' values of minutes 2-4 are kept (first 60 s = warm-up). Run statistic = median of
# those; cap statistic = median over its 3 runs. PASS iff cap8 >= 0.97 x cap4 on both clips AND clip 17 at cap 8 raises
# 0 fall alerts. Output: training/data/system_test/cap_latency.txt (+ raw logs).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PATH:/c/Program Files/Docker/Docker/resources/bin"
OUT=training/data/system_test/cap_latency; mkdir -p $OUT; R=training/data/system_test/cap_latency.txt; : > $R
OV=$OUT/override.yml
W=backend-elderly-surveillance-main-celery_worker-1
compose() { docker compose -f docker-compose.yml -f docker-compose.cpu4.yml -f $OV "$@"; }
for clip in 17 14; do
  for cap in 4 8 4 8 4 8; do
    printf 'services:\n  celery_worker:\n    environment:\n      - V3_NUM_POSES=%s\n' $cap > $OV
    python tools/system_test.py remove >/dev/null
    tw=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    compose up -d --force-recreate celery_worker >/dev/null 2>&1; sleep 20
    t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    python tools/system_test.py clips $clip >/dev/null
    sleep 255   # rate lines appear at t=0,60,120,180,240 s; keep 120,180,240 (minutes 2-4)
    log=$OUT/clip${clip}_cap${cap}_$(date +%H%M%S).log
    docker logs --since "$t0" $W > $log 2>&1
    # The identity line can be printed when the worker preloads the detector, i.e. before t0: read it since the recreate.
    docker logs --since "$tw" $W 2>&1 | grep "model identity" | tail -1 | grep -q "'num_poses': $cap"       || { echo "ABORT: identity does not show num_poses $cap" | tee -a $R; exit 1; }
    rates=$(grep -oE "Detection rate: [0-9.]+" $log | awk '{print $3}' | tail -n +3 | head -3 | tr '\n' ' ')
    falls=$(grep -c "FALL ALERT" $log || true)
    echo "clip $clip cap $cap rates(min2-4) $rates falls $falls" | tee -a $R
  done
done
python tools/system_test.py remove >/dev/null
printf 'services:\n  celery_worker:\n    environment: []\n' > $OV
compose up -d --force-recreate celery_worker >/dev/null 2>&1   # back to the default cap (4)
python - "$R" <<'EOF' | tee -a "$R"
import sys, statistics as st, re
rows = [l.split() for l in open(sys.argv[1]) if l.startswith('clip ')]
res, ok = {}, True
for r in rows:
    clip, cap = r[1], r[3]
    rates = [float(x) for x in r[r.index('rates(min2-4)') + 1:r.index('falls')]]
    res.setdefault((clip, cap), {'meds': [], 'falls': 0})
    res[(clip, cap)]['meds'].append(st.median(rates) if rates else 0.0)
    res[(clip, cap)]['falls'] += int(r[-1])
for clip in ('17', '14'):
    a, b = st.median(res[(clip, '4')]['meds']), st.median(res[(clip, '8')]['meds'])
    p = b >= 0.97 * a
    ok &= p
    print('clip %s: cap4 %.2f fps, cap8 %.2f fps (%.1f%%) -> %s' % (clip, a, b, 100 * (b / a - 1), 'PASS' if p else 'FAIL'))
f17 = res[('17', '8')]['falls']
ok &= f17 == 0
print('clip 17 cap 8 fall alerts over 3 runs: %d -> %s' % (f17, 'PASS' if f17 == 0 else 'FAIL'))
print('LATENCY+FA GATE:', 'PASS' if ok else 'FAIL')
EOF
