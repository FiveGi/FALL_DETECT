#!/bin/sh
# Overnight E1 (agreed 8 Oct; manifest .ai_evidence/plan_oct7_8/overnight_manifest.md frozen before any replay).
# One job at a time on the host, during the R2 soak. Option A @0.70, cap 4, k = 2. Exploration only (not in 9 Oct build).
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
O=training/data/stage1/e1; mkdir -p $O; L=$O/run.log
LIST=training/data/stage1/nightaug_s44_cache_lists.txt
DAY=$(grep ^DAY $LIST | cut -d' ' -f2-); IR=$(grep ^IR $LIST | cut -d' ' -f2-); OW=$(grep ^OWNER $LIST | cut -d' ' -f2-)
M=training/data/stage1/t2_truncfull_s45
export PYTHONIOENCODING=utf-8 V3_THRESHOLD=0.70
step() { echo "$(date +%H:%M) start $1" >> $L; }
fail() { echo "$(date +%H:%M) FAILED $1" >> $L; exit 1; }
[ -e $O/p3_ledger_k2.json ] && fail "outputs exist -- refuse (run once)"
step ledger
python training/measure/p3_ledger.py 2 $M $OW > $O/p3_ledger_k2.json 2> $O/p3_ledger_k2.err || fail ledger
step "gates OFF"
PER_CLIP_OUT=$O/perclip_off.json python training/measure/stage1_eval_pinned.py --name P3off_A070 --model $M \
  --pose nightaug_s44 --day $DAY --ir $IR --owner $OW --fixed 0.70 > $O/gates_off.txt 2>&1 || fail "gates OFF"
step "gates ON"
V3_HELD_ALERT_MAX=2 PER_CLIP_OUT=$O/perclip_on.json python training/measure/stage1_eval_pinned.py --name P3on_A070 --model $M \
  --pose nightaug_s44 --day $DAY --ir $IR --owner $OW --fixed 0.70 > $O/gates_on.txt 2>&1 || fail "gates ON"
step "D3 OFF"
python training/measure/track_metric.py $M $OW > $O/d3_off.txt 2>&1 || fail "D3 OFF"
step "D3 ON"
V3_HELD_ALERT_MAX=2 python training/measure/track_metric.py $M $OW > $O/d3_on.txt 2>&1 || fail "D3 ON"
echo "$(date +%H:%M) E1 DONE" >> $L
