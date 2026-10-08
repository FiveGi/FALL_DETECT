#!/bin/sh
# PROPOSAL (2026-10-04 night; NOT launched until Codex + Gemini agree).
# T3 failed its method gate (>= 50% fewer object false persons per seed: 1 of 3 seeds). The single checkpoint
# posneg_s44 is still the best pose candidate on objects (60.4/1000 vs nightaug_s44 141.2, stock 72) with equal lying
# presence (0.473 vs 0.462). Checkpoint question, separate from the method gate: does posneg_s44 match nightaug_s44
# (the current N1 candidate) on the frozen Stage-1 gates and on D3? Same protocol as n_s44_chain.sh:
#   19 CPU caches (8 day phases, IR seeds 0/7/13, 8 owner phases; 3 at a time) -> pinned eval with the deployed
#   classifier at 0.65 (row P1_posneg44_deployed065, comparator N1_nightaug44_deployed065) -> track_metric.
# Runs after a NEW 'STEP1B DONE' line (quiet GPU/CPU, nothing else queued). Fails closed; pauses while the game runs.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
S=training/data/stage1; L=training/data/stage1/t3b.log; export PYTHONIOENCODING=utf-8
n=$(cat $S/step1b.log 2>/dev/null | wc -l)
until tail -n +$((n + 1)) $S/step1b.log 2>/dev/null | grep -q "STEP1B DONE"; do sleep 120; done
while tasklist //FI "IMAGENAME eq TOTClient-Win64-Shipping.exe" //NH | grep -qi TOTClient; do sleep 60; done
die() { echo "$(date +%H:%M) $* -- stop" >> $L; exit 1; }
M=$(pwd -W)/training/data/pose_ir/posneg_s44_p2/weights/best.pt; [ -f "$M" ] || die "no checkpoint $M"
J=training/data/gate_jobs_posneg44.txt; : > $J
for p in 0 1 2 3 4 5 6 7; do echo "U posneg_s44|$M|auto|0|$p|0|0|0" >> $J; done
for sd in 0 7 13; do echo "U posneg_s44|$M|auto|0|0|1|0|$sd" >> $J; done
for p in 0 1 2 3 4 5 6 7; do echo "O posneg_s44|$M|$p" >> $J; done
echo "$(date +%H:%M) caches start (19 jobs, PAR=3)" >> $L
PAR=3 sh training/gate_jobs_runner.sh $J
python training/measure/list_model_caches.py posneg_s44 > $S/posneg_s44_cache_lists.raw 2>> $L || die "cache lists incomplete"
tr -d '\r' < $S/posneg_s44_cache_lists.raw > $S/posneg_s44_cache_lists.txt
C=$S/posneg_s44_cache_lists.txt; DAY=$(grep ^DAY $C | cut -d' ' -f2-); IR=$(grep ^IR $C | cut -d' ' -f2-); OW=$(grep ^OWNER $C | cut -d' ' -f2-)
grep -q "\"name\": \"P1_posneg44_deployed065\"" $S/results_pinned.jsonl $S/*.log 2>/dev/null && die "row name already used"
python training/measure/stage1_eval_pinned.py --name P1_posneg44_deployed065 --model models --pose posneg_s44 \
  --day $DAY --ir $IR --owner $OW --fixed 0.65 >> $L 2>&1 || die "pinned eval FAILED"
O=training/data/multi_diag_v2/track_metric_posneg44.txt
V3_THRESHOLD=0.65 python training/measure/track_metric.py models $OW > $O.tmp 2>&1 && grep -q '^{' $O.tmp \
  && grep '^{' $O.tmp > $O || { tail -3 $O.tmp >> $L; die "D3 metric FAILED"; }
rm -f $O.tmp
echo "$(date +%H:%M) T3B DONE" >> $L
