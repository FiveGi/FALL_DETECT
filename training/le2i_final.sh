#!/bin/sh
# Le2i half 1, ONCE (pre-registered in AI_HANDOFF.md). 2 configs x 8 ROI phases, 4 runs at a time, fail closed.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/le2i_final/run.log; mkdir -p training/data/le2i_final
# Codex P1: ONE run -- refuse to start if any earlier output exists (no stale JSON can count toward 16/16).
ls training/data/le2i_final/*_p[0-7].json >/dev/null 2>&1 && { echo "$(date +%H:%M) old outputs present -- refuse" >> $L; exit 1; }
A_POSE=training/data/pose_ir/nightaug_s44_p2/weights/best.pt; A_CLS=training/data/stage1/t2_truncfull_s45
J=training/data/le2i_final/jobs.txt; : > $J
for p in 0 1 2 3 4 5 6 7; do echo "A_finalist $A_POSE $A_CLS 0.70 $p" >> $J; echo "REF_production - models 0.65 $p" >> $J; done
xargs -P 4 -L 1 sh -c 'PYTHONIOENCODING=utf-8 python training/measure/le2i_eval.py run "$0" "$1" "$2" "$3" "$4" > training/data/le2i_final/$0_p$4.log 2>&1 && echo "$(date +%H:%M) done $0 p$4" >> training/data/le2i_final/run.log || echo "$(date +%H:%M) FAILED $0 p$4" >> training/data/le2i_final/run.log' < $J
n=$(ls training/data/le2i_final/*_p[0-7].json 2>/dev/null | wc -l)
if [ "$n" -eq 16 ] && ! grep -q FAILED $L; then echo "$(date +%H:%M) LE2I RUNS DONE" >> $L
else echo "$(date +%H:%M) LE2I RUNS INCOMPLETE ($n/16 or a FAILED job) -- stop" >> $L; exit 1; fi
