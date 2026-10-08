#!/bin/sh
# Holiday queue (owner: datasets + training). Launched DETACHED via Start-Process so it survives Claude
# session ends (2026-10-04: bash-launched waiters died overnight). Steps are idempotent; rerun-safe.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/holiday_queue.log
echo "$(date '+%d %H:%M') queue start" >> $L
grep -q "T1 EVAL DONE" training/data/stage1/t1_eval.log 2>/dev/null || { : > training/data/stage1/t1_eval.log; sh training/t1_eval.sh; }
echo "$(date '+%d %H:%M') T1 eval done" >> $L
(cd D:/project/PROJECT/datasets/posneg && python download.py >> download.log 2>&1)
echo "$(date '+%d %H:%M') background download done: $(ls D:/project/PROJECT/datasets/posneg/images | wc -l)" >> $L
