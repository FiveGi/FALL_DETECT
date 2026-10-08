#!/bin/sh
# PLAN step 2: full-pipeline CPU timing. Runs when GPU training and step 1 are done (quiet machine).
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
until grep -q "STEP1 DONE" training/data/stage1/step1.log 2>/dev/null && test -f training/data/holiday_queue.log.t3marker; do sleep 120; done
S=training/data/stage1; P=training/data/pose_ir/nightaug_s44_p2/weights/best.pt
PYTHONIOENCODING=utf-8 python training/measure/pipeline_timing.py deployed=- s44=$P "s44_ens3=$P:$S/ctrlA_s46,$S/ctrlA_s47" \
  > training/data/multi_diag_v2/pipeline_timing.txt 2>&1   && [ "$(grep -c sustained training/data/multi_diag_v2/pipeline_timing.txt)" -eq 3 ]   && echo "$(date '+%d %H:%M') STEP2 timing done" >> training/data/holiday_queue.log   || echo "$(date '+%d %H:%M') STEP2 timing FAILED" >> training/data/holiday_queue.log
