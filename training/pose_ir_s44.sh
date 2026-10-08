#!/bin/sh
# Third POSE-IR seed (Codex: the adopted design is 3 seeds + colour controls). Night-aug s44 first,
# then the colour controls s43 and s44 if time allows. WORKERS=2 (8 GB floor).
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
Q=training/data/pose_ir/queue.log
for sc in "44 0" "43 1" "44 1"; do set -- $sc
  echo "$(date +%H:%M) start seed $1 control $2" >> $Q
  SEED=$1 CONTROL=$2 WORKERS=2 PYTHONIOENCODING=utf-8 python training/finetune_pose_ir.py > training/data/pose_ir/run_s$1_c$2.log 2>&1 \
    && echo "$(date +%H:%M) done seed $1 control $2" >> $Q || echo "$(date +%H:%M) FAILED seed $1 control $2" >> $Q
done
echo "$(date +%H:%M) POSE-IR QUEUE DONE" >> $Q
