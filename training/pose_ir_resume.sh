#!/bin/sh
# Resume POSE-IR after the 2026-10-01 pause: the runs not yet finished (42 c0/c1 and 43 c0 are done).
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/pose_ir/queue.log
for sc in "43 1" "44 0" "44 1"; do
  set -- $sc; s=$1; c=$2
  echo "$(date +%H:%M) start seed $s control $c" >> $L
  SEED=$s CONTROL=$c WORKERS=${WORKERS:-2} PYTHONIOENCODING=utf-8 python training/finetune_pose_ir.py > training/data/pose_ir/run_s${s}_c${c}.log 2>&1 \
    && echo "$(date +%H:%M) done seed $s control $c" >> $L || echo "$(date +%H:%M) FAILED seed $s control $c" >> $L
done
echo "$(date +%H:%M) POSE-IR QUEUE DONE" >> $L
