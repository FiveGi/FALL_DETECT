#!/bin/sh
# POSE-IR (Codex design 2026-10-01): night-augmented vs colour-only control, paired seeds.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/pose_ir/queue.log; mkdir -p training/data/pose_ir
for s in 42 43 44; do
  for c in 0 1; do
    echo "$(date +%H:%M) start seed $s control $c" >> $L
    SEED=$s CONTROL=$c PYTHONIOENCODING=utf-8 python training/finetune_pose_ir.py > training/data/pose_ir/run_s${s}_c${c}.log 2>&1 \
      && echo "$(date +%H:%M) done seed $s control $c" >> $L || echo "$(date +%H:%M) FAILED seed $s control $c" >> $L
  done
done
echo "$(date +%H:%M) POSE-IR QUEUE DONE" >> $L
