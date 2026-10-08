#!/bin/sh
# Overnight GPU training queue, reprioritised 2026-10-02 02:15 and again 04:45 (FALLPOSE s42 done;
# s43 replicate first) for the 10:00 report: FALLPOSE s42
# first (the lying-person fix), then POSE-IR s43 colour control (restarted from scratch), s44 pair,
# FALLPOSE s43. Caches build on the CPU meanwhile (cpu_cache_queue.sh), so the GPU is training-only.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/fallpose_queue.log; Q=training/data/pose_ir/queue.log
export PYTHONIOENCODING=utf-8
run_fp() {
  echo "$(date +%H:%M) fallpose s$1 start" >> $L
  [ -f D:/project/PROJECT/datasets/fallpose/fallpose_mix_s$1.yaml ] || SEED=$1 python training/fallpose_data.py >> $L 2>&1
  CONTROL=1 SEED=$1 WORKERS=2 RUN_NAME=fallpose_s$1 COCO_POSE_YAML=D:/project/PROJECT/datasets/fallpose/fallpose_mix_s$1.yaml \
    python training/finetune_pose_ir.py > training/data/fallpose_s$1.log 2>&1 \
    && echo "$(date +%H:%M) fallpose s$1 done" >> $L || echo "$(date +%H:%M) fallpose s$1 FAILED" >> $L
}
run_ir() {
  echo "$(date +%H:%M) start seed $1 control $2" >> $Q
  SEED=$1 CONTROL=$2 WORKERS=2 python training/finetune_pose_ir.py > training/data/pose_ir/run_s$1_c$2.log 2>&1 \
    && echo "$(date +%H:%M) done seed $1 control $2" >> $Q || echo "$(date +%H:%M) FAILED seed $1 control $2" >> $Q
}
: # (s43 c1 note logged at 02:12)
run_fp 43
run_ir 43 1
run_ir 44 0
run_ir 44 1
echo "$(date +%H:%M) POSE-IR QUEUE DONE" >> $Q
echo "$(date +%H:%M) FALLPOSE QUEUE DONE" >> $L
