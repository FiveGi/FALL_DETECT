#!/bin/sh
# Runs training/data/pose_ir_eval_cpu_jobs.txt three at a time (RAM: ~1.5 GB each, 8 GB floor).
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/pose_ir_eval_cpu/queue.log
xargs -P 3 -I{} sh training/cpu_cache_job.sh "{}" < training/data/pose_ir_eval_cpu_jobs.txt
echo "$(date +%H:%M) CPU CACHE QUEUE DONE" >> $L
