#!/bin/sh
# FALLPOSE s42 screening caches, 2 at a time beside the main CPU queue (3): RAM ~1.5 GB each.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
xargs -P 2 -I{} sh training/cpu_cache_job.sh "{}" < training/data/pose_ir_eval_cpu_jobs_fallpose.txt
echo "$(date +%H:%M) FALLPOSE CPU CACHES DONE" >> training/data/pose_ir_eval_cpu/queue.log
