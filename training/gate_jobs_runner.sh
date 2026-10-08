#!/bin/sh
# Runs a gate job list ("U <cpu_cache_job line>" or "O <cpu_owner_job line>"), 2 at a time.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
job() { case "$1" in U\ *) sh training/cpu_cache_job.sh "${1#U }";; O\ *) sh training/cpu_owner_job.sh "${1#O }";; esac; }
export -f job 2>/dev/null
xargs -P ${PAR:-2} -I{} sh -c 'case "$1" in U\ *) sh training/cpu_cache_job.sh "${1#U }";; O\ *) sh training/cpu_owner_job.sh "${1#O }";; esac' _ "{}" < $1
# Each job logs "done" or "FAILED" itself; report failures here so a caller grepping DONE sees them.
n=$(grep -c . $1); echo "$(date +%H:%M) GATE JOBS DONE $1 ($n jobs; check FAILED lines above -- list_model_caches.py asserts completeness)" >> training/data/pose_ir_eval_cpu/queue.log
