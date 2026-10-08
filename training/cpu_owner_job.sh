#!/bin/sh
# One owner-segment pose cache on the CPU (deployed profile, 4 threads). Arg: "tag|pose_model|phase".
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
IFS='|'; set -- $1; IFS=' '
L=training/data/pose_ir_eval_cpu/queue.log; O=training/data/owner_caches_log; mkdir -p $O
echo "$(date +%H:%M) start owner $1 phase=$3" >> $L
V3_POSE_MODEL=$2 V3_DEVICE=cpu CPU_THREADS=4 OMP_NUM_THREADS=4 ROI_PHASE=$3 PYTHONIOENCODING=utf-8 \
  python training/measure/cache_owner_segments.py > $O/$1_p$3.log 2>&1 \
  && echo "$(date +%H:%M) done owner $1 phase=$3" >> $L || echo "$(date +%H:%M) FAILED owner $1 phase=$3" >> $L
