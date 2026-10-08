#!/bin/sh
# FALLPOSE s42: the remaining crop phases for the frozen Stage-1 gates -- URFD day phases 1,2,3,5,6,7
# (2 at a time) then owner phases 1,2,3,5,6,7 -- so FALLPOSE (+E2) can be judged on all 8 phases.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/pose_ir_eval_cpu/queue.log; O=training/data/fallpose_owner
M=$(pwd -W)/training/data/pose_ir/fallpose_s42_p2/weights/best.pt
xargs -P 2 -I{} sh training/cpu_cache_job.sh "{}" < training/data/pose_ir_eval_cpu_jobs_fallpose2.txt
for p in 1 2 3 5 6 7; do
  V3_POSE_MODEL=$M V3_DEVICE=cpu CPU_THREADS=4 OMP_NUM_THREADS=4 ROI_PHASE=$p PYTHONIOENCODING=utf-8 \
    python training/measure/cache_owner_segments.py > $O/cache_p$p.log 2>&1 || { echo "$(date +%H:%M) FAILED fallpose_owner $p" >> $L; exit 1; }
done
echo "$(date +%H:%M) FALLPOSE FULL CACHES DONE" >> $L
