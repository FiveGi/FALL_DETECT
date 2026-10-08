#!/bin/sh
# FALLPOSE s42 on the owner's 126 segments (multi-person included): pose cache at the CPU profile,
# phases 0 and 4, then the deployed classifier replayed at 0.65 and scored against stock's same phases.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/pose_ir_eval_cpu/queue.log; O=training/data/fallpose_owner; mkdir -p $O
M=$(pwd -W)/training/data/pose_ir/fallpose_s42_p2/weights/best.pt
for p in 0 4; do
  echo "$(date +%H:%M) start fallpose_owner phase=$p" >> $L
  V3_POSE_MODEL=$M V3_DEVICE=cpu CPU_THREADS=4 OMP_NUM_THREADS=4 ROI_PHASE=$p PYTHONIOENCODING=utf-8 \
    python training/measure/cache_owner_segments.py > $O/cache_p$p.log 2>&1 || { echo "$(date +%H:%M) FAILED fallpose_owner $p" >> $L; exit 1; }
  C=$(grep -a "^done" $O/cache_p$p.log | awk '{print $2}')
  V3_THRESHOLD=0.65 python training/measure/replay_owner_segments.py "$C" models $O/fallpose_phase$p.json > /dev/null 2>&1
  cp test_result/incidents/alerts_cpu_320px_8fps_roi256_phase$p.json $O/stock_phase$p.json
  echo "$(date +%H:%M) done fallpose_owner phase=$p" >> $L
done
PYTHONIOENCODING=utf-8 python training/measure/score_incidents.py --paired "$O/stock_phase*.json" "$O/fallpose_phase*.json" > $O/score.txt 2>&1
echo "$(date +%H:%M) FALLPOSE OWNER DONE" >> $L
