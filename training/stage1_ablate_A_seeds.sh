#!/bin/sh
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/stage1/ablate.log
for s in 43 44; do
  RESAMPLE_FPS=0 TEMPORAL_STRIDE=2 EXCLUDE_NO_TIMEBASE=1 sh training/stage1_ablate.sh abl_A $s >> $L 2>&1 || echo "FAILED abl_A s$s" >> $L
done
echo "$(date +%H:%M) A SEEDS DONE" >> $L
