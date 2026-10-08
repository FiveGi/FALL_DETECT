#!/bin/sh
# NIGHT-NOISE-v1 GPU accuracy screen (Codex design 2026-10-01). Arms that passed the CPU cost
# screen (training/data/multi_diag_v2/preprocess_cost.txt, +<=5 ms mean): D1 bilateral->auto,
# G auto with grey gate chroma<6. B (deployed auto) = the stock screening caches in
# training/data/pose_ir_eval/stock (same settings). Stock pose model. Priority tiers (screen_lib.sh).
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
. training/screen_lib.sh
R=$(pwd -W); OUT=$R/training/data/pose_ir_eval; L=$OUT/noise_queue.log
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 V3_DEVICE=cuda V3_IMGSZ=320 TARGET_FPS=8 \
       V3_ROI_IMGSZ=256 V3_ROI_FULL_EVERY=8 V3_POSE_CONF=0.3
unset V3_POSE_MODEL
ARMS="D1:bilateral,auto:0 G:auto:6"
IFS='|'; for tier in $TIERS; do IFS=' '
  for a in $ARMS; do IFS=':'; set -- $a; IFS=' '; tag=noise_$1
    export V3_PREPROCESS=$2 V3_PREPROCESS_GREY_BELOW=$3
    for run in $tier; do IFS=','; set -- $run; IFS=' '
      build $L $OUT/$tag $1 $2 $3 $4 || exit 1
    done
  done
  echo "$(date +%H:%M) tier done" >> $L; IFS='|'
done; IFS=' '
for t in noise_D1 noise_G; do python training/measure/check_cache_set.py "$OUT/$t" 8 >> $L 2>&1 \
  || { echo "$(date +%H:%M) NOISE FAILED (incomplete $t)" >> $L; exit 1; }; done
echo "$(date +%H:%M) NOISE CACHES DONE" >> $L
