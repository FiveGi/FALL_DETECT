#!/bin/sh
# POSE-IR screening (plan in AI_HANDOFF 2026-10-01 "independent night degradation"): each pose
# checkpoint vs the stock pose model, deployed classifier unchanged at its threshold 0.65.
# GPU, every model the same way, own CACHE_DIR (the URFD cache key has no device field). Stock and
# winner re-checked on CPU. Codex review: explicit variables, stop on failure, completeness check.
# 2026-10-02 00:30: reordered into priority tiers (screen_lib.sh) -- the night builds were ~30 min
# each with the GPU shared, and the full set would not finish before the morning report.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
. training/screen_lib.sh
R=$(pwd -W); OUT=$R/training/data/pose_ir_eval; L=$OUT/queue.log; P=$R/training/data/pose_ir
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 V3_DEVICE=cuda V3_IMGSZ=320 TARGET_FPS=8 \
       V3_ROI_IMGSZ=256 V3_ROI_FULL_EVERY=8 V3_POSE_CONF=0.3 V3_PREPROCESS=auto V3_PREPROCESS_GREY_BELOW=0
clean_partial "$OUT"
MODELS="stock:- nightaug_s42:$P/nightaug_s42_p2/weights/best.pt colour_s42:$P/colour_s42_p2/weights/best.pt nightaug_s43:$P/nightaug_s43_p2/weights/best.pt"
IFS='|'; for tier in $TIERS; do IFS=' '
  for m in $MODELS; do tag=${m%%:*}; pm=${m#*:}
    if [ "$pm" = "-" ]; then unset V3_POSE_MODEL; else export V3_POSE_MODEL=$pm; fi
    for run in $tier; do IFS=','; set -- $run; IFS=' '
      build $L $OUT/$tag $1 $2 $3 $4 || exit 1
    done
  done
  echo "$(date +%H:%M) tier done" >> $L; IFS='|'
done; IFS=' '
for m in $MODELS; do python training/measure/check_cache_set.py "$OUT/${m%%:*}" 8 >> $L 2>&1 \
  || { echo "$(date +%H:%M) NIGHT EVAL FAILED (incomplete ${m%%:*})" >> $L; exit 1; }; done
echo "$(date +%H:%M) NIGHT EVAL CACHES DONE" >> $L
