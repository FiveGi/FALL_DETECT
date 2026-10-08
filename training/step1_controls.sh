#!/bin/sh
# PLAN step 1 (agreed 2026-10-04): recipe-A controls retrained with the CURRENT code (RESAMPLE_FPS=0,
# TEMPORAL_STRIDE=2, no syn, no trunc), seeds 45/46/47; then pinned scorecard on nightaug_s44 caches at fixed
# 0.65 and by the half-A rule (extended grid). Waits for T2 training to finish (GPU). Detached, idempotent.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
R=$(pwd -W); L=training/data/stage1/step1.log
# Plan round 3 G5 (agreed): one GPU job at a time -> wait for T2 AND T3 training.
until grep -q "T2 TRAIN DONE" training/data/stage1/t2.log 2>/dev/null && test -f training/data/holiday_queue.log.t3marker; do sleep 60; done
for s in 45 46 47; do
  NAME=ctrlA_s$s; DIR=$R/training/data/stage1/$NAME; mkdir -p "$DIR"
  [ -f "$DIR/fall_classifier_v3.onnx" ] && continue
  echo "$(date +%H:%M) start $NAME" >> $L
  ( export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 SPLIT=group EXCLUDE_NO_TIMEBASE=1 WINDOW_STEP=2 TEMPORAL_STRIDE=2 \
      BALANCED_SAMPLING=1 RESAMPLE_FPS=0 RUNTIME_MISSES=1 TRAIN_SEED=$s CKPT_PATH=$DIR/best.pt \
      SYN_DIR= SYN_FRACTION=0 TRUNC_AUG=0 TRUNC_FULL=0 USE_LE2I=0   # Codex P2: controls explicit
    env | grep -E "^(SPLIT|RESAMPLE_FPS|WINDOW_STEP|BALANCED_SAMPLING|RUNTIME_MISSES|TEMPORAL_STRIDE|EXCLUDE_NO_TIMEBASE|SYN_[A-Z]+|TRUNC_[A-Z]+|USE_LE2I)=" > "$DIR/recipe.env"
    cd training && python train.py ) > "$DIR/train.log" 2>&1 \
  && EXPORT_CKPT_PATH=$DIR/best.pt EXPORT_ONNX_OUT=$DIR/fall_classifier_v3.onnx python training/export_onnx.py > "$DIR/export.log" 2>&1 \
  && echo "$(date +%H:%M) done $NAME" >> $L || { echo "$(date +%H:%M) FAILED $NAME -- stop" >> $L; exit 1; }
done
export THRESHOLD_GRID=extended
LST=training/data/stage1/nightaug_s44_cache_lists.txt; DAY=$(grep ^DAY $LST | cut -d' ' -f2-); IR=$(grep ^IR $LST | cut -d' ' -f2-); OW=$(grep ^OWNER $LST | cut -d' ' -f2-)
for s in 45 46 47; do
  python training/measure/stage1_eval_pinned.py --name ctrlA_s${s}_065 --model $R/training/data/stage1/ctrlA_s$s --pose nightaug_s44 --day $DAY --ir $IR --owner $OW --fixed 0.65 >> $L 2>&1     || { echo "$(date +%H:%M) EVAL FAILED ctrlA_s$s -- stop" >> $L; exit 1; }    # Codex P2: fail closed
  python training/measure/stage1_eval_pinned.py --name ctrlA_s${s}_rule --model $R/training/data/stage1/ctrlA_s$s --pose nightaug_s44 --day $DAY --ir $IR --owner $OW >> $L 2>&1     || { echo "$(date +%H:%M) EVAL FAILED ctrlA_s$s rule -- stop" >> $L; exit 1; }
done
echo "$(date +%H:%M) STEP1 DONE" >> $L
