#!/bin/sh
# T2 TRUNC-v1 (frozen design + P4): recipe as T1 control (RESAMPLE_FPS=8, no syn) + TRUNC_AUG=1, seeds
# 45/46/47; controls = t1_ctrl_s45..47. Plus TRUNC-full (Gemini's variant) on seed 45. Detached.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
R=$(pwd -W); L=training/data/stage1/t2.log
for job in "trunc 45 0" "trunc 46 0" "trunc 47 0" "truncfull 45 1"; do set -- $job
  NAME=t2_$1_s$2; DIR=$R/training/data/stage1/$NAME; mkdir -p "$DIR"
  [ -f "$DIR/fall_classifier_v3.onnx" ] && continue
  echo "$(date +%H:%M) start $NAME" >> $L
  ( export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 SPLIT=group EXCLUDE_NO_TIMEBASE=1 WINDOW_STEP=2 TEMPORAL_STRIDE=2 \
      BALANCED_SAMPLING=1 RESAMPLE_FPS=8 RUNTIME_MISSES=1 TRAIN_SEED=$2 CKPT_PATH=$DIR/best.pt TRUNC_AUG=1 TRUNC_FULL=$3
    env | grep -E "^(SPLIT|RESAMPLE_FPS|WINDOW_STEP|BALANCED_SAMPLING|RUNTIME_MISSES|TEMPORAL_STRIDE|EXCLUDE_NO_TIMEBASE|TRAIN_SEED|TRUNC_[A-Z]+)=" > "$DIR/recipe.env"
    cd training && python train.py ) > "$DIR/train.log" 2>&1 \
  && EXPORT_CKPT_PATH=$DIR/best.pt EXPORT_ONNX_OUT=$DIR/fall_classifier_v3.onnx python training/export_onnx.py > "$DIR/export.log" 2>&1 \
  && echo "$(date +%H:%M) done $NAME" >> $L || echo "$(date +%H:%M) FAILED $NAME" >> $L
done
echo "$(date +%H:%M) T2 TRAIN DONE" >> $L
