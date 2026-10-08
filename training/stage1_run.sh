#!/bin/sh
# Stage 1, one arm x one seed: train -> export ONNX -> every frozen gate (stage1_eval.py).
# Usage: sh training/stage1_run.sh <arm: base|syn10|syn25> <seed>
# Recipe (AI_HANDOFF.md 2026-10-01): SPLIT=group RESAMPLE_FPS=8 WINDOW_STEP=2
# BALANCED_SAMPLING=1 RUNTIME_MISSES=1, everything else at the deployed recipe's defaults.
set -e
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
ARM=$1; SEED=$2; NAME=${ARM}_s${SEED}; R=$(pwd -W); DIR=$R/training/data/stage1/$NAME
mkdir -p "$DIR"
case $ARM in
  base)  SYN="" ;  FRAC=0 ;;
  syn10) SYN=$R/training/data/poses_omnifall_syn ; FRAC=0.10 ;;
  syn25) SYN=$R/training/data/poses_omnifall_syn ; FRAC=0.25 ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 SPLIT=group RESAMPLE_FPS=8 WINDOW_STEP=2 BALANCED_SAMPLING=1 \
       RUNTIME_MISSES=1 TRAIN_SEED=$SEED SYN_DIR=$SYN SYN_FRACTION=$FRAC CKPT_PATH=$DIR/best.pt
( cd training && python train.py ) > "$DIR/train.log" 2>&1
EXPORT_CKPT_PATH=$DIR/best.pt EXPORT_ONNX_OUT=$DIR/fall_classifier_v3.onnx \
  python training/export_onnx.py > "$DIR/export.log" 2>&1
python training/measure/stage1_eval.py "$NAME" "$DIR" > "$DIR/eval.log" 2>&1
tail -n 1 "$DIR/eval.log"
