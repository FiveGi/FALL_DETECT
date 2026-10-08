#!/bin/sh
# Stage 1 ablation (Codex design 2026-10-01): base recipe with the caller's overrides, one seed.
# Usage: BALANCED_SAMPLING=0 sh training/stage1_ablate.sh abl_D 42
# A separate file from stage1_run.sh on purpose: that one is read by a running queue.
set -e
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
NAME=$1_s$2; SEED=$2; R=$(pwd -W); DIR=$R/training/data/stage1/$NAME
mkdir -p "$DIR"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 SPLIT=${SPLIT:-group} RESAMPLE_FPS=${RESAMPLE_FPS:-8} \
       WINDOW_STEP=${WINDOW_STEP:-2} BALANCED_SAMPLING=${BALANCED_SAMPLING:-1} \
       RUNTIME_MISSES=${RUNTIME_MISSES:-1} TRAIN_SEED=$SEED SYN_DIR=${SYN_DIR:-} SYN_FRACTION=${SYN_FRACTION:-0} CKPT_PATH=$DIR/best.pt
env | grep -E "^(SPLIT|RESAMPLE_FPS|WINDOW_STEP|BALANCED_SAMPLING|RUNTIME_MISSES|GMDCSA24_|TEMPORAL_STRIDE|EXCLUDE_NO_TIMEBASE|SYN_|USE_LE2I)=" > "$DIR/recipe.env"
( cd training && python train.py ) > "$DIR/train.log" 2>&1
EXPORT_CKPT_PATH=$DIR/best.pt EXPORT_ONNX_OUT=$DIR/fall_classifier_v3.onnx python training/export_onnx.py > "$DIR/export.log" 2>&1
python training/measure/stage1_eval.py "$NAME" "$DIR" > "$DIR/eval.log" 2>&1
echo "$(date +%H:%M) $NAME $(tail -n 1 "$DIR/eval.log")"
