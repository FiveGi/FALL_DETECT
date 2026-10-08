#!/bin/sh
# T1 CLS-ADAPT (Codex freeze 2026-10-02, codex_clsadapt_final.md): recipe A with RESAMPLE_FPS=8 for BOTH
# arms (OF-Syn is 8 fps), BALANCED_SAMPLING=1, seeds 45/46/47; arm SYN = audited OF-Syn (582 clips accepted
# by Claude AND Gemini, poses from nightaug_s44) at 10% of windows; arm CTRL = same, SYN_FRACTION=0.
# Training only here; evaluation is pinned on nightaug_s44 caches by t1_eval.sh (stage1_ablate's own
# unpinned eval is skipped).
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
R=$(pwd -W); L=training/data/stage1/t1.log
for s in 45 46 47; do for arm in syn ctrl; do
  NAME=t1_${arm}_s$s; DIR=$R/training/data/stage1/$NAME; mkdir -p "$DIR"
  [ -f "$DIR/fall_classifier_v3.onnx" ] && continue
  F=0.10; [ $arm = ctrl ] && F=0
  echo "$(date +%H:%M) start $NAME" >> $L
  ( export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 SPLIT=group EXCLUDE_NO_TIMEBASE=1 WINDOW_STEP=2 TEMPORAL_STRIDE=2 \
      BALANCED_SAMPLING=1 RESAMPLE_FPS=8 RUNTIME_MISSES=1 TRAIN_SEED=$s CKPT_PATH=$DIR/best.pt \
      SYN_DIR=$R/training/data/poses_ofsyn_s44_audited SYN_FRACTION=$F
    env | grep -E "^(SPLIT|RESAMPLE_FPS|WINDOW_STEP|BALANCED_SAMPLING|RUNTIME_MISSES|TEMPORAL_STRIDE|EXCLUDE_NO_TIMEBASE|SYN_)=" > "$DIR/recipe.env"
    cd training && python train.py ) > "$DIR/train.log" 2>&1 \
  && EXPORT_CKPT_PATH=$DIR/best.pt EXPORT_ONNX_OUT=$DIR/fall_classifier_v3.onnx python training/export_onnx.py > "$DIR/export.log" 2>&1 \
  && echo "$(date +%H:%M) done $NAME" >> $L || echo "$(date +%H:%M) FAILED $NAME" >> $L
done; done
echo "$(date +%H:%M) T1 TRAIN DONE" >> $L
