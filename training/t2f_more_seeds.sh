#!/bin/sh
# AGREED 2026-10-05 (Claude + Codex + Gemini): T2-full recipe stability, seeds 48/49/50, same recipe as t2_trunc.sh truncfull.
# Pre-registered (Codex): primary = unchanged dev-threshold rule; 0.65 descriptive. PASS = 3/3 seeds meet falls>=21.8,
# ADL>=16.1, val>=6.1, night FA<=13/3, owner FA<=5 (these 5 checks are not full-method success). Report every seed.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
R=$(pwd -W); S=training/data/stage1; L=$S/t2f_more.log; export PYTHONIOENCODING=utf-8
LST=$S/nightaug_s44_cache_lists.txt; DAY=$(grep ^DAY $LST | cut -d' ' -f2-); IR=$(grep ^IR $LST | cut -d' ' -f2-); OW=$(grep ^OWNER $LST | cut -d' ' -f2-)
die() { echo "$(date +%H:%M) $* -- stop" >> $L; exit 1; }
for s in 48 49 50; do
  N=t2_truncfull_s$s; DIR=$R/$S/$N; mkdir -p "$DIR"
  while tasklist //FI "IMAGENAME eq TOTClient-Win64-Shipping.exe" //NH | grep -qi TOTClient; do sleep 60; done
  if [ ! -f "$DIR/fall_classifier_v3.onnx" ]; then
    echo "$(date +%H:%M) start $N" >> $L
    ( export PYTHONUNBUFFERED=1 SPLIT=group EXCLUDE_NO_TIMEBASE=1 WINDOW_STEP=2 TEMPORAL_STRIDE=2 BALANCED_SAMPLING=1 \
        RESAMPLE_FPS=8 RUNTIME_MISSES=1 TRAIN_SEED=$s CKPT_PATH=$DIR/best.pt SYN_DIR= SYN_FRACTION=0 TRUNC_AUG=1 TRUNC_FULL=1 USE_LE2I=0
      env | grep -E "^(SPLIT|RESAMPLE_FPS|WINDOW_STEP|BALANCED_SAMPLING|RUNTIME_MISSES|TEMPORAL_STRIDE|EXCLUDE_NO_TIMEBASE|TRAIN_SEED|SYN_[A-Z]+|TRUNC_[A-Z]+|USE_LE2I)=" > "$DIR/recipe.env"
      cd training && python train.py ) > "$DIR/train.log" 2>&1 \
    && EXPORT_CKPT_PATH=$DIR/best.pt EXPORT_ONNX_OUT=$DIR/fall_classifier_v3.onnx python training/export_onnx.py > "$DIR/export.log" 2>&1 \
    && echo "$(date +%H:%M) done $N" >> $L || die "FAILED $N"
  fi
  for mode in rule 065; do
    grep -q "\"name\": \"T2_truncfull_s${s}_$mode\"" $S/results_pinned.jsonl && die "row T2_truncfull_s${s}_$mode exists"
    F=; [ $mode = 065 ] && F="--fixed 0.65"
    THRESHOLD_GRID=extended python training/measure/stage1_eval_pinned.py --name T2_truncfull_s${s}_$mode --model $DIR --pose nightaug_s44 \
      --day $DAY --ir $IR --owner $OW $F >> $L 2>&1 || die "EVAL FAILED $N $mode"
  done
  THR=$(grep "\"name\": \"T2_truncfull_s${s}_rule\"" $S/results_pinned.jsonl | tail -1 | python -c "import json,sys; print(json.loads(sys.stdin.read())['threshold'])")
  O=training/data/multi_diag_v2/track_metric_t2f_s$s.txt
  V3_THRESHOLD=$THR python training/measure/track_metric.py $DIR $OW > $O.tmp 2>&1 && grep -q '^{' $O.tmp && grep '^{' $O.tmp > $O \
    || { tail -3 $O.tmp >> $L; die "D3 FAILED $N"; }
  rm -f $O.tmp; echo "$(date +%H:%M) D3 $N at rule threshold $THR" >> $L
done
echo "$(date +%H:%M) T2F MORE DONE" >> $L
