#!/bin/sh
# PROPOSAL v2 (2026-10-04, after Codex review; NOT launched until Codex + Gemini agree).
# Why: the 4 stale runners (killed 22:33) started a SECOND copy of step 1 at 21:36, so ctrlA_s45 and ctrlA_s46 were
# trained twice at once into the same folders. ctrlA_s47 (22:20) ran alone. One GPU job at a time, never during the
# CPU timing run:
#  1. wait for NEW 'STEP1 DONE' and 'STEP2 timing done' lines (written after this script started)
#  2. quarantine ctrlA_s45/s46 (unique *_void_<time> name; stop if that fails), retrain as ctrlA_s45c/s46c, evaluate
#     under NEW row names (the evaluator appends rows; the void rows keep their old names and are ignored)
#  3. train truncfull s46 + s47 (agreed T2 plan), pinned evals at 0.65 and by the rule
#  4. track_metric (D3) on ctrlA_s45c/s46c/s47 and truncfull s45..47; any failure stops the script (no DONE line)
# Mock mode for review: STAGE=<tmp dir> QUEUE=<tmp file> DRY=1 sh training/step1b_redo_t2full.sh
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
R=$(pwd -W); S=${STAGE:-training/data/stage1}; Q=${QUEUE:-training/data/holiday_queue.log}; L=$S/step1b.log
export PYTHONIOENCODING=utf-8
n1=$(cat $S/step1.log 2>/dev/null | wc -l); n2=$(cat $Q 2>/dev/null | wc -l)   # Codex P2: run-scoped markers
until tail -n +$((n1 + 1)) $S/step1.log 2>/dev/null | grep -q "STEP1 DONE" \
   && tail -n +$((n2 + 1)) $Q 2>/dev/null | grep -q "STEP2 timing done"; do sleep ${POLL:-120}; done
[ -n "$DRY" ] || [ "$(grep -c sustained training/data/multi_diag_v2/pipeline_timing.txt 2>/dev/null)" -ge 3 ] \
  || echo "$(date +%H:%M) WARNING step 2 timing produced no result lines (CPU is quiet either way; continuing)" >> $L
LST=training/data/stage1/nightaug_s44_cache_lists.txt
DAY=$(grep ^DAY $LST | cut -d' ' -f2-); IR=$(grep ^IR $LST | cut -d' ' -f2-); OW=$(grep ^OWNER $LST | cut -d' ' -f2-)
die() { echo "$(date +%H:%M) $* -- stop" >> $L; exit 1; }
train() {  # $1 name  $2 seed  $3 RESAMPLE_FPS  $4 TRUNC_AUG  $5 TRUNC_FULL
  DIR=$R/$S/$1; [ -f "$DIR/fall_classifier_v3.onnx" ] && return 0; mkdir -p "$DIR"
  while tasklist //FI "IMAGENAME eq TOTClient-Win64-Shipping.exe" //NH | grep -qi TOTClient; do sleep 60; done
  echo "$(date +%H:%M) start $1" >> $L
  if [ -n "$DRY" ]; then echo dry > "$DIR/fall_classifier_v3.onnx"; echo "$(date +%H:%M) done $1 (dry)" >> $L; return 0; fi
  ( export PYTHONUNBUFFERED=1 SPLIT=group EXCLUDE_NO_TIMEBASE=1 WINDOW_STEP=2 TEMPORAL_STRIDE=2 BALANCED_SAMPLING=1 \
      RESAMPLE_FPS=$3 RUNTIME_MISSES=1 TRAIN_SEED=$2 CKPT_PATH=$DIR/best.pt SYN_DIR= SYN_FRACTION=0 TRUNC_AUG=$4 TRUNC_FULL=$5 USE_LE2I=0
    env | grep -E "^(SPLIT|RESAMPLE_FPS|WINDOW_STEP|BALANCED_SAMPLING|RUNTIME_MISSES|TEMPORAL_STRIDE|EXCLUDE_NO_TIMEBASE|TRAIN_SEED|SYN_[A-Z]+|TRUNC_[A-Z]+|USE_LE2I)=" > "$DIR/recipe.env"
    cd training && python train.py ) > "$DIR/train.log" 2>&1 \
  && EXPORT_CKPT_PATH=$DIR/best.pt EXPORT_ONNX_OUT=$DIR/fall_classifier_v3.onnx python training/export_onnx.py > "$DIR/export.log" 2>&1 \
  && echo "$(date +%H:%M) done $1" >> $L || die "FAILED $1"
}
evalp() {  # $1 row name prefix (new, unused)  $2 model dir name
  if [ -n "$DRY" ]; then echo "$(date +%H:%M) eval $1 (dry)" >> $L; return 0; fi
  grep -q "\"name\": \"$1_065\"" $S/*.log 2>/dev/null && die "row name $1 already used"
  THRESHOLD_GRID=extended python training/measure/stage1_eval_pinned.py --name $1_065 --model $R/$S/$2 --pose nightaug_s44 --day $DAY --ir $IR --owner $OW --fixed 0.65 >> $L 2>&1 \
  && THRESHOLD_GRID=extended python training/measure/stage1_eval_pinned.py --name $1_rule --model $R/$S/$2 --pose nightaug_s44 --day $DAY --ir $IR --owner $OW >> $L 2>&1 \
  || die "EVAL FAILED $2"
}
for s in 45 46; do                                          # Codex P1: quarantine must succeed
  if [ -d $S/ctrlA_s$s ]; then
    V=$S/ctrlA_s${s}_void_$(date +%H%M%S); mv $S/ctrlA_s$s $V && [ ! -d $S/ctrlA_s$s ] && [ -d $V ] || die "quarantine ctrlA_s$s failed"
    echo "$(date +%H:%M) ctrlA_s$s -> $V (trained twice concurrently; void)" >> $L
  fi
  train ctrlA_s${s}c $s 0 0 0; evalp ctrlA_s${s}c ctrlA_s${s}c
done
for s in 46 47; do train t2_truncfull_s$s $s 8 1 1; evalp T2_truncfull_s$s t2_truncfull_s$s; done
O=training/data/multi_diag_v2/track_metric_step1b.txt; [ -n "$DRY" ] && O=$S/track_metric_dry.txt; : > $O
for d in ctrlA_s45c ctrlA_s46c ctrlA_s47 t2_truncfull_s45 t2_truncfull_s46 t2_truncfull_s47; do
  [ -n "$DRY" ] && { echo "{\"dry\": \"$d\"}" >> $O; continue; }
  V3_THRESHOLD=0.65 python training/measure/track_metric.py $S/$d $OW > $O.tmp 2>&1 && grep -q '^{' $O.tmp \
    && grep '^{' $O.tmp >> $O || { tail -3 $O.tmp >> $L; die "D3 metric FAILED $d"; }
done; rm -f $O.tmp
echo "$(date +%H:%M) STEP1B DONE" >> $L
