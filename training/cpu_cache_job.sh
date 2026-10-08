#!/bin/sh
# One CPU cache build (deployed profile, 4 threads). Line: tag|pose_model|preprocess|grey|phase|ir|alt|seed
# pose_model "-" = stock. Output root training/data/pose_ir_eval_cpu/<tag> (own root: cache keys
# carry no device field). Logs to training/data/pose_ir_eval_cpu/queue.log.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
IFS='|'; set -- $1; IFS=' '
R=$(pwd -W); OUT=$R/training/data/pose_ir_eval_cpu; L=$OUT/queue.log; mkdir -p $OUT
[ "$2" = "-" ] && unset V3_POSE_MODEL || export V3_POSE_MODEL=$2
echo "$(date +%H:%M) start $1 phase=$5 ir=$6 alt=$7 seed=$8" >> $L
env PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 V3_DEVICE=cpu CPU_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    V3_IMGSZ=320 TARGET_FPS=8 V3_ROI_IMGSZ=256 V3_ROI_FULL_EVERY=8 V3_POSE_CONF=0.3 \
    V3_PREPROCESS=$3 V3_PREPROCESS_GREY_BELOW=$4 CACHE_DIR=$OUT/$1 ROI_PHASE=$5 SIMULATE_IR=$6 \
    SIMULATE_NIGHT_ALT=$7 SIMULATE_DARK_SEED=$8 SIMULATE_DARK=1.0 \
    python training/measure/cache_pose_streams.py > "$OUT/$1.p$5_ir$6_alt$7_s$8.log" 2>&1 \
  && echo "$(date +%H:%M) done $1 phase=$5 ir=$6 alt=$7 seed=$8" >> $L \
  || echo "$(date +%H:%M) FAILED $1 phase=$5 ir=$6 alt=$7 seed=$8" >> $L
