#!/bin/sh
# T3 POSE-NEG: POSE-IR recipe (night aug, CONTROL=0) + 10% person-free backgrounds, seeds 42/43/44.
# Matched controls = existing nightaug_s42/43/44 (same recipe, no backgrounds). Then object false persons
# (COCO val, disjoint) for all six + presence check. Launched detached.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/holiday_queue.log
until [ "$(ls D:/project/PROJECT/datasets/posneg/images | wc -l)" -ge 5000 ]; do sleep 60; done
for s in 42 43 44; do
  [ -f training/data/pose_ir/posneg_s${s}_p2/weights/best.pt ] && continue
  # Owner games on this PC: never start a training run while the game is open (RAM + GPU).
  while tasklist //FI "IMAGENAME eq TOTClient-Win64-Shipping.exe" //NH | grep -qi TOTClient; do sleep 60; done
  Y=$(SEED=$s python training/posneg_data.py | tail -1)
  [ -f "$Y" ] || { echo "$(date '+%d %H:%M') T3 data build FAILED s$s -- stop" >> $L; exit 1; }
  echo "$(date '+%d %H:%M') T3 posneg s$s start" >> $L
  CONTROL=0 SEED=$s WORKERS=${WORKERS:-4} RUN_NAME=posneg_s$s COCO_POSE_YAML=$Y PYTHONIOENCODING=utf-8 \
    python training/finetune_pose_ir.py > training/data/posneg_s$s.log 2>&1 \
    && echo "$(date '+%d %H:%M') T3 posneg s$s done" >> $L || { echo "$(date '+%d %H:%M') T3 posneg s$s FAILED -- stop" >> $L; exit 1; }
done
P=training/data/pose_ir
PYTHONIOENCODING=utf-8 python training/measure/object_false_person.py $P/nightaug_s42_p2/weights/best.pt@320 $P/posneg_s42_p2/weights/best.pt@320 \
  $P/nightaug_s43_p2/weights/best.pt@320 $P/posneg_s43_p2/weights/best.pt@320 $P/nightaug_s44_p2/weights/best.pt@320 $P/posneg_s44_p2/weights/best.pt@320 \
  > training/data/multi_diag_v2/object_false_person_t3.txt 2>&1
PYTHONIOENCODING=utf-8 python training/measure/teacher_presence.py $P/posneg_s42_p2/weights/best.pt@320 $P/posneg_s43_p2/weights/best.pt@320 $P/posneg_s44_p2/weights/best.pt@320 \
  2>&1 | grep half > training/data/multi_diag_v2/presence_t3.txt
echo "$(date '+%d %H:%M') T3 DONE" >> $L
echo ok > training/data/holiday_queue.log.t3marker   # read by step1_controls.sh / step2_timing.sh
