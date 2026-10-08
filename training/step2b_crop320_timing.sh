#!/bin/sh
# PROPOSAL (plan round 3 D7; NOT launched until Codex + Gemini agree): CPU timing of crop 320 vs 256, run on a
# quiet machine after a NEW 'T3B DONE' (or a t3b stop). Same frames/protocol as step 2 (pipeline_timing.py).
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/stage1/t3b.log; n=$(cat $L 2>/dev/null | wc -l)
until tail -n +$((n + 1)) $L 2>/dev/null | grep -q "T3B DONE\|-- stop"; do sleep 120; done
P=training/data/pose_ir/nightaug_s44_p2/weights/best.pt; O=training/data/multi_diag_v2/pipeline_timing_crop320.txt
PYTHONIOENCODING=utf-8 python training/measure/pipeline_timing.py deployed=- crop320=-@320 s44=$P s44_crop320=$P@320 > $O 2>&1 \
  && [ "$(grep -c sustained $O)" -eq 4 ] && echo "$(date '+%d %H:%M') STEP2B timing done" >> training/data/holiday_queue.log \
  || echo "$(date '+%d %H:%M') STEP2B timing FAILED" >> training/data/holiday_queue.log
