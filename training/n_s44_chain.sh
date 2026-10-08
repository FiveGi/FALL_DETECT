#!/bin/sh
# When POSE-IR nightaug s44 has trained: all its gate caches (8 day, IR 0/7/13, 8 owner; 3 at a time),
# then N1 unchanged (pre-registered 08:20 in AI_HANDOFF). Third seed for Codex's 3-seed requirement.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
until grep -q "seed 44 control 0" training/data/pose_ir/queue.log && grep -q "\(done\|FAILED\) seed 44 control 0" training/data/pose_ir/queue.log; do sleep 30; done
grep -q "done seed 44 control 0" training/data/pose_ir/queue.log || exit 1
M=$(pwd -W)/training/data/pose_ir/nightaug_s44_p2/weights/best.pt; J=training/data/gate_jobs_nightaug44.txt; : > $J
for p in 0 1 2 3 4 5 6 7; do echo "U nightaug_s44|$M|auto|0|$p|0|0|0" >> $J; done
for sd in 0 7 13; do echo "U nightaug_s44|$M|auto|0|0|1|0|$sd" >> $J; done
for p in 0 1 2 3 4 5 6 7; do echo "O nightaug_s44|$M|$p" >> $J; done
PAR=3 sh training/gate_jobs_runner.sh $J
until python training/measure/list_model_caches.py nightaug_s44 > training/data/stage1/nightaug_s44_cache_lists.raw 2>/dev/null; do sleep 30; done
tr -d '\r' < training/data/stage1/nightaug_s44_cache_lists.raw > training/data/stage1/nightaug_s44_cache_lists.txt
L=training/data/stage1/nightaug_s44_cache_lists.txt; DAY=$(grep ^DAY $L | cut -d' ' -f2-); IR=$(grep ^IR $L | cut -d' ' -f2-); OW=$(grep ^OWNER $L | cut -d' ' -f2-)
PYTHONIOENCODING=utf-8 python training/measure/stage1_eval_pinned.py --name N1_nightaug44_deployed065 --model models --pose nightaug_s44 --day $DAY --ir $IR --owner $OW --fixed 0.65
echo "$(date +%H:%M) N s44 DONE" >> training/data/pose_ir_eval_cpu/queue.log
