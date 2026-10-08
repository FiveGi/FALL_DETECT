#!/bin/sh
# T2 evaluation (pinned, nightaug_s44 caches; same protocol as t1_eval): each TRUNC run at fixed 0.65 and by the
# half-A rule (extended grid); controls = t1_ctrl_s45/46/47 (same RESAMPLE_FPS=8 recipe, no trunc), already scored.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
export PYTHONIOENCODING=utf-8 THRESHOLD_GRID=extended
S=$(pwd -W)/training/data/stage1; L=training/data/stage1/nightaug_s44_cache_lists.txt
DAY=$(grep ^DAY $L | cut -d' ' -f2-); IR=$(grep ^IR $L | cut -d' ' -f2-); OW=$(grep ^OWNER $L | cut -d' ' -f2-)
O=training/data/stage1/t2_eval.log
for m in trunc_s45 trunc_s46 trunc_s47 truncfull_s45; do
  python training/measure/stage1_eval_pinned.py --name T2_${m}_065 --model $S/t2_$m --pose nightaug_s44 --day $DAY --ir $IR --owner $OW --fixed 0.65 >> $O 2>&1 || { echo "FAILED $m" >> $O; exit 1; }
  python training/measure/stage1_eval_pinned.py --name T2_${m}_rule --model $S/t2_$m --pose nightaug_s44 --day $DAY --ir $IR --owner $OW >> $O 2>&1 || { echo "FAILED $m rule" >> $O; exit 1; }
done
echo "T2 EVAL DONE" >> $O
