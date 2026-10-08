#!/bin/sh
# T1 evaluation (pinned, nightaug_s44 caches; Codex freeze): each run alone at fixed 0.65 and by the half-A
# rule (THRESHOLD_GRID=extended), then each arm as a 3-seed E2-style ensemble (members fixed: s45 primary).
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
until grep -q "T1 TRAIN DONE" training/data/stage1/t1.log; do sleep 60; done
export PYTHONIOENCODING=utf-8 THRESHOLD_GRID=extended
S=$(pwd -W)/training/data/stage1; L=training/data/stage1/nightaug_s44_cache_lists.txt
DAY=$(grep ^DAY $L | cut -d' ' -f2-); IR=$(grep ^IR $L | cut -d' ' -f2-); OW=$(grep ^OWNER $L | cut -d' ' -f2-)
O=training/data/stage1/t1_eval.log
for arm in syn ctrl; do
  for s in 45 46 47; do
    python training/measure/stage1_eval_pinned.py --name T1_${arm}_s${s}_065 --model $S/t1_${arm}_s$s --pose nightaug_s44 --day $DAY --ir $IR --owner $OW --fixed 0.65 >> $O 2>&1
    python training/measure/stage1_eval_pinned.py --name T1_${arm}_s${s}_rule --model $S/t1_${arm}_s$s --pose nightaug_s44 --day $DAY --ir $IR --owner $OW >> $O 2>&1
  done
  V3_ENSEMBLE=$S/t1_${arm}_s46/fall_classifier_v3.onnx,$S/t1_${arm}_s47/fall_classifier_v3.onnx python training/measure/stage1_eval_pinned.py --name T1_${arm}_ens_rule --model $S/t1_${arm}_s45 --pose nightaug_s44 --day $DAY --ir $IR --owner $OW >> $O 2>&1
done
echo "T1 EVAL DONE" >> $O
