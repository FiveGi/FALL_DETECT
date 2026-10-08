#!/bin/sh
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
until python training/measure/list_model_caches.py nightaug_s43 > training/data/stage1/nightaug_s43_cache_lists.raw 2>/dev/null; do sleep 30; done
tr -d '\r' < training/data/stage1/nightaug_s43_cache_lists.raw > training/data/stage1/nightaug_s43_cache_lists.txt
L=training/data/stage1/nightaug_s43_cache_lists.txt; DAY=$(grep ^DAY $L | cut -d' ' -f2-); IR=$(grep ^IR $L | cut -d' ' -f2-); OW=$(grep ^OWNER $L | cut -d' ' -f2-)
S=$(pwd -W)/training/data/stage1; export PYTHONIOENCODING=utf-8
python training/measure/stage1_eval_pinned.py --name N1_nightaug43_deployed065 --model models --pose nightaug_s43 --day $DAY --ir $IR --owner $OW --fixed 0.65
V3_ENSEMBLE=$S/abl_A_s46/fall_classifier_v3.onnx,$S/abl_A_s47/fall_classifier_v3.onnx python training/measure/stage1_eval_pinned.py --name N2_nightaug43_E2_rule --model $S/abl_A_s45 --pose nightaug_s43 --day $DAY --ir $IR --owner $OW
echo "$(date +%H:%M) N s43 DONE" >> training/data/pose_ir_eval_cpu/queue.log
