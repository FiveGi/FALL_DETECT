#!/bin/sh
# After Codex code review 2026-10-03: rerun every result that fed the reports with the fixed tools.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
export PYTHONIOENCODING=utf-8; O=training/data/rerun_0310; mkdir -p $O
S=$(pwd -W)/training/data/stage1
for tag in nightaug_s42 nightaug_s43 nightaug_s44; do
  L=training/data/stage1/${tag}_cache_lists.txt; DAY=$(grep ^DAY $L | cut -d' ' -f2-); IR=$(grep ^IR $L | cut -d' ' -f2-); OW=$(grep ^OWNER $L | cut -d' ' -f2-)
  python training/measure/stage1_eval_pinned.py --name N1_${tag}_v2 --model models --pose $tag --day $DAY --ir $IR --owner $OW --fixed 0.65 >> $O/pinned.log 2>&1 || echo "FAILED N1 $tag" >> $O/pinned.log
done
L=training/data/stage1/fallpose_cache_lists.txt; DAY=$(grep ^DAY $L | cut -d' ' -f2- | tr -d '\r'); IR=$(grep ^IR $L | cut -d' ' -f2- | tr -d '\r'); OW=$(grep ^OWNER $L | cut -d' ' -f2- | tr -d '\r')
python training/measure/stage1_eval_pinned.py --name F1_fallpose_v2 --model models --pose fallpose_s42 --day $DAY --ir $IR --owner $OW --fixed 0.65 >> $O/pinned.log 2>&1 || echo "FAILED F1" >> $O/pinned.log
P=training/data/pose_ir
python training/measure/teacher_presence.py models/yolo26s-pose.pt@320 $P/colour_s42_p2/weights/best.pt@320 $P/nightaug_s44_p2/weights/best.pt@320 $P/fallpose_s42_p2/weights/best.pt@320 $P/fallnight_s42_p2/weights/best.pt@320 models/yolo26x-pose.pt@1280 2>&1 | grep -E "half|Error" > $O/teacher_presence_v2.txt
for h in A B; do python training/measure/night_motion_probe.py $h 2>&1 | grep -E "half|Error"; done > $O/night_motion_v2.txt
echo done > $O/DONE
