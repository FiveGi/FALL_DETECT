#!/bin/sh
# After FALLPOSE s43 finishes: FALLNIGHT s42 = FALLPOSE's 15% fall/lying mix WITH POSE-IR's night
# degradation (CONTROL=0: 15% grey, 15% IR, blur p0.5) -- the two levers that each won one half of the
# overnight screen (lying people / night), in one model. Same schedule, imgsz 320, seed 42.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/fallpose_queue.log
while powershell -NoProfile -c "if (Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | ? { \$_.CommandLine -match 'finetune_pose_ir' -and \$_.CommandLine -notmatch 'multiprocessing' }) { exit 0 } else { exit 1 }"; do sleep 30; done
grep -q "fallpose_s43" training/data/fallpose_s43.log 2>/dev/null && tail -1 training/data/fallpose_s43.log | grep -q "^done" && echo "$(date +%H:%M) fallpose s43 done" >> $L
echo "$(date +%H:%M) fallnight s42 start" >> $L
CONTROL=0 SEED=42 WORKERS=2 RUN_NAME=fallnight_s42 COCO_POSE_YAML=D:/project/PROJECT/datasets/fallpose/fallpose_mix_s42.yaml \
  PYTHONIOENCODING=utf-8 python training/finetune_pose_ir.py > training/data/fallnight_s42.log 2>&1 \
  && echo "$(date +%H:%M) fallnight s42 done" >> $L || echo "$(date +%H:%M) fallnight s42 FAILED" >> $L
