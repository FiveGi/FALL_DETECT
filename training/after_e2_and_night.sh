#!/bin/sh
# RAM rule (12 GB free): POSE-IR training (~6 GB at WORKERS=1) starts only after E2 and the
# night-screening caches have both finished.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
until grep -q "E2 DONE" training/data/stage1/ablate.log && grep -q "NIGHT EVAL CACHES DONE" training/data/pose_ir_eval/queue.log; do sleep 60; done
WORKERS=1 sh training/pose_ir_resume.sh
