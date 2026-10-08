#!/bin/sh
# E5 chain (pre-registered .ai_evidence/plan_oct7_8/e5_preregistration.md; Codex + Gemini AGREE 8 Oct 02:09).
# Every heavy step under tools/train_watchdog.py (pre-launch RAM/fps/deadline checks, 60-s monitoring, hard stop 09:30).
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
export PYTHONIOENCODING=utf-8 STOP_AT="2026-10-08 09:30" LOG=training/data/pose_ir/e5_watchdog.log
L=training/data/pose_ir/e5_run.log
P=training/data/pose_ir
run() { echo "$(date +%H:%M) start $1" >> $L; shift; python tools/train_watchdog.py -- "$@"; rc=$?; echo "$(date +%H:%M) rc $rc" >> $L; [ $rc -eq 0 ] || exit $rc; }
run "E5 train"    sh -c 'WORKERS=0 ARM=e5 python training/e5_train.py > training/data/pose_ir/e5_train.out 2>&1'
run "A-cont train" sh -c 'WORKERS=0 ARM=acont python training/e5_train.py > training/data/pose_ir/acont_train.out 2>&1'
run "confirmation phantom probe" sh -c "IDS_FILE=training/data/system_test/phantom_confirm_ids.txt \
  EXTRA_MODELS=Acont=$P/acont_s45/weights/last.pt,E5=$P/e5_bg10_s45/weights/last.pt \
  PER_IMAGE_OUT=training/data/system_test/phantom_confirm_per_image.json \
  python training/measure/phantom_probe_coco.py > training/data/system_test/phantom_confirm.txt 2>&1"
echo "$(date +%H:%M) E5 CHAIN DONE (caches/gates next, separately)" >> $L
