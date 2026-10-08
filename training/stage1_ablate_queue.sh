#!/bin/sh
# Codex's first three ablations, in order: D (no balanced sampling), C (GMDCSA24 3 train / 1 val), CD.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/stage1/ablate.log
BALANCED_SAMPLING=0 sh training/stage1_ablate.sh abl_D 42 >> $L 2>&1 || echo "FAILED abl_D" >> $L
GMDCSA24_TRAIN_SUBJECTS=2,3,4 GMDCSA24_VAL_SUBJECTS=1 sh training/stage1_ablate.sh abl_C 42 >> $L 2>&1 || echo "FAILED abl_C" >> $L
BALANCED_SAMPLING=0 GMDCSA24_TRAIN_SUBJECTS=2,3,4 GMDCSA24_VAL_SUBJECTS=1 sh training/stage1_ablate.sh abl_CD 42 >> $L 2>&1 || echo "FAILED abl_CD" >> $L
echo "$(date +%H:%M) ABLATIONS DONE" >> $L
