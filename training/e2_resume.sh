#!/bin/sh
# Resume E2 after the 2026-10-01 pause: seeds 46/47 of recipe A, then the frozen E2 ensemble eval.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/stage1/ablate.log
for s in ${E2_SEEDS:-46 47}; do
  RESAMPLE_FPS=0 TEMPORAL_STRIDE=2 EXCLUDE_NO_TIMEBASE=1 sh training/stage1_ablate.sh abl_A $s >> $L 2>&1 || echo "FAILED abl_A s$s" >> $L
done
D=D:/project/PROJECT/Backend-Elderly-Surveillance-main/training/data/stage1
V3_ENSEMBLE=$D/abl_A_s46/fall_classifier_v3.onnx,$D/abl_A_s47/fall_classifier_v3.onnx python training/measure/stage1_eval.py ensA_45_46_47 $D/abl_A_s45 >> $L.ens 2>&1 && echo "$(date +%H:%M) ensA_45_46_47 $(tail -n 1 $L.ens)" >> $L
echo "$(date +%H:%M) E2 DONE" >> $L
