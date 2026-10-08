#!/bin/sh
# Recipe A ensembles (Codex design pending review): E1 = seeds 42/43/44 averaged; E2 = new seeds 45/46/47.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/stage1/ablate.log
V3_ENSEMBLE=D:/project/PROJECT/Backend-Elderly-Surveillance-main/training/data/stage1/abl_A_s43/fall_classifier_v3.onnx,D:/project/PROJECT/Backend-Elderly-Surveillance-main/training/data/stage1/abl_A_s44/fall_classifier_v3.onnx   python training/measure/stage1_eval.py ensA_42_43_44 D:/project/PROJECT/Backend-Elderly-Surveillance-main/training/data/stage1/abl_A_s42 >> $L.ens 2>&1 && echo "$(date +%H:%M) ensA_42_43_44 $(tail -n 1 $L.ens)" >> $L
for s in 45 46 47; do
  RESAMPLE_FPS=0 TEMPORAL_STRIDE=2 EXCLUDE_NO_TIMEBASE=1 sh training/stage1_ablate.sh abl_A $s >> $L 2>&1 || echo "FAILED abl_A s$s" >> $L
done
V3_ENSEMBLE=D:/project/PROJECT/Backend-Elderly-Surveillance-main/training/data/stage1/abl_A_s46/fall_classifier_v3.onnx,D:/project/PROJECT/Backend-Elderly-Surveillance-main/training/data/stage1/abl_A_s47/fall_classifier_v3.onnx   python training/measure/stage1_eval.py ensA_45_46_47 D:/project/PROJECT/Backend-Elderly-Surveillance-main/training/data/stage1/abl_A_s45 >> $L.ens 2>&1 && echo "$(date +%H:%M) ensA_45_46_47 $(tail -n 1 $L.ens)" >> $L
echo "$(date +%H:%M) ENSEMBLE DONE" >> $L
