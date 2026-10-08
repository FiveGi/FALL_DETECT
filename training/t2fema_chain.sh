#!/bin/sh
# Runs the two EMA arms one after the other (one GPU job at a time). Arm 0.9999 starts only if arm 0.999 finished.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
sh training/t2fema_seeds.sh && grep -q "T2FEMA DONE" training/data/stage1/t2fema.log && sh training/t2fema4_seeds.sh
