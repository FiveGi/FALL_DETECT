#!/bin/sh
# 7 Oct morning soak: 4 h, freeze test at +60 min, worker restart at +90 min (loads reader v2 + loop-end fix).
cd "$(dirname "$0")/.."
HOURS=4 FREEZE_AT_MIN=60 RESTART_AT_MIN=90 sh tools/soak_run.sh
