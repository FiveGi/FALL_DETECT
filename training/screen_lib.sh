#!/bin/sh
# Shared by the overnight GPU screening runners (POSE-IR, NIGHT-NOISE, FALLPOSE). One cache build,
# every simulation variable explicit (Codex review); a failure stops the caller.
# build <log> <cache_dir> <roi_phase> <ir> <alt> <seed>
build() {
  echo "$(date +%H:%M) $(basename $2) build phase=$3 ir=$4 alt=$5 seed=$6" >> $1
  env CACHE_DIR=$2 ROI_PHASE=$3 SIMULATE_IR=$4 SIMULATE_NIGHT_ALT=$5 SIMULATE_DARK_SEED=$6 SIMULATE_DARK=1.0 \
      python training/measure/cache_pose_streams.py > "$2.p$3_ir$4_alt$5_s$6.log" 2>&1 \
    || { echo "$(date +%H:%M) FAILED $(basename $2) phase=$3 ir=$4 alt=$5 seed=$6 -- queue stopped" >> $1; return 1; }
}
# Priority tiers: what a first verdict needs (day phase 0, IR seed 0, NIGHT_ALT seed 0) for every
# model before the rest, so a slow night still ends with a paired comparison of all of them.
TIERS="0,0,0,0 0,1,0,0 0,0,1,0|4,0,0,0 0,1,0,7 0,0,1,7|0,1,0,13 0,0,1,13"
clean_partial() {   # a killed build can leave a truncated npz that the per-clip resume would skip
  python -c "
import glob, os, numpy as np
for f in glob.glob('$1/*/*.npz'):
    try: np.load(f)['counts']
    except Exception: os.remove(f); print('removed', f)
"
}
