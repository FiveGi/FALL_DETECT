#!/bin/sh
# Stage 1, all nine runs, unattended. Log: training/data/stage1/all.log
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/stage1/all.log; mkdir -p training/data/stage1
FV="C:/Users/USER/AppData/Local/Temp/claude/d--project-PROJECT/15c950a0-ee51-49fb-a8fc-fd89561db1d9/scratchpad/fv"
echo "$(date +%H:%M) waiting for FallVision rates" >> $L
until [ -f "$FV/probe.done" ]; do sleep 60; done
cp "$FV/fallvision_fps.json" training/data/fallvision_fps.json
( cd training && python build_source_fps.py ) >> $L 2>&1
python - >> $L 2>&1 <<'PY'
import json, sys
u = json.load(open('training/data/source_fps.json'))['unknown']
need = ('poses_fallvision/', 'poses_ofitw_yolopose_matched/', 'poses_caucafall_yolopose/', 'poses_yolopose/')
bad = [x for x in u if x.startswith(need)]
print('unknown rates in training sources:', len(bad), bad[:5])
sys.exit(1 if bad else 0)
PY
if [ $? -ne 0 ]; then echo "$(date +%H:%M) ABORT: unknown source rates" >> $L; exit 1; fi
for s in 42 43 44; do
  echo "$(date +%H:%M) base s$s" >> $L
  sh training/stage1_run.sh base $s >> $L 2>&1 || echo "FAILED base s$s" >> $L
done
echo "$(date +%H:%M) waiting for synthetic extraction" >> $L
# Training uses adults only (SYN_AGES), so wait for every adult clip's offset-0 file, not for the
# extractor to finish (it goes on to teens and children afterwards).
until python training/syn_adults_done.py; do sleep 60; done
for arm in syn10 syn25; do for s in 42 43 44; do
  echo "$(date +%H:%M) $arm s$s" >> $L
  sh training/stage1_run.sh $arm $s >> $L 2>&1 || echo "FAILED $arm s$s" >> $L
done; done
echo "$(date +%H:%M) ALL DONE" >> $L
