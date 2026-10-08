#!/bin/sh
# plan_v5 T2c: uncapped (V3_NUM_POSES=99) owner-segment pose caches for all 8 ROI phases, then the parity check
# against the pinned cap-4 caches (stock pose = production). Starts only after the Le2i runs are done and enough
# RAM is free (7 Oct: 2.4 GB available with Le2i x4 running). Stops on any build failure or parity mismatch.
#   sh training/uncapped_owner.sh                         stock pose (production), stock cache list
#   POSE=<path used by the pinned caches> LIST=training/data/stage1/nightaug_s44_cache_lists.txt sh training/uncapped_owner.sh
# A POSE that differs from the pinned caches' key makes the twin directory differ, and the parity check fails loudly.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
L=training/data/stage1/uncapped_owner.log; LIST=${LIST:-training/data/stage1/stock_cache_lists.txt}
until grep -qE "LE2I RUNS (DONE|INCOMPLETE)" training/data/le2i_final/run.log 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M) le2i finished; building uncapped owner caches (PAR=${PAR:-3})" >> $L
seq 0 7 | xargs -P ${PAR:-3} -I{} sh -c 'V3_POSE_MODEL=${POSE:-yolo26s-pose.pt} V3_DEVICE=cpu ROI_PHASE={} V3_NUM_POSES=99 CPU_THREADS=4 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 \
  python training/measure/cache_owner_segments.py > training/data/stage1/uncapped_owner_p{}.log 2>&1 \
  && echo "$(date +%H:%M) built phase {}" >> '"$L"' || echo "$(date +%H:%M) FAILED phase {}" >> '"$L"
grep -q FAILED $L && { echo "$(date +%H:%M) STOP: a build failed" >> $L; exit 1; }
# Pair each pinned phase with its uncapped twin (same key except num_poses), then check parity.
PAIRS=$(python - "$LIST" <<'EOF'
import sys, json, os, hashlib
owner = [l.split()[1:] for l in open(sys.argv[1]) if l.startswith('OWNER')][0]
root = 'training/data/pose_cache_owner'
out = []
for d in owner:
    k = json.load(open(os.path.join(d, 'key.json'))); k['num_poses'] = 99
    twin = os.path.join(root, hashlib.sha1(json.dumps(k, sort_keys=True).encode()).hexdigest()[:12])
    out += [d, twin]
print(' '.join(out))
EOF
)
# 126 = the owner segment set every pinned owner cache holds. Parity failure must END the script non-zero (Codex):
# `a && echo OK || echo FAILED` exits 0 even when the check fails.
if python training/measure/check_cap_parity.py --expect 126 $PAIRS >> $L 2>&1; then
  echo "$(date +%H:%M) PARITY OK all phases" >> $L
else
  echo "$(date +%H:%M) PARITY FAILED -- stop, do not compare caps" >> $L; exit 1
fi
