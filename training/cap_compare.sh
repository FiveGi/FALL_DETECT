#!/bin/sh
# plan_v5 T2d: owner-segment gate, cap 4 vs cap 8, option A (nightaug_s44 pose caches + t2_truncfull_s45 @0.70).
# Both caps replay the SAME uncapped caches (parity with the pinned cap-4 caches checked: uncapped_owner.log), so the
# cap is the only difference. Sanity first: cap 4 on the uncapped caches must reproduce cap 4 on the pinned caches.
# Pre-registered rule (plan_v5 T2): adopt 8 only if no owner number is worse, no new false alarm, latency +<=3%.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
O=training/data/stage1/cap_compare; mkdir -p $O; L=$O/run.log; : > $L
MODEL=training/data/stage1/t2_truncfull_s45
export V3_POSE_MODEL="D:/project/PROJECT/Backend-Elderly-Surveillance-main/training/data/pose_ir/nightaug_s44_p2/weights/best.pt"
export V3_THRESHOLD=0.70 PYTHONIOENCODING=utf-8
PINNED=$(grep ^OWNER training/data/stage1/nightaug_s44_cache_lists.txt | cut -d' ' -f2-)
UNCAPPED="323f9a38a467 091620bbf01e bd0660117b74 34230000458e 0f90ae9fad9b 2f5b60cab098 7acddfd7d180 ea858a72c9f3"
run() {   # $1 tag, $2 cap, $3.. cache dirs
  tag=$1; cap=$2; shift 2; files=""; p=0
  for c in "$@"; do
    f=$O/${tag}_p$p.json; rm -f $f
    V3_NUM_POSES=$cap python training/measure/replay_owner_segments.py "$c" $MODEL $f > $O/${tag}_p$p.log 2>&1 \
      || { echo "FAILED $tag p$p" >> $L; exit 1; }
    files="$files $f"; p=$((p+1))
  done
  python training/measure/score_incidents.py $files > $O/${tag}_score.txt 2>&1 || { echo "FAILED score $tag" >> $L; exit 1; }
  echo "== $tag (cap $cap)" >> $L; cat $O/${tag}_score.txt >> $L
}
run pinned_cap4 4 $PINNED
run uncapped_cap4 4 $(for u in $UNCAPPED; do printf 'training/data/pose_cache_owner/%s ' $u; done)
run uncapped_cap8 8 $(for u in $UNCAPPED; do printf 'training/data/pose_cache_owner/%s ' $u; done)
echo "DONE" >> $L
