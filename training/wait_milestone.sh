#!/bin/bash
# Exit with the first new milestone line in any overnight queue log (for the assistant's wake-ups).
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main/training/data"
F="pose_ir_eval_cpu/queue.log stage1/ablate.log pose_ir/queue.log pose_ir_eval/queue.log pose_ir_eval/noise_queue.log fallpose_queue.log"
touch $F; declare -A n; for f in $F; do n[$f]=$(wc -l < $f); done
while true; do
  for f in $F; do c=$(wc -l < $f)
    if [ $c -gt ${n[$f]} ]; then
      new=$(tail -n $((c-${n[$f]})) $f | grep -E "done|DONE|FAIL|ok|ensA|abl_A_s4|complete|INCOMPLETE|start seed|fallpose s" | grep -v -E "^[0-9:]+ (done|start) (owner )?(stock|nightaug|colour|noise|fallpose|fallnight)_?[a-zA-Z0-9]* phase=")
      n[$f]=$c; [ -n "$new" ] && { echo "$(date +%H:%M) $f: $new" | cut -c1-220; exit 0; }
    fi
  done; sleep 30
done
