#!/usr/bin/env bash
# Plan S3: how many simulated RTSP cameras the 4-core stack sustains, per resolution.
# Needs the CPU stack up with docker-compose.cpu4.yml. Writes training/data/system_test/capacity.txt
# Each config: (re)start N simulated cameras, add/start them through the API, wait WAIT s, then
# record each camera's "Detection rate" lines and container CPU/RAM.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=training/data/system_test; mkdir -p $OUT; R=$OUT/capacity.txt
WAIT=${WAIT:-240}
W=backend-elderly-surveillance-main-celery_worker-1
run() {  # $1 N  $2 res  $3 codec
  PYTHONIOENCODING=utf-8 python tools/system_test.py remove >/dev/null 2>&1
  bash tools/rtsp_sim.sh up "$1" "$3" "$2" >/dev/null
  sleep 5
  T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  PYTHONIOENCODING=utf-8 python tools/system_test.py add "$1" >/dev/null 2>&1
  sleep "$WAIT"
  {
    echo "=== $1 camera(s) $3 ${2}p  ($(date '+%H:%M'))"
    docker logs --since "$T0" $W 2>&1 | grep "Detection rate" | sed -E 's/.*\[Camera ([0-9]+)\].*Detection rate: ([0-9.]+)\/([0-9.]+) fps.*/cam\1 \2/' | tail -n $(( $1 * 3 ))
    docker stats --no-stream --format '{{.Name}} cpu {{.CPUPerc}} mem {{.MemUsage}}' | grep -E "worker|rtsp-server" | sed 's/backend-elderly-surveillance-main-//'
  } >> $R
}
echo "capacity run $(date '+%Y-%m-%d %H:%M') WAIT=$WAIT (4 cores pinned, whole stack)" > $R
for spec in "2 1080 h264" "3 1080 h264" "4 1080 h264" "4 720 h264" "4 360 h264" "2 1080 h265"; do run $spec; done
PYTHONIOENCODING=utf-8 python tools/system_test.py remove >/dev/null 2>&1
echo "DONE $(date '+%H:%M')" >> $R
