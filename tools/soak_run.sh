#!/usr/bin/env bash
# Plan S5: overnight soak of the CPU stack (4 cores pinned) with simulated RTSP cameras.
#   cam1 Test/1.mp4, cam2 Test/4.mp4, cam3 EMPTY ROOM (Le2i coffee-room frame, dev-only), cam4 Test/9.mp4, 720p H.264.
# Every 10 min: worker memory/CPU, last fps per camera, alerts per camera, errors, LINE still off.
# Every 30 min: cam2 publisher stopped 60 s then restarted (camera drop + recovery).
# At +2 h: cam4 frozen 60 s (docker pause). At +4 h: celery_worker restarted (resume test).
# Output: training/data/system_test/soak.txt   Usage: HOURS=9 bash tools/soak_run.sh
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=training/data/system_test/soak.txt
HOURS=${HOURS:-9}; FREEZE_AT_MIN=${FREEZE_AT_MIN:-120}; RESTART_AT_MIN=${RESTART_AT_MIN:-240}
W=backend-elderly-surveillance-main-celery_worker-1
DB="docker compose exec -T db psql -U postgres -d postgres -tA -c"
PYTHONIOENCODING=utf-8 python tools/system_test.py remove >/dev/null 2>&1
SIM_CLIPS="Test/1.mp4 Test/4.mp4 training/data/system_test/empty_room.mp4 Test/9.mp4" bash tools/rtsp_sim.sh up 4 h264 720 >/dev/null
sleep 5
START=$(date -u +%Y-%m-%dT%H:%M:%SZ); START_S=$(date +%s)
FIRST_ID=$($DB "SELECT coalesce(max(id),0) FROM notification_history;")
PYTHONIOENCODING=utf-8 python tools/system_test.py add 4 >/dev/null 2>&1
echo "soak start $(date '+%Y-%m-%d %H:%M') hours=$HOURS first_notification_id_after=$FIRST_ID" > $OUT
froze=0; restarted=0; i=0
while [ $(( $(date +%s) - START_S )) -lt $(( HOURS * 3600 )) ]; do
  sleep 600; i=$((i+1)); el=$(( ($(date +%s) - START_S) / 60 ))
  {
    echo "--- t+${el}min $(date '+%H:%M')"
    docker stats --no-stream --format '{{.Name}} cpu {{.CPUPerc}} mem {{.MemUsage}}' | grep -E "worker|backend-1|db-1" | sed 's/backend-elderly-surveillance-main-//'
    docker logs --since 10m $W 2>&1 | grep "Detection rate" | sed -E 's/.*\[Camera ([0-9]+)\].*Detection rate: ([0-9.]+)\/.*/cam\1 \2/' | sort | awk '{last[$1]=$2} END{for(c in last) printf "%s %s  ", c, last[c]; print ""}'
    echo "alerts since start (camera: fall / other): $($DB "SELECT string_agg(c.name||': '||sum_f||'/'||sum_o, ', ') FROM (SELECT camera_id, sum((detection_type LIKE '%fall%')::int) sum_f, sum((detection_type NOT LIKE '%fall%')::int) sum_o FROM notification_history WHERE id > $FIRST_ID GROUP BY camera_id) x JOIN cameras c ON c.id=x.camera_id;")"
    echo "LINE enabled rows: $($DB "SELECT count(*) FROM line_settings WHERE enabled;")"
    echo "errors/tracebacks (10 min): $(docker logs --since 10m $W 2>&1 | grep -c -E 'Traceback|raised unexpected')  reconnects: $(docker logs --since 10m $W 2>&1 | grep -c 'reconnected')  tracks reset: $(docker logs --since 10m $W 2>&1 | grep -c 'tracks reset')"
  } >> $OUT
  if [ $((i % 3)) -eq 0 ]; then
    docker stop rtsp-pub2 >/dev/null; sleep 60; docker start rtsp-pub2 >/dev/null; echo "    [drop test] cam2 stopped 60 s and restarted $(date '+%H:%M')" >> $OUT
  fi
  if [ $froze -eq 0 ] && [ $el -ge ${FREEZE_AT_MIN:-120} ]; then
    docker pause rtsp-pub4 >/dev/null; sleep 60; docker unpause rtsp-pub4 >/dev/null; froze=1; echo "    [freeze test] cam4 paused 60 s $(date '+%H:%M')" >> $OUT
  fi
  if [ $restarted -eq 0 ] && [ $el -ge ${RESTART_AT_MIN:-240} ]; then
    docker restart $W >/dev/null; restarted=1; echo "    [restart test] celery_worker restarted $(date '+%H:%M')" >> $OUT
  fi
done
echo "SOAK DONE $(date '+%H:%M')" >> $OUT
