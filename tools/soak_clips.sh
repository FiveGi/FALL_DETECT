#!/usr/bin/env bash
# Overnight plan R2 (8 Oct, agreed by all three): soak of the EXACT 9 Oct configuration -- option A, the cap chosen by
# the pre-registered latency gate, 4 pinned cores -- with clip cameras added the way the web form adds them
# (file source /app/...): ST clip14 (real fall, loops -> repeated alerts), ST clip17 (crowd, no fall: must stay silent),
# ST clip1 (compilation), ST empty (empty room: must stay silent). LINE stays off.
# Every 10 min: worker memory/CPU, last fps per camera, alerts per camera, LINE rows, tracebacks.
# At +RESTART_AT_MIN (default 120): celery_worker restarted (resume test on option A).
# PASS (judged afterwards): 0 tracebacks, worker memory growth < 10 % first->last sample, 0 fall alerts on clip17 and
# empty, every camera still reporting fps at the end.   Usage: HOURS=4 bash tools/soak_clips.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$PATH:/c/Program Files/Docker/Docker/resources/bin"
OUT=training/data/system_test/soak_clips.txt
HOURS=${HOURS:-4}; RESTART_AT_MIN=${RESTART_AT_MIN:-120}
W=backend-elderly-surveillance-main-celery_worker-1
DB="docker compose exec -T db psql -U postgres -d postgres -tA -c"
API=${API:-http://localhost:8932}
PYTHONIOENCODING=utf-8 python tools/system_test.py remove >/dev/null 2>&1
# Every exit -- normal end, error or Ctrl+C -- removes the soak's cameras (ST clip* and ST empty) (Codex, 8 Oct: the
# empty-room camera survived SOAK DONE and kept running).
trap 'PYTHONIOENCODING=utf-8 python tools/system_test.py remove >> $OUT 2>&1' EXIT
START_S=$(date +%s)
FIRST_ID=$($DB "SELECT coalesce(max(id),0) FROM notification_history;")
PYTHONIOENCODING=utf-8 python tools/system_test.py clips 14 17 1 >/dev/null 2>&1
TOKEN=$(curl -s -X POST $API/api/auth/login -H 'Content-Type: application/json' -d '{"username":"admin","password":"admin123"}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
CID=$(curl -s -X POST $API/api/cameras -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' -d \
  '{"name":"ST empty","room_name":"rehearsal","url":"/app/training/data/system_test/empty_room.mp4","detection_type":"fall_v2","alert_start_time":"00:00","alert_end_time":"23:59"}' \
  | python -c "import sys,json;d=json.load(sys.stdin);print(d.get('id') or d['camera']['id'])")
curl -s -X POST $API/api/cameras/$CID/start -H "Authorization: Bearer $TOKEN" >/dev/null
echo "soak_clips start $(date '+%Y-%m-%d %H:%M') hours=$HOURS first_notification_id_after=$FIRST_ID cameras: $($DB "SELECT string_agg(id||'='||name, ', ') FROM cameras WHERE name LIKE 'ST %';")" > $OUT
echo "identity: $(docker logs $W 2>&1 | grep 'model identity' | tail -1 | cut -c1-260)" >> $OUT
restarted=0
while [ $(( $(date +%s) - START_S )) -lt $(( HOURS * 3600 )) ]; do
  sleep 600; el=$(( ($(date +%s) - START_S) / 60 ))
  {
    echo "--- t+${el}min $(date '+%H:%M')"
    docker stats --no-stream --format '{{.Name}} cpu {{.CPUPerc}} mem {{.MemUsage}}' | grep -E "worker|backend-1|db-1" | sed 's/backend-elderly-surveillance-main-//'
    docker logs --since 10m $W 2>&1 | grep "Detection rate" | sed -E 's/.*\[Camera ([0-9]+)\].*Detection rate: ([0-9.]+)\/.*/cam\1 \2/' | sort | awk '{last[$1]=$2} END{for(c in last) printf "%s %s  ", c, last[c]; print ""}'
    echo "alerts since start (camera: fall / other): $($DB "SELECT string_agg(c.name||': '||sum_f||'/'||sum_o, ', ') FROM (SELECT camera_id, sum((detection_type LIKE '%fall%')::int) sum_f, sum((detection_type NOT LIKE '%fall%')::int) sum_o FROM notification_history WHERE id > $FIRST_ID GROUP BY camera_id) x JOIN cameras c ON c.id=x.camera_id;")"
    echo "LINE enabled rows: $($DB "SELECT count(*) FROM line_settings WHERE enabled;")"
    echo "errors/tracebacks (10 min): $(docker logs --since 10m $W 2>&1 | grep -c -E 'Traceback|raised unexpected')"
  } >> $OUT
  if [ $restarted -eq 0 ] && [ $el -ge $RESTART_AT_MIN ]; then
    docker restart $W >/dev/null; restarted=1; echo "    [restart test] celery_worker restarted $(date '+%H:%M')" >> $OUT
  fi
done
echo "SOAK DONE $(date '+%H:%M')" >> $OUT
