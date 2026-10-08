#!/usr/bin/env bash
# plan_v5, 8 Oct (Codex P1): rehearse the SERVER UPGRADE, not a fresh install. A copy of the version GitHub had before
# (origin/main 0506063, the closest guess to what the server runs -- the real SHA comes from runbook 0.1) is started
# with its own database, given a camera + an alert, then upgraded EXACTLY as docs/runbook_9oct.md section 0 says,
# checked, rolled back as section 0.5 says, and checked again. Preparation, not proof: the server's state is unknown.
#
#   bash tools/upgrade_rehearsal.sh            # needs the release tag on GitHub (after the 8 Oct 12:00 push)
# Env: TAG (systest-2026-10-09), OLD (0506063), DIR (D:/project/PROJECT/upgrade_rehearsal), REPO (GitHub URL)
#
# Safety (Codex review 7 Oct): the rehearsal runs as its OWN compose project (fallrehearsal) -- every plain
# `docker compose` below resolves to it through COMPOSE_PROJECT_NAME, inherited compose overrides are refused, and
# `down -v` can only ever remove fallrehearsal's volumes. The dev stack (same host ports) is STOPPED, not recreated,
# so its cpu4 override and data survive, and is STARTED again on every exit, success or failure, then health-checked.
# Uses the existing elderly-surveillance-app:latest image (no rebuild; requirements/Dockerfile unchanged since
# 0506063). LINE is OFF in the copy.
set -euo pipefail
MAIN="$(cd "$(dirname "$0")/.." && (pwd -W 2>/dev/null || pwd))"   # D:/... form: docker.exe gets a Windows path even with MSYS_NO_PATHCONV=1
DEV_PROJECT=backend-elderly-surveillance-main
TAG=${TAG:-systest-2026-10-09}; OLD=${OLD:-0506063}
DIR=${DIR:-D:/project/PROJECT/upgrade_rehearsal}; REPO=${REPO:-https://github.com/FiveGi/FALL_DETECT.git}
API=http://localhost:8932; FRONT=http://127.0.0.1:3001
log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { log "REHEARSAL FAILED: $*"; exit 1; }
for v in COMPOSE_PROJECT_NAME COMPOSE_FILE COMPOSE_PROFILES COMPOSE_PATH_SEPARATOR; do
  [ -n "${!v:-}" ] && { echo "refusing: $v is set in this shell ('${!v}'); unset it first"; exit 1; }
done
export COMPOSE_PROJECT_NAME=fallrehearsal
[ "$COMPOSE_PROJECT_NAME" != "$DEV_PROJECT" ] || exit 1
dev() { docker compose --project-directory "$MAIN" -p "$DEV_PROJECT" "$@"; }
sql() { docker compose exec -T db psql -U postgres -d postgres -tA -v ON_ERROR_STOP=1 -c "$1"; }
alerts() { sql "select count(*) from notification_history"; }
counts() { sql "select (select count(*) from cameras)||' cameras, '||(select count(*) from notification_history)||' alerts'"; }
token() { curl -s -X POST $API/api/auth/login -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin123"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])"; }
# --fail: an HTTP 500 is NOT ready (Codex P2); bounded so a hung backend cannot stall the loop.
healthy() { curl -s --fail --max-time 5 -o /dev/null $API/api/health; }
wait_api() { for i in $(seq 1 60); do healthy && return 0; sleep 3; done; fail "backend did not answer 2xx"; }
wait_alert_above() {   # $1 = current alert count; the camera must add at least one within 3 min
  for i in $(seq 1 36); do [ "$(alerts)" -gt "$1" ] && return 0; sleep 5; done; fail "$2: no new alert on clip 14 within 3 min"; }
VITE_PID=
cleanup() {
  set +e
  [ -n "$VITE_PID" ] && kill $VITE_PID 2>/dev/null
  if [ -d "$DIR" ]; then log "cleanup: removing the rehearsal stack (project $COMPOSE_PROJECT_NAME only)"; (cd "$DIR" && docker compose down -v); fi
  log "cleanup: starting the dev stack again"; dev start
  for i in $(seq 1 40); do healthy && { log "dev stack healthy"; return; }; sleep 3; done
  log "WARNING: dev stack did NOT come back healthy -- check 'docker compose ps' in $MAIN"
}

[ -e "$DIR" ] && fail "$DIR exists -- remove it first (left from an earlier rehearsal?)"
curl -s --fail --max-time 5 -o /dev/null http://127.0.0.1:9333/json/version || fail "no CDP browser on :9333 (see tools/smoke_test_frontend.mjs header)"
trap cleanup EXIT     # BEFORE the stop: a partial stop that fails must still restart dev (Codex P2)
log "stopping the dev stack (containers kept, volumes kept)"; dev stop

log "1. old version $OLD"
git clone -q "$REPO" "$DIR"; cd "$DIR"; git checkout -q "$OLD"
cp .env.example .env
sed -i 's/^LINE_ENABLED=.*/LINE_ENABLED=false/; s/^LINE_CHANNEL_ACCESS_TOKEN=.*/LINE_CHANNEL_ACCESS_TOKEN=/; s/^LINE_USER_ID=.*/LINE_USER_ID=/' .env
cp "$MAIN/frontend/.env" frontend/.env          # untracked on the server too; checkout never touches it
docker compose up -d; wait_api
H="Authorization: Bearer $(token)"
cid=$(curl -s -X POST $API/api/cameras -H "$H" -H 'Content-Type: application/json' -d \
  '{"name":"R cam14","room_name":"rehearsal","url":"/app/Test/14.mp4","detection_type":"fall_v2","alert_start_time":"00:00","alert_end_time":"23:59"}' \
  | python -c "import sys,json;d=json.load(sys.stdin);print(d.get('id') or d['camera']['id'])")
curl -s -X POST $API/api/cameras/$cid/start -H "$H" >/dev/null
wait_alert_above 0 "old version"
curl -s -X POST $API/api/cameras/$cid/stop -H "$H" >/dev/null; sleep 5
BEFORE=$(counts); N_BACKUP=$(alerts); log "old version data: $BEFORE"

log "2. runbook 0.1-0.4, literally"
[ -z "$(git status --short)" ] || fail "git status not clean in the copy"
STAMP=rehearsal-$(date +%H%M%S); B=~/backup-$STAMP; mkdir -p $B
cp .env $B/.env; git rev-parse HEAD > $B/commit.txt
docker compose exec -T db pg_dump -U postgres postgres > $B/db.sql
[ -s $B/db.sql ] && [ "$(grep -c 'CREATE TABLE' $B/db.sql)" -gt 0 ] || fail "backup empty"
git fetch -q origin --tags
SHA=$(git rev-parse "$TAG^{commit}"); log "tag $TAG = $SHA"
git checkout -q "$TAG"
git lfs pull   # runbook 0.3; .onnx and .mp4 are Git LFS files (Codex, 8 Oct: verify delivery from a fresh checkout)
for f in models/fall_classifier_v3.onnx models/fall_classifier_t2full_s45.onnx Test/14.mp4; do
  [ "$(stat -c %s "$f")" -gt 102400 ] || fail "$f is an LFS pointer, not the file ($(stat -c %s "$f") bytes)"
done
docker compose up -d --force-recreate; wait_api; sleep 20
docker compose logs backend | grep -i "Could not add" && fail "a column could not be added"
for c in notification_history.still_down_seconds line_settings.line_group_id; do
  sql "select count(*) from information_schema.columns where table_name='${c%.*}' and column_name='${c#*.}'" | grep -qx 1 || fail "column $c missing after upgrade"
done
AFTER=$(counts); log "after upgrade: $AFTER"; [ "$AFTER" = "$BEFORE" ] || fail "data changed by the upgrade: $BEFORE -> $AFTER"
H="Authorization: Bearer $(token)"
N=$(alerts); curl -s -X POST $API/api/cameras/$cid/start -H "$H" >/dev/null
wait_alert_above "$N" "new version"
docker compose logs celery_worker | grep "model identity" | tail -1 | grep -q classifier || fail "no model identity line"
curl -s -X POST $API/api/cameras/$cid/stop -H "$H" >/dev/null
log "frontend of the new version: npm ci + vite :3001 + smoke test (must be all pages OK)"
(cd frontend && npm ci --silent >/dev/null 2>&1) || fail "npm ci failed"
(cd frontend && exec npx vite --port 3001 --strictPort > "$B/vite.log" 2>&1) & VITE_PID=$!
for i in $(seq 1 40); do curl -s --fail --max-time 5 -o /dev/null $FRONT && break; sleep 2; done
APP_URL=$FRONT node "$MAIN/tools/smoke_test_frontend.mjs" | tee "$B/smoke_new.txt" | tail -3
grep -q "all pages OK" "$B/smoke_new.txt" || fail "frontend smoke test failed on the new version"

log "3. runbook 0.5 rollback, literally"
git checkout -q "$(cat $B/commit.txt)"; cp $B/.env .env
docker compose up -d --force-recreate; wait_api
ROLL=$(sql "select count(*) from cameras")" cameras"; log "after rollback: $ROLL, $(alerts) alerts"
[ "$(sql "select count(*) from cameras")" = "${BEFORE%% cameras*}" ] || fail "cameras changed by the rollback"
H="Authorization: Bearer $(token)"
[ "$(curl -s -o /dev/null -w '%{http_code}' $API/api/cameras -H "$H")" = 200 ] || fail "old version cameras API not 200"
N=$(alerts); curl -s -X POST $API/api/cameras/$cid/start -H "$H" >/dev/null
wait_alert_above "$N" "rolled-back version"
curl -s -X POST $API/api/cameras/$cid/stop -H "$H" >/dev/null
sleep 5   # vite reloads the rolled-back frontend files
APP_URL=$FRONT node "$MAIN/tools/smoke_test_frontend.mjs" > "$B/smoke_old.txt" || true
grep -qE "^PASS  login" "$B/smoke_old.txt" && grep -qE "^PASS  Monitor " "$B/smoke_old.txt" \
  || fail "rolled-back frontend: login or Monitor page failed (see $B/smoke_old.txt)"

log "4. restore test: the backup loads into a scratch database with the alert count it was taken with"
docker compose exec -T db psql -U postgres -v ON_ERROR_STOP=1 -c "drop database if exists restore_test" -c "create database restore_test" >/dev/null
docker compose exec -T db psql -U postgres -d restore_test -v ON_ERROR_STOP=1 -q < $B/db.sql >/dev/null || fail "restore failed"
R=$(docker compose exec -T db psql -U postgres -d restore_test -tA -c "select count(*) from notification_history")
[ "$R" = "$N_BACKUP" ] || fail "restored $R alerts, backup was taken with $N_BACKUP"

log "REHEARSAL PASSED: old $BEFORE -> upgrade $AFTER (alert + model identity + frontend OK) -> rollback (alert + login/Monitor OK) -> restore $R alerts; backup $B"
