#!/usr/bin/env bash
# SUPERSEDED (8 Oct): we cannot SSH to the server; the 9 Oct build is delivered by GitHub tag -- see docs/runbook_9oct.md. Kept for reference only.
# Plan S7 (8 Oct, WITH the owner): copy the 9 Oct build to the server, with a backup first and a rollback path.
# DRY RUN by default: prints what it would do. Nothing touches the server unless GO=1.
#
#   SERVER_DIR=/path/on/server bash tools/deploy_server.sh            # dry run
#   SERVER_DIR=/path/on/server GO=1 bash tools/deploy_server.sh       # backup + copy + recreate
#   SERVER_DIR=/path/on/server GO=1 bash tools/deploy_server.sh rollback <backup-stamp>
# Host alias from ~/.ssh/config (default server-14). The app code is bind-mounted (.:/app), so copying files and
# recreating the containers is enough; requirements did not change, so no image rebuild is needed.
set -euo pipefail
cd "$(dirname "$0")/.."
H=${HOST:-server-14}
D=${SERVER_DIR:?set SERVER_DIR to the project folder on the server (from tools/server_inventory.sh)}
STAMP=$(date +%Y%m%d-%H%M)
run() { if [ "${GO:-0}" = 1 ]; then ssh -o BatchMode=yes "$H" "$1"; else echo "[dry] ssh $H: $1"; fi; }
copy() { if [ "${GO:-0}" = 1 ]; then scp -q "$1" "$H:$D/$1"; else echo "[dry] scp $1 -> $H:$D/$1"; fi; }

if [ "${1:-}" = rollback ]; then
  B=${2:?backup stamp}
  run "cd $D && tar -xzf ../deploy-backup-$B/code.tgz && cp ../deploy-backup-$B/.env .env && docker compose up -d --force-recreate"
  echo "rolled back to $B (database not restored; restore ../deploy-backup-$B/db.sql by hand only if needed)"; exit 0
fi

# docker-compose.yml is NOT copied: the server's copy may carry its own ports (15080/15081) and limits. Nothing in it
# is required -- RTSP-over-TCP is set in code (setdefault), and V3_TARGET_FPS only needs .env if it is changed.
# BEFORE GO=1: compare these files with the server's versions (server_inventory.sh shows its commit) -- the dev copies
# also carry earlier uncommitted work; diff first so nothing unexpected ships.
# Files that make up the 9 Oct build (code changes since the server's version + new tools + candidate models).
FILES=(app/services/camera_manager.py app/services/detection_dispatch.py app/services/escalation_service.py
       app/services/line_service.py app/services/notification_service.py app/services/stream_service.py
       app/detection/v3_fall_detection.py frontend/src/utils/detectionType.js
       tools/line_backlog.sh tools/server_inventory.sh docs/runbook_9oct.md
       models/pose_nightaug_s44.pt models/fall_classifier_t2full_s45.onnx)

echo "== 1. backup on the server ($STAMP): code (without heavy data), .env, database dump, image ids"
run "mkdir -p $D/../deploy-backup-$STAMP && cd $D && tar --exclude=./training --exclude=./Test --exclude=./runs -czf ../deploy-backup-$STAMP/code.tgz . && cp .env ../deploy-backup-$STAMP/.env && docker compose exec -T db pg_dump -U postgres postgres > ../deploy-backup-$STAMP/db.sql && docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' > ../deploy-backup-$STAMP/images.txt"
echo "== 2. copy the build"
for f in "${FILES[@]}"; do copy "$f"; done
echo "== 3. LINE backlog check (must be 0 before LINE is switched on)"
run "cd $D && bash tools/line_backlog.sh check"
echo "== 4. recreate containers and prove which model runs"
run "cd $D && docker compose up -d --force-recreate && sleep 20 && docker compose logs celery_worker | grep 'model identity' | tail -1"
echo "done. Backup stamp: $STAMP  (rollback: SERVER_DIR=$D GO=1 bash tools/deploy_server.sh rollback $STAMP)"
