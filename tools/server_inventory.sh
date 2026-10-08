#!/usr/bin/env bash
# Plan S1: READ-ONLY inventory of the production server before the 8 Oct deploy. Changes nothing.
# Run from the dev PC:  bash tools/server_inventory.sh [ssh-host-alias]   (default: server-14)
# Prints names of .env settings, never their values (except a short allow-list of non-secret switches).
set -uo pipefail
H=${1:-server-14}
ssh -o BatchMode=yes -o ConnectTimeout=15 "$H" 'bash -s' <<'REMOTE'
echo "== host"; hostname; date; cat /etc/timezone 2>/dev/null; timedatectl 2>/dev/null | grep -i "time zone"
echo "== cpu/ram/disk"; nproc; grep -c ^processor /proc/cpuinfo; free -h | head -2; df -h / | tail -1
echo "== docker"; docker --version; docker compose version 2>/dev/null | head -1
echo "== containers"; docker ps -a --format '{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
echo "== compose projects"; docker compose ls 2>/dev/null
for d in $(docker inspect --format '{{index .Config.Labels "com.docker.compose.project.working_dir"}}' $(docker ps -aq) 2>/dev/null | sort -u); do
  echo "== project dir $d"; ls -la "$d" | head -40
  ( cd "$d" && git log -1 --format='commit %h %ad %s' --date=short 2>/dev/null; git status --short 2>/dev/null | head -20 )
  echo "-- .env keys (values hidden; switches shown)"
  [ -f "$d/.env" ] && sed -E 's/^([A-Za-z0-9_]+)=.*/\1/' "$d/.env" | grep -v '^#' | grep . | tr '\n' ' '; echo
  [ -f "$d/.env" ] && grep -E '^(LINE_ENABLED|V3_[A-Z_]+|NOTIFICATION_COOLDOWN|ESCALATION_[A-Z_]+|MAX_ESCALATIONS|CAMERA_MAX_PARALLEL|TZ)=' "$d/.env"
  echo "-- PUBLIC_BASE_URL host only"; grep -E '^PUBLIC_BASE_URL=' "$d/.env" 2>/dev/null | sed -E 's#^PUBLIC_BASE_URL=(https?://[^/]+).*#\1#'
  echo "-- models"; ls -la "$d/models" 2>/dev/null | head -30; (cd "$d/models" 2>/dev/null && sha256sum fall_classifier_v3.onnx yolo26s-pose.pt 2>/dev/null | cut -c1-12)
  echo "-- compose files"; ls "$d"/docker-compose*.yml 2>/dev/null
done
echo "== listening ports"; ss -ltnp 2>/dev/null | awk '{print $4}' | sort -u | tr '\n' ' '; echo
echo "== HTTPS front (nginx/caddy?)"; docker ps --format '{{.Names}} {{.Image}}' | grep -i -E "nginx|caddy|traefik|proxy" ; which nginx caddy 2>/dev/null
REMOTE
