#!/usr/bin/env bash
# Before turning LINE on: see / neutralise fall alerts that escalation would otherwise re-send to a phone.
#   bash tools/line_backlog.sh check   -> how many unacknowledged fall alerts are still inside the escalation window
#   bash tools/line_backlog.sh clear   -> mark them as fully escalated (MAX_ESCALATIONS) so nothing old is re-sent;
#                                         the alerts themselves stay (history, dashboard), only their re-sends stop
# Run in the project folder on the machine where docker compose runs (prefix with sudo if docker needs it).
# Fails LOUDLY (exit 1, no number) if it cannot read the database or the running app's settings: an empty count
# that looks like "nothing pending" is exactly the answer that would let an old backlog reach a phone (Codex, 7 Oct).
set -euo pipefail
cd "$(dirname "$0")/.."
die() { echo "LINE BACKLOG CHECK FAILED: $1 -- do NOT switch LINE on; send this message to the team" >&2; exit 1; }
num() { [[ "$1" =~ ^[0-9]+$ ]] || die "$2 (got: '${1:0:120}')"; }

# The window and the limit the RUNNING escalation uses, read from the app itself (its .env + code defaults),
# not from this shell -- a host shell without those variables would silently check the wrong window.
cfg=$(docker compose exec -T celery_maintenance python -c \
  "from app.config import Config as C; print(C.ESCALATION_MAX_AGE_MINUTES, C.MAX_ESCALATIONS)" 2>&1 | tail -1) \
  || die "could not read the app settings from celery_maintenance"
read -r WIN MAX <<< "$cfg" || true
num "${WIN:-}" "escalation window not readable from the app"; num "${MAX:-}" "max escalations not readable from the app"

Q="detection_type LIKE '%fall%' AND acknowledged_at IS NULL AND sent_at > now() - interval '$WIN minutes' AND coalesce(escalation_count,0) < $MAX"
psql() { docker compose exec -T db psql -U postgres -d postgres -tA -v ON_ERROR_STOP=1 -c "$1"; }
case "${1:-check}" in
  check) pending=$(psql "SELECT count(*) FROM notification_history WHERE $Q;" 2>&1) || die "database query failed: $pending"
         num "$pending" "pending count is not a number"
         on=$(psql "SELECT count(*) FROM line_settings WHERE enabled;" 2>&1) || die "database query failed: $on"
         num "$on" "LINE-on count is not a number"
         echo "settings in use: escalation window $WIN min, max $MAX re-sends"
         echo "unacknowledged fall alerts that could still be re-sent: $pending"
         echo "LINE switched on for users: $on" ;;
  clear) n=$(psql "WITH u AS (UPDATE notification_history SET escalation_count = $MAX WHERE $Q RETURNING 1) SELECT count(*) FROM u;" 2>&1) \
           || die "database update failed: $n"
         num "$n" "update count is not a number"
         echo "neutralised: $n" ;;
  *) sed -n 2,6p "$0"; exit 1 ;;
esac
