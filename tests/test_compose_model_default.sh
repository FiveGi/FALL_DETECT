#!/usr/bin/env bash
# The deployed fall model is chosen in docker-compose.yml (PRODUCTION by default, owner 8 Oct) and switched to option A by
# three EXPLICIT lines in .env. Checks what Compose actually resolves, for both
# services that load the detector, without starting anything:
#   1. no V3_* values given        -> production in backend AND celery_worker
#   2. the runbook's 3 option-A lines -> option A in both
# Run: bash tests/test_compose_model_default.sh   (needs the docker CLI; nothing is started). Shell V3_*/COMPOSE_*
# variables are removed for the check: Compose lets them override the env file, which would test the shell (Codex P3).
set -uo pipefail
cd "$(dirname "$0")/.."
fails=0
resolved() {   # $1 = env file; prints "service var=value" for the three model variables
  env -u V3_POSE_MODEL -u V3_CLASSIFIER -u V3_THRESHOLD -u COMPOSE_FILE -u COMPOSE_PROJECT_NAME docker compose --env-file "$1" config --format json 2>/dev/null | python -c "
import sys, json
c = json.load(sys.stdin)
for svc in ('backend', 'celery_worker'):
    e = c['services'][svc].get('environment', {})
    print(svc, ' '.join('%s=%s' % (k, e.get(k)) for k in ('V3_POSE_MODEL', 'V3_CLASSIFIER', 'V3_THRESHOLD')))"
}
expect() {  # $1 name, $2 env file, $3 expected "var=value ..." line
  out=$(resolved "$2")
  if [ "$(printf '%s\n' "$out" | grep -cF -- "$3")" -eq 2 ]; then echo "PASS $1"; else echo "FAIL $1 -- got: $out"; fails=$((fails+1)); fi
}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
: > "$T/empty.env"
printf 'V3_POSE_MODEL=pose_nightaug_s44.pt\nV3_CLASSIFIER=fall_classifier_t2full_s45.onnx\nV3_THRESHOLD=0.70\n' > "$T/option_a.env"
expect "default is production in both services" "$T/empty.env" \
  "V3_POSE_MODEL=yolo26s-pose.pt V3_CLASSIFIER=fall_classifier_v3.onnx V3_THRESHOLD=0.65"
expect "runbook option-A lines select A in both services" "$T/option_a.env" \
  "V3_POSE_MODEL=pose_nightaug_s44.pt V3_CLASSIFIER=fall_classifier_t2full_s45.onnx V3_THRESHOLD=0.70"
for f in models/pose_nightaug_s44.pt models/fall_classifier_t2full_s45.onnx models/yolo26s-pose.pt models/fall_classifier_v3.onnx; do
  [ -s "$f" ] && echo "PASS model file present: $f" || { echo "FAIL missing $f"; fails=$((fails+1)); }
done
echo "RESULT $([ $fails -eq 0 ] && echo PASS || echo FAIL) ($fails)"; exit $fails
