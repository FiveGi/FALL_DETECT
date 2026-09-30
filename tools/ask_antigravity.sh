#!/usr/bin/env bash
# Hand a task to Antigravity (Gemini, on the owner's Google AI Pro account) and wait for it.
#
# Third participant after Claude and Codex. Its role is narrower than theirs, and on purpose:
# it LOOKS -- contact sheets, frames, a second opinion on a diff -- and it does not write to the
# repository. Two writers already share this working tree with 24+ uncommitted files; a third
# autonomous editor would be a way to lose work, not a way to gain it. So it runs in
# `--mode plan` unless AGY_MODE says otherwise.
#
# Why Antigravity at all: the owner's Pro subscription is refused by Gemini Code Assist and by
# Gemini CLI ("no longer supported for individuals"); Antigravity CLI accepts it and has a
# non-interactive `--print` mode, which is what makes it drivable from here.
#
# Usage:
#   tools/ask_antigravity.sh "Review the latest Claude entry in AI_HANDOFF.md ..."
# Environment:
#   AGY_BIN    explicit path (default %LOCALAPPDATA%\agy\bin\agy.exe)
#   AGY_MODEL  default gemini-3.1-pro-high
#   AGY_MODE   default plan (read-only); accept-edits only when the owner has said so
#   TIMEOUT    seconds, default 900
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Run from the workspace root (one level up), not the repository. That is the folder the owner
# marked trusted, it holds GEMINI.md, and it is NOT a git repository -- so evidence can be put in
# $WS/.ai_evidence where Gemini may read it without a permission prompt (headless mode denies
# reads outside the working folder) and without it landing in anybody's diff. Allowing
# read_file everywhere instead would also allow .env, which holds live keys.
WS="$(cd "$REPO/.." && pwd)"
EVIDENCE_DIR="$WS/.ai_evidence"
AGY="${AGY_BIN:-${LOCALAPPDATA:-/c/Users/$USERNAME/AppData/Local}/agy/bin/agy.exe}"
MODEL="${AGY_MODEL:-gemini-3.1-pro-high}"
MODE="${AGY_MODE:-plan}"
TIMEOUT="${TIMEOUT:-900}"

if [ ! -f "$AGY" ]; then
    echo "ask_antigravity: agy not found at $AGY -- set AGY_BIN" >&2
    exit 127
fi
TASK="${*:-}"
if [ -z "$TASK" ]; then
    echo "usage: tools/ask_antigravity.sh <what Antigravity should do this turn>" >&2
    exit 2
fi

# Every run is a fresh session, so the protocol travels in the prompt.
read -r -d '' PROMPT <<EOF || true
You are Gemini (running in Antigravity CLI), the third participant on this repository with
Claude and Codex. Your working folder is the workspace root; the application is the folder
Backend-Elderly-Surveillance-main/ inside it. Read GEMINI.md (here, at the root) first. The
three of you coordinate through (paths from the workspace root):

- Backend-Elderly-Surveillance-main/docs/AI_COLLABORATION.md  -- the agreed working rules.
- Backend-Elderly-Surveillance-main/AI_HANDOFF.md             -- the live handoff. "Current state" is the context you need; the
                               latest entries are the conversation. Your section is "Gemini".

Your task this turn:
$TASK

Your role: bounded, independent evidence and review. Claude implements; do not start a
competing implementation and do not invoke another assistant. You do not modify application
files. When this run is in plan mode you cannot write at all -- put your full answer in your
reply and Claude will record it in your section, attributed to you.

HARD LIMIT: never run a shell command or terminal tool of any kind. This run is headless; a
command cannot be approved, and ONE denied command aborts your whole reply -- nothing reaches
Claude. Use only file reading. If you need search results, say what to search for and Claude
will provide them next time.

Budget: do NOT read SKILL.md (290 KB, a historical log; measured figures live in
tools/check_config_coherence.py and app/services/detector_profiles.py). Do not read
docs/reviews/*archive*. Read what the task names, with bounded output.

Evidence rules, which the budget does not excuse: every claim names its file and line, command,
or image. Say "I am guessing" when you are. Never report a check you did not run. If you could
not view an image, say so rather than describing what it probably shows.
EOF

echo "ask_antigravity: $("$AGY" --version 2>/dev/null)  model=$MODEL  mode=$MODE"
echo "---"
# AGY_EXTRA passes extra flags, e.g. AGY_EXTRA="--add-dir <evidence dir>" so Gemini can read
# evidence Claude gathered instead of running shell commands that would need approval.
# shellcheck disable=SC2086
cd "$WS" && timeout "$TIMEOUT" "$AGY" --mode "$MODE" --model "$MODEL" ${AGY_EXTRA:-} \
    --print-timeout "$((TIMEOUT - 30))s" -p "$PROMPT" < /dev/null
rc=$?
echo "---"
[ $rc -ne 0 ] && echo "ask_antigravity: exited $rc -- blocked, not accepted" >&2
exit $rc
