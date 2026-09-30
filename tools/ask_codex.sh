#!/usr/bin/env bash
# Hand the current slice to Codex and wait for its answer. Scoped to this repository.
#
# The two assistants working on this project are separate applications and cannot message each
# other. The owner was copying text between them by hand. This closes that loop in one
# direction: Claude runs this, Codex does a turn in the same working tree, and Claude reads
# what it wrote. The other direction needs nothing -- Codex's reply is this script's output.
#
# The executable is DISCOVERED, not hardcoded. It ships inside the ChatGPT VS Code extension,
# whose directory name carries the extension version and changes when it updates. (It is also
# not on PATH, which is how a `command -v codex` check produced a confident "not installed".)
#
# Usage:
#   tools/ask_codex.sh "review the AUTHZ-1 entry in AI_HANDOFF.md"
#   SANDBOX=read-only tools/ask_codex.sh "..."     # let it look but not write
#
# Environment:
#   CODEX_BIN   explicit path, skips discovery
#   SANDBOX     read-only | workspace-write   (default: workspace-write)
#   MODEL       passed to --model if set
#   TIMEOUT     seconds, default 1800
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SANDBOX="${SANDBOX:-workspace-write}"
TIMEOUT="${TIMEOUT:-1800}"

find_codex() {
    if [ -n "${CODEX_BIN:-}" ]; then printf '%s\n' "$CODEX_BIN"; return; fi
    if command -v codex >/dev/null 2>&1; then command -v codex; return; fi
    # Newest extension directory wins, so an extension update does not break this.
    local hit
    hit=$(ls -dt "$HOME"/.vscode*/extensions/openai.chatgpt-*/bin/*/codex.exe 2>/dev/null | head -1)
    [ -n "$hit" ] && printf '%s\n' "$hit"
}

CODEX="$(find_codex)"
if [ -z "$CODEX" ] || [ ! -f "$CODEX" ]; then
    echo "ask_codex: could not find the codex executable." >&2
    echo "  Looked on PATH and in ~/.vscode*/extensions/openai.chatgpt-*/bin/*/codex.exe" >&2
    echo "  Set CODEX_BIN to the full path if it lives somewhere else." >&2
    exit 127
fi

# Being logged out produces a failure that reads like a model error, so it is checked up front.
if ! "$CODEX" login status >/dev/null 2>&1; then
    echo "ask_codex: codex is not logged in. Run: \"$CODEX\" login" >&2
    exit 126
fi

TASK="${*:-}"
if [ -z "$TASK" ]; then
    echo "usage: tools/ask_codex.sh <what Codex should do this turn>" >&2
    exit 2
fi

# The prompt carries the protocol rather than assuming Codex remembers it: `codex exec` starts a
# fresh session every time, so anything not said here is not in context. Pointing at the two
# files is deliberate -- they are the shared state, and re-deriving it would waste the tokens
# this arrangement exists to save.
read -r -d '' PROMPT <<EOF || true
You are Codex, working with Claude on this repository. You are not starting fresh work: read
AI_HANDOFF.md -- its "Current state" section is written to be the only context you need -- and
docs/AI_COLLABORATION.md for the agreed working rules.

Your task this turn:
$TASK

BUDGET, and it is not advice. This session gets one Codex run. Three previous runs consumed
about 156,000 tokens between them and exhausted the quota mid-task, and the single largest
cause was reading SKILL.md.

- **Do NOT read SKILL.md.** It is 290 KB; reading it once costs roughly 70,000 tokens, which is
  most of your budget. It is a historical narrative, not a reference. Measured figures live in
  tools/check_config_coherence.py (the MEASURED table) and app/services/detector_profiles.py.
- Do not read docs/reviews/2026-09-29-handoff-archive.md unless the task names it.
- Read the files the task names and their callers. Use targeted searches with bounded output,
  never a whole-file dump of anything over a few hundred lines.

Rules that matter more than finishing quickly, and that the budget does not excuse:
- Append your reply to your own section of AI_HANDOFF.md. Do not edit or delete Claude's text.
  Keep it short and link evidence instead of pasting it.
- Every claim carries its evidence: the command, the dataset and clip counts, the file and line.
  Say "I am guessing" where you are guessing. Do not report a check you did not run, and do not
  claim a check passed to finish inside the budget -- running out is an acceptable outcome and
  a false "verified" is not.
- If you disagree with something in Claude's entry, say so plainly and name the measurement
  that would settle it.
- End your handoff entry with the concise block from docs/AI_COLLABORATION.md, including
  "Next actor / exact next action".
EOF

MODEL_ARG=()
[ -n "${MODEL:-}" ] && MODEL_ARG=(--model "$MODEL")

echo "ask_codex: $("$CODEX" --version 2>/dev/null)  sandbox=$SANDBOX  timeout=${TIMEOUT}s"
echo "ask_codex: $CODEX"
echo "---"

timeout "$TIMEOUT" "$CODEX" exec \
    --sandbox "$SANDBOX" \
    --skip-git-repo-check \
    -C "$REPO" \
    "${MODEL_ARG[@]}" \
    "$PROMPT" < /dev/null  # never wait on a keyboard: a background run has none, and `codex exec` sat 20 min on "Reading additional input from stdin"
rc=$?

echo "---"
if [ $rc -eq 124 ]; then
    echo "ask_codex: TIMED OUT after ${TIMEOUT}s. Codex may have written a partial entry;" >&2
    echo "  check AI_HANDOFF.md before assuming nothing happened." >&2
elif [ $rc -ne 0 ]; then
    echo "ask_codex: codex exited $rc -- treat this as blocked, not as an accepted review." >&2
fi
exit $rc
