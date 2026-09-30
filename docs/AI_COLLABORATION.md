# Claude + Codex + Gemini: efficient work and independent review

Owner-requested working rules, recorded by Codex on 2026-09-29.
Extended to three assistants at the owner's request on 2026-09-30.
Apply to all three assistants. Reduce duplication, not evidence or necessary checks.
No fixed token cap may force incomplete work or a false claim of verification.

## Scope and roles

- Work only on `Backend-Elderly-Surveillance-main`, including its embedded `frontend/`, and
  its collaboration artifacts. Do not expand into sibling projects or machine-wide setup.
- Claude coordinates the queue and handles bulk implementation. Codex independently reviews
  high-risk changes and test evidence. Gemini, accessed through the owner's Antigravity CLI,
  handles bounded supporting investigations: test-case gaps, data/result consistency, or media
  triage once actual media support has been verified. Roles may switch for a named task.
- Do not have all three read or implement every task. Use the third assistant only for a
  distinct question, independent evidence, or an unresolved disagreement.
- A media/model opinion is a proposed label, not ground truth. Preserve timestamps and source
  identity; uncertain clips require human review. Do not send private media to a new external
  service unless the owner has authorized that material and use.
- Each task names its implementation owner, reviewer, allowed files and acceptance criteria.
  Only one writer owns a file at a time. Supporting reviewers read code and write separate
  evidence artifacts; they do not overwrite another assistant's work.
- Do not approve by majority vote. Resolve disagreements with source evidence and tests.
- Gemini's initial task is a read-only check of R5/R6 test coverage: follow the current handoff
  to the relevant tests and worker code, identify missing multi-person/low-FPS boundary cases,
  and report file/line evidence. This does not approve those fixes or replace Codex's review.

## Read only the context needed for this task

- Read these rules once per session; reread when they change.
- Start from the latest relevant handoff, the assigned issue and current working-tree state.
  Read the affected code and its callers, data contracts and tests as needed.
- Use targeted searches and bounded excerpts. Exclude dependencies, datasets and generated
  output unless the task concerns them. Do not dump the entire repository or full logs.
- Do not repeatedly load all of `SKILL.md`, all prior conversation or all measurement JSON.
  Historical context is available by reference. Preserve any applicable requirements to read
  project context before changing the detection pipeline; do not skip them to save tokens.
- Link evidence rather than copying it between files. Read full artifacts when the conclusion
  depends on details omitted by a summary. If context is insufficient, expand the inspection.

## Alternate ownership and review

- One bounded issue per work slice; record implementer, reviewer and affected files before edits.
  Existing uncommitted edits belong to ongoing work: inspect them and preserve them.
- The implementer changes code and performs relevant checks. The other assistant independently
  examines the diff, affected behavior and evidence; it does not approve from the summary alone.
- During review, report concrete defects with file/line, trigger, impact and a way to verify.
  Do not silently edit the implementer's files at the same time.
- If changes are needed, return them to the implementer. Review the resulting delta and affected
  checks rather than restarting the entire review without a reason.
- After acceptance, switch roles when useful; default to Claude implementing and Codex reviewing
  to conserve the owner's Codex quota. Give Gemini a distinct supporting slice when needed.
  An acceptance message should include the next handoff; avoid separate acknowledgement loops.

## Verification is required where behavior matters

- Auth, camera access, alerts, timing, tracking, model/data changes and deployment configuration
  require deeper review of negative cases, boundaries and failure behavior.
- Small text/style changes need proportionate verification, not a new full test suite.
- Prefer deterministic checks for repeatable facts. Record the command, outcome and evidence path.
  Reuse a prior result only when the relevant code, configuration, inputs and environment remain
  applicable; the reviewer must inspect enough evidence to establish this.
- Build success is not runtime correctness. Distinguish static inspection, isolated reproduction,
  mock tests, cache replay and live integration tests. State what was not tested.
- For accuracy/speed claims, include dataset/split, clip counts, model/config identity, measurement
  command and artifact. Separate simulated footage, cached results and real-camera measurements.
- Do not delete a failing test, weaken an assertion, omit a known failure or claim success merely
  to end a review round or reduce tokens. Investigate the discrepancy.

## Concise handoff format

Aim for 10–20 lines where practical; expand for important risks or complex evidence.

```text
Task ID / objective:
Implementer / reviewer / supporting investigator (if needed):
Phase: proposed | implementing | ready_for_review | changes_requested | accepted | blocked
Changed files / diff or commit reference:
Behavior before -> after:
Checks: command; outcome; evidence path
Known failures / untested cases:
Decision needed, if any:
Next actor / exact next action:
```

Keep detailed evidence in task reports and link it from `AI_HANDOFF.md`. Append only within
your own handoff section; preserve the other assistant's text. Do not relabel historical results
as current without verification. A finding is not a fix and a proposal is not an approved result.

## Stop unproductive loops and idle token use

- After two unsuccessful correction/review cycles on the same disagreement, do not repeat the
  same arguments. Specify the unresolved claim and a discriminating test. Run it when feasible;
  involve the owner only when a genuine requirement/decision/access issue remains.
- End a slice when required checks and independent review support acceptance. Broaden testing
  only for a material unresolved concern or new changes/failures.
- If automation is later installed, a normal controller should wait on process completion or
  state changes and invoke an assistant only when work is ready. Avoid AI turns just to poll files.
- Rate limits, errors and missing results mean waiting/blocked, never accepted. Preserve the last
  successful handoff and do not launch both writers for one task.

## Handoff and execution

Use the current-state section and latest relevant entries of `AI_HANDOFF.md`; historical
findings may already have been fixed. Do not treat the original audit as current task status.
Claude has added `tools/ask_codex.sh` to invoke a separate Codex run and collect its result.
Do not run another Codex review concurrently with that invocation for the same slice.
The owner has shown Antigravity CLI open; this alone does not verify headless invocation,
media support, or automatic three-party routing. No Gemini runner was added by this update.
Writing these rules does not send a message to an idle chat or guarantee it has read them.

## Three-party roles — agreed 2026-09-30 (Claude proposed; Codex and Gemini critiqued)

| | pays from | role |
|---|---|---|
| Claude | its own large session budget | coordinator; all implementation, experiments, measurement, live verification, owner updates |
| Codex | smallest quota, shared with the owner's own Codex window | independent review of **high-risk diffs only** (auth, alerts, timing, tracking, deploy config), one per finished slice |
| Gemini (Antigravity CLI) | owner's Google AI Pro quota | **on demand only**: images/frames/sheets, and methodology on results that are consequential or uncertain. Not a routine gate. |

- Every review request names: the exact diff or revision, the input/config identity (dataset,
  split, clip counts, profile), and the acceptance criterion.
- Evidence is staged in `<workspace>/.ai_evidence/`, but reviewers may also read targeted source
  (callers, failure paths) when staged evidence is not enough. Gemini runs headless in plan mode:
  it can read workspace files, but shell commands are auto-denied, so command output is staged.
- No duplicate reviews of one slice; no three-party gates on routine work.
- Owner gets 1–3 line updates per finished task; details on request.
- Decisions queued for the owner, never guessed: MJPEG auth design, admin password, commit,
  what "fall 69" means, the `out_of_domain` definition.

### Owner update 2026-09-30: conserve Claude's budget
Claude's session budget is running low. Shift work outward: Codex implements bounded slices
(workspace-write) and Claude reviews the diff; Gemini takes analysis, visual checks and
code reading (plan mode, reply returned to Claude). Claude keeps coordination, live tests,
and short owner updates. Verification rules above still apply in full.
