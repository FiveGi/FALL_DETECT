# Gemini — R5/R6 independent test-coverage review (2026-09-30)

Written by Gemini (gemini-3.1-pro-high via Antigravity CLI, plan mode, read-only).
Recorded verbatim by Claude, because plan mode cannot write files. Evidence it read:
`.ai_evidence/r5r6_evidence.txt` in the workspace root (diffs, checker output, greps).

### 2026-09-30 — Gemini: R5/R6 independent test coverage review

Task ID / objective: R5/R6 test coverage review
Phase: ready_for_review (read-only findings, no code modifications made)
Checks: read `.ai_evidence/r5r6_evidence.txt`, `training/measure/tier_accuracy.py`, `training/measure/recovery_after_alert.py`, and `app/services/camera_manager.py`

**Findings on (1) R5 (Alert Track Assignment):**
- **Is the alert now credited to a flagged track?** Yes. Based on `app/services/camera_manager.py` lines 68-70 in the diff, `_flagged = [r for r in results if r[1]]` and `max(_flagged or results...)` ensure that a flagged person is selected if any exists, preventing the alert from being wrongly attributed to an unflagged bystander who merely happens to have a higher score.
- **Does ANY test exercise a two-person case where the flagged person is not the top scorer?** No. I read `training/measure/tier_accuracy.py` (lines 66-68) and `training/measure/recovery_after_alert.py` (lines 78-81). Neither script exercises a two-person tracking edge case. In fact, `tier_accuracy.py` still contains the old buggy logic `top = max(results, key=lambda r: r[2])` and ignores the new `_flagged` logic entirely. `recovery_after_alert.py` simply takes the very first flagged person it iterates over.

**Findings on (2) R6 (Still Down Frame-Rate Independence):**
- **Does `still_down_confirmed` remove the frame-rate dependence?** Yes. I reviewed the implementation in `app/services/notification_service.py` (lines 16-35) and the output of `tools/check_still_down_rule.py`. The boolean function compares `frames_down >= STILL_DOWN_FRACTION * frames_elapsed` purely as a ratio of frames, completely removing `target_fps` and `rate`.
- **Are `frames_at_alert` and `frame_count` on the same clock as `frames_since_upright`?** Yes. I verified in `app/services/camera_manager.py` (lines 841, 1008-1010, 1025) that `frame_count` is the core loop iteration counter. I am guessing that `frames_since_upright` advances strictly once per loop iteration inside the pose detector step.
- **What does the checker miss?** `tools/check_still_down_rule.py` is a stateless unit test. It completely misses the stateful event handling and lifecycle of the alert track inside `camera_manager.py`. For example, it misses the video loop reset handler (`awaiting_still_down.clear()` at `camera_manager.py` line 813). It also misses track expirations: if a person is lost from tracking, `person is None` triggers a pop of the alert (line 1034), so it goes unanswered instead of confirming, which the pure-math checker doesn't cover.

**Findings on (3) Gap Analysis & Next Steps:**
- **What is untested?** Both multi-person tracking attribution and the true end-to-end "still down" tier escalation (`STILL_DOWN_SECONDS`) remain untested by any test suite or measurement script. 
- **What single test closes the biggest gap?** As noted in `AI_HANDOFF.md`, `STILL_DOWN_SECONDS` is 10s and the longest test clip is 10s, meaning the window never actually closes in any test. The single biggest gap can be closed by an integration test that runs a video clip (or simulated loop) > 10 seconds of a confirmed fall to actually trigger the `confirmed` tier escalation branch inside the stateful loop.

Next actor / exact next action: Claude — please incorporate these findings into the queue, save them to `docs/reviews/gemini-r5-r6-review.md`, and append them to my section in `AI_HANDOFF.md`.
