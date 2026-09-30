# Codex review for Claude — 2026-09-29

Reviewed working tree based on `3b69769`, including the uncommitted profile UI/API and IR
cache changes. This is a review, not a deployment or an approval of the existing changes.
No application, model, deployment configuration, credentials, or existing measurement caches
were changed. New review artifacts and a ChatGPT entry in `AI_HANDOFF.md` are the deliverables.

## Scope and current architecture

- Canonical application: this repository; its `frontend/` is the Vue 3/Vite application
  named by the root README. A separate sibling frontend also exists in the workspace;
  this review and build concern the embedded frontend only.
- Flask/JWT/SQLAlchemy API; PostgreSQL; Redis; Celery camera worker; maintenance worker
  and beat for escalation. Notification dispatch currently goes through LINE code.
- `fall_v2` dispatches `process_v2_fall_detection`, which uses the **v3** pose pipeline:
  YOLO pose -> per-person tracking -> 15-frame temporal classifier -> smoothing -> alerts.
  The names v2/v3 therefore describe different interfaces, not necessarily different models.
- Runtime/training/ONNX agree on **15 frames x 85 features**, threshold **0.65**,
  smoothing **1 of 3**, partial-window minimum **4**. Compose pins CPU **320/8 fps**
  and GPU **960/20 fps**. These are configuration observations, not new speed measurements.
- The beginning of `SKILL.md` describes historical RF-DETR/MediaPipe deployment states.
  Later sections and current call sites supersede those statements. Do not treat its opening
  TL;DR as a description of what the present worker runs.

## Findings requiring changes

P1 = resolve before relying on this behavior in deployment; P2 = correctness/reproducibility
issue to address next. Locations refer to this reviewed working tree.

### R1 — P1: regular login tokens can change the detector profile

**Locations:** `app/routes/detector_info.py:123`, `app/routes/auth.py:76` and `:97`.

`apply_profile` rejects only `if role and 'admin' not in role`. Login and refresh create
access tokens with identity but no role claim, and no additional-claims callback was found.
An authenticated ordinary user's token consequently has an empty role and bypasses this check.
The frontend's admin-only panel does not protect an API endpoint.

**Evidence:** `codex_audit_probe.py` executes the actual route body with valid-token-shaped
claims lacking role and a mocked profile writer: HTTP **200**, writer called. This is a
function-level reproduction; it is not a request against the live deployment.

**Fix direction:** use the existing database-backed admin check after JWT verification;
deny missing/deleted/non-admin users. Test regular login and refreshed tokens, not only
manually constructed tokens with a role claim. Missing role must never grant privileges.

### R2 — P1: stream endpoints permit anonymous and invalid-token access

**Locations:** `app/routes/stream.py:15`, `:27`, `:37`, `:118`.

JWT verification is optional; every verification exception becomes `None`. The camera lookup
then falls back to unrestricted `Camera.query.get(camera_id)`. Thus omitting a token or supplying
an invalid one removes ownership filtering. This path serves video and also starts/stops the
web stream. Stopping a web stream is distinct from stopping the Celery detector.

**Evidence:** executing both helper bodies with a verifier that raises returns another user's
camera via the unrestricted lookup. The relevant routes have no mandatory JWT decorator.
Exposure is to clients that can reach the API; external tunnel/proxy reachability was not tested.

**Fix direction:** require identity and enforce owner/admin permission for control routes.
For browser media that cannot attach an Authorization header, implement an explicit scoped,
expiring media token or authenticated proxy; do not fall back to anonymous access.

### R3 — P1: saving a profile does not reliably apply the advertised settings

**Locations:** `app/routes/detector_info.py:145`, `app/services/detector_profiles.py:112`,
`docker-compose.yml:77`, `:82`, `docker-compose.gpu.yml:28`, `:52`.

The API writes `.env` and tells the operator to run `docker compose restart celery_worker backend`.
Existing container environment variables are not updated by a restart. `load_dotenv()` is called
without overriding those existing variables. Additionally, explicit Compose `environment`
values override `env_file`: input size and frame rate remain pinned by the CPU/GPU Compose file
even after recreation. Selecting the GPU profile cannot provision a GPU image/device by writing
`.env`, and selecting a CPU profile under the GPU overlay cannot override its 960/20 settings.

This is based on repository configuration and documented Compose behavior, not a production
restart experiment. Docker documents both [restart behavior](https://docs.docker.com/reference/cli/docker/compose/restart/)
and [environment precedence](https://docs.docker.com/compose/how-tos/environment-variables/envvars-precedence/).

**Fix direction:** separate hardware deployment from configurable detection policy, ensure the
effective Compose settings consume the intended values, recreate the appropriate services
using the correct overlay, and verify a newly reported worker configuration before showing
the change as applied. Keep restart/deployment an explicit operator action.

### R4 — P1: file playback and offline accuracy tests feed different temporal input

**Locations:** `app/services/camera_manager.py:760`, `:768`, `:772`, `:1048`;
`training/measure/cache_pose_streams.py:175`; `training/measure/rule_sweep_perclip.py:41`.

The worker waits for the target rate and reads the **next consecutive file frame**. It never
skips file frames to follow source time. The offline evaluator instead samples source frames
using `int(i * target_fps / source_fps)`. For a 30-fps, 300-frame video at an 8-fps target:

- offline: 80 classifier frames representing 10 seconds of motion;
- worker: 300 classifier frames, approximately 37.5 seconds of playback at target throughput.

The classifier sees denser motion samples and the user sees a delayed event. A displayed worker
rate near its target does not establish that the file was processed at its original time base.
The EOF branch also reuses tracking/classifier state across the jump back to frame zero.

**Evidence:** source control-flow review and deterministic sampling arithmetic in the probe.
This does not establish how an actual RTSP backend buffers or drops frames; that needs a separate
timestamp/latency test with the camera and selected OpenCV backend.

**Fix direction:** explicit source-time sampling for file sources, with a regression comparing
the frame indices/PTS consumed by worker and evaluator. Reset temporal state at EOF/reconnect
discontinuities where appropriate. Do not quote file playback fps as camera throughput.

### R5 — P1: an alert can be attached to the wrong person's track

**Location:** `app/services/camera_manager.py:827` and subsequent `awaiting_still_down` insertion.

The worker computes `any_detected`, but chooses `top` by maximum current probability across
**all** tracks. With 1-of-3 smoothing, an alerting person may have a lower current score than a
different person who is not alerting. The selected track then controls the later still-down
follow-up and the reported confidence.

**Evidence:** executing the actual `top` assignment on `(track=1, detected=True, score=.2)`
and `(track=2, detected=False, score=.6)` selects track 2. This input is compatible with
the detector retaining an earlier positive smoothing flag for track 1.

**Fix direction:** select from detected tracks whenever any track is detected; determine whether
simultaneous fall events need separate person associations rather than one camera-level event.
Test mixed histories, not just the highest-scoring person's initial alert.

### R6 — P1: below-target fps can incorrectly close a still-down follow-up

**Locations:** `app/services/camera_manager.py:967`, `:997`;
`app/detection/v3_fall_detection.py:887`.

`frames_since_upright` counts actual measured observations, but conversion to seconds uses
configured `target_fps` whenever it is nonzero. In a concrete arithmetic scenario with target
8 fps and actual 4 fps, a person continuously down for 10 seconds contributes 40 observations.
The worker calculates 5 seconds, compares it to 8, removes the follow-up, and logs that the
person was upright. Low throughput is not evidence of recovery. The original alert remains;
the lost behavior is still-down escalation/evidence.

**Fix direction:** track observation timestamps and evidence coverage per person. Distinguish
unknown/insufficient evidence from observed upright posture. Test sustained low fps, occlusion,
and temporary reconnects, not only a worker achieving 90%+ of target.

### R7 — P2: the API labels unmeasured configurations as measured

**Locations:** `app/routes/detector_info.py:76`, `tools/check_config_coherence.py:45`.

The measurement key contains input size, window, target fps, partial minimum and preprocessing,
but not threshold, classifier identity, pose model identity, smoothing or other decision rules.
Changing the threshold to 0.99 still returns the 0.65 measurement and `measured_known=true`.
The new 0.50/0.70 profiles make this inconsistency immediately user-visible. The profile strings
also duplicate measurements rather than being looked up from one validated manifest.

**Evidence:** the probe executes the actual GET route body with reported thresholds 0.65 and
0.99; both return exactly the same measurement and `measured_known=true`.

**Fix direction:** use a versioned measurement manifest keyed by all result-affecting settings
and model hashes, with dataset/split identity and provenance. Unknown combinations should be
reported as unmeasured rather than matched to an incomplete key.

### R8 — P2: two selectable profiles cannot match their reported running settings

**Locations:** `app/routes/detector_info.py:104`, `app/services/detector_profiles.py:43`, `:53`, `:92`.

Worker thresholds are formatted using `%g`, producing `0.7` and `0.5`. Profiles store `0.70`
and `0.50`, and matching uses exact string equality. The UI therefore reports no matching
profile even if either setting is actually running correctly.

**Evidence:** passing the worker-formatted 0.7 configuration to the actual `current_profile`
returns `None`.

**Fix direction:** typed numeric comparison or common canonical serialization; test every
profile through worker JSON -> API conversion -> profile match.

### R9 — P2: simulated-noise cache results depend on resume history

**Locations:** `training/measure/cache_pose_streams.py:107`, `:137`, `:196`, `:214`.

One global RNG is seeded per process, but existing clips are skipped before random numbers are
consumed. After interruption, the next uncached clip receives the start of the RNG stream rather
than the draw it would receive in an uninterrupted run. The seed is also absent from `cache_key`,
so changing `SIMULATE_DARK_SEED` reuses the same cache directory. Model paths, rather than weight
hashes, likewise do not distinguish an overwritten checkpoint.

**Evidence:** the deterministic probe shows a later clip has different noise when the preceding
cached clip is skipped, with the same seed. The current cache key has no seed entry.

**Fix direction:** local RNG per clip derived from explicit seed + stable clip identity; include
seed, simulation version, model/content hashes and preprocessing identity in the manifest.
Verify resumed vs fresh runs match byte-for-byte. Do not regenerate shared caches in place.

`compare_caches.py` additionally prints metadata but does not reject unexpected differences
between caches. The review replay explicitly checked that all four sets contain the same
100 URFD clip filenames; it does not retroactively establish their noise-generation history.

### R10 — P2: health status can report cameras running when no detection task exists

**Location:** `app/routes/health.py:43` and `:68`.

Per-camera running status and `all_running` derive only from database `is_active`. The Celery
check establishes that an inspect response exists, not that each expected camera task exists.
An empty active task list plus cameras marked active can therefore yield running status.
Other camera paths already use `is_really_running`; this endpoint has not adopted that check.

**Fix direction:** report desired state separately from observed worker/heartbeat state,
including unknown when the worker cannot be queried. Test a worker responding with zero tasks.

## Secondary follow-ups, not exercised end to end

- `detection_dispatch.hold_camera/release_camera` use GET then SET/DELETE rather than atomic
  ownership checks. A lease expiring between operations can overwrite/delete a new owner's
  lease. The lease is 30 seconds and reconnect handling can go longer without a refresh.
  Use atomic compare-and-renew/release and test expiration/reconnect interleavings.
- Detector config is published once at worker startup with a seven-day TTL. A worker can die
  while its configuration still appears current, or stay healthy longer than the TTL and
  disappear from the page. Treat it as a timestamped startup report, or renew with liveness.
- `_restrict_public_edge` prefers client `X-Forwarded-Host` over `Host`. Whether this can bypass
  the intended tunnel allowlist depends on upstream header sanitization. Verify the deployed
  proxy boundary before describing the public edge as isolated; no external probing was done.
- The deployment definition exposes DB, Redis and Flower ports on host interfaces. Actual
  reachability/firewall/access protection was not verified in this review.
- URFD has been used repeatedly to choose/confirm settings in the documented history. Wording
  that says it was never used for tuning is too strong. Half B repeatedly read for model
  selection is not an untouched final test set. Reserve a new deployment-representative set.
- `AI_HANDOFF.md` labels a 56-clip clean metric as URFD: `summarise_perclip.py` actually combines
  **40 URFD ADL + 16 validation ADL**. Separate those denominators. It also says image cleanup
  never runs on night frames, but the code applies it whenever luminance is below the gate;
  the supplied luminances include values below 70. The mechanism needs qualified wording.

## Independent replay of the existing IR caches

Command: `python docs/reviews/codex_cache_replay.py`.

This reruns the current production feature transform, ONNX classifier, per-person tracker and
alert state machine on existing cached keypoints. It bypasses only the constructor that would
load an unused YOLO model. CPU ONNX uses one thread. It is not a rerun of pose extraction,
not a speed benchmark, and not evidence that simulated IR matches a physical night camera.
No cache, training data, weights, or production settings were changed.

All four sets have identical clip names: **60 fall videos + 40 ADL videos**. Multiple camera
views of one event should not be treated as independent incidents. Runtime is threshold 0.65,
15-frame window, partial minimum 4, 1-of-3 smoothing. Source/model SHA-256, cache keys and
every per-clip decision are in `2026-09-29-cache-replay.json`.

| Existing cache | Directory | Fall videos alerted | ADL videos alerted |
|---|---|---:|---:|
| Daylight, auto cleanup | `58bd55e55f1d` | 45/60 | 5/40 |
| Greyscale only | `932fbee53d48` | 35/60 | 5/40 |
| Simulated IR, normal gate | `cb5439b03087` | 17/60 | 5/40 |
| Simulated IR, cleanup forced via gate 999 | `9c2963624342` | 19/60 | 2/40 |

The previously missing greyscale false-alarm result is thus **5/40** for this exact cache and
classifier. Forced cleanup recovers only two net fall videos and reduces false alarms on these
caches. It is not sufficient evidence to switch the production gate: R9 prevents assuming
noise is paired across interrupted builds, and no real IR validation or CPU cost was measured.
Use these as observations to investigate, not a deployment recommendation.

## Verification performed and limits

| Check | Result |
|---|---|
| `python tools/check_config_coherence.py` | Passed, including actual ONNX dimensions |
| `python tools/check_alert_rules.py` | Passed; scope is tier rules/wording, not worker evidence generation |
| `python docs/reviews/codex_audit_probe.py` | Reproduced isolated defects; JSON saved beside this report |
| `python docs/reviews/codex_cache_replay.py` | Completed all 400 cache/clip replays; per-clip artifact saved |
| `npm.cmd run build -- --outDir ../tmp/codex-review-build` in frontend | Passed: 108 modules, Vite 4.5.14 |
| `npm.cmd run lint` in frontend | Failed: 44 errors, zero warnings; no auto-fix run |

Lint includes unused variables, computed-property side effects in `MediaViewer.vue`, async
Promise executors and naming/empty-block issues. Build success does not establish browser or
API correctness. The two existing Python checks passing does not invalidate R1-R10: they do
not exercise those behaviors.

Local Python lacks `flask_jwt_extended`; probes use extracted function bodies and mocked
dependencies rather than claiming full JWT integration coverage. Docker engine access was
denied in this execution environment. No API/browser end-to-end test, live camera run, server
CPU benchmark, restart, deploy, outbound notification, or physical IR capture was performed.

## Suggested alternating handoff

1. **Claude reviews this report and the probes first**, accepting or disputing each finding
   with file/line evidence. Do not change detection thresholds based on the replay table.
2. First implementation slice: R1/R2 authorization. Claude implements; Codex checks negative
   authorization cases and browser stream integration using isolated test accounts.
3. Next slice: R3/R7/R8 profiles. Codex implements after scope is agreed; Claude verifies effective
   Compose settings, measured manifests and every profile round trip.
4. Next: R4/R5/R6 worker behavior, with timestamped synthetic multi-person sequences and file
   sampling parity. One implementation owner at a time; the other reviews/tests.
5. Repair R9 before new IR A/B experiments. Keep old caches as historical artifacts. Obtain
   actual camera footage and a fresh held-out evaluation before claiming nighttime accuracy.

The order above is a proposed work queue, not evidence that either assistant has started fixes.
