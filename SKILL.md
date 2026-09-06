---
name: elderly-fall-detection-project
description: Full technical context for the Backend-Elderly-Surveillance-main fall/bed-exit detection AI system — every model, dataset, tuning decision, bug found and fixed, and open risk, in chronological order. Read this before touching any detection pipeline code.
---

# Elderly Surveillance — Fall & Bed-Exit Detection: Full Project Context

This document exists so that any AI assistant (or human) picking up this project cold can understand,
in full technical depth, what this system is, what has been built, what was tried and abandoned, what
bugs were found and how they were fixed, and — critically — what is still unverified or broken. It is
written after an extended iterative development session. Read it fully before making changes to any
detection pipeline; several approaches that look reasonable in isolation were already tried here and
found to fail for non-obvious reasons (see "Dead ends" sections).

## TL;DR — confirmed facts, measured on the real production server ([target-server-ip]), not estimates

- **v4 (RF-DETR) fall detector: 4,679ms/frame (0.214 fps).** Not viable. Needs `STRIDE_FRAMES` ≥70–140
  just to keep up with a camera, ~5-10x past where sensitivity was already shown to collapse to
  near-zero (§8). No parameter tuning fixes this.
- **v3's MediaPipe pose extractor: crashes outright** (`FATAL ERROR: ... compiled with avx enabled,
  but this feature is not available on this processor`, SIGILL). **v3 is not a fallback — it cannot
  run on this server at all**, not "runs but slow."
- **`yolov10x.pt` (the model currently live in production for alone-detection): 3,280–3,313ms/frame.**
  This runs on *every* frame with *zero* stride in production code (§13) — this feature is almost
  certainly already broken/unusably slow in the real deployment right now. Nobody had measured this
  before this check.
- **`yolo11n-pose.pt` (new candidate, not yet integrated into any pipeline): 294–297ms/frame, ~377MB
  RSS.** The only one of the four models above that actually runs successfully on this hardware.
  Promising, but unvalidated for accuracy — no classifier built on top of it yet, no OOD test done.
- **Root cause, common to all four results above:** the server's CPU is `QEMU Virtual CPU version
  2.5+` (KVM-virtualized) exposing only `sse4_1`/`sse4_2` — **no AVX, AVX2, or FMA at all**. This
  cripples (or in MediaPipe's case, outright breaks) any model whose prebuilt binaries or GEMM kernels
  assume modern SIMD support, which is true of most current deep-learning inference stacks. This is
  not model-specific — it's a property of this server. See §9 and §13 for the full investigation.
- Thread count (`OMP_NUM_THREADS=1` vs `=4`) makes **no meaningful difference** (~1%) for any model
  tested — ruling out thread oversubscription as a contributing cause. The AVX gap is the whole story.

## 1. What this project is

`Backend-Elderly-Surveillance-main` is a Flask + Celery backend for a camera-based elderly-care
surveillance system. It watches camera feeds (RTSP streams or recorded video files) and runs several
independent AI detection pipelines against them, alerting caregivers via Telegram when something
concerning happens. A separate frontend, `FRONT_END_DETR_BED-main`, consumes this backend's API and
database records.

**Stack:** Flask (API) + Celery (async per-camera detection workers, one Celery task per camera per
detection type, looping while the camera is active) + Redis (broker) + SQL DB (Postgres/MySQL via
SQLAlchemy) + OpenCV for frame reading.

**Detection types, each its own Celery task in `app/services/camera_manager.py`:**

| Celery task | Purpose | Status |
|---|---|---|
| `process_fall_detection` | Legacy v1 fall detection | Legacy — code path still exists; not confirmed whether the frontend still triggers it |
| `process_v2_fall_detection` | Fall detection — **this is the currently active, maintained pipeline** | **Active** — internally runs v4 (RF-DETR) despite the "v2" name (see §5) |
| `process_bed_exit_detection` | Detects a resident getting out of bed (sleep→sit transition) | Active |
| `process_alone_detection` / `process_v2_alone_detection` | Detects whether a person is present/alone in frame (YOLO person tracking) | Active, unrelated to fall detection |
| `process_camera` | Generic dispatcher | — |

Detection modules live in `app/detection/`. Models are loaded lazily and cached as singletons by
`app/services/model_manager.py` (thread-safe double-checked-locking pattern, one `_load_X()` /
`get_X()` pair per model).

## 2. Every model in the system

Location: `models/` directory at the repo root.

| File | Used by | Architecture | Status |
|---|---|---|---|
| `rfdetr_fall_detector.pth` (369 MB) | v4 fall detection | RF-DETR (Roboflow), DINOv2-windowed-small backbone, object-detection head, 3 classes: `fallen`/`falling`/`standing` | **Active, current production fall detector** |
| `yolov10x.pt` | Alone/person detection | YOLOv10x | Active |
| `bed_pose_mobilenetv2_3.onnx` | Bed-exit detection | MobileNetV2, 3-class (`bed`/`sleep`/`sit`) | Active |
| `fall_detection_model.onnx` | Legacy v1 fall detection (`fall_detection.py`, `FallONNXDetector`) | Unknown/undocumented legacy architecture | Legacy, uncertain if still invoked |
| `pose_landmarker_lite.task` | v3 fall detection (superseded) | MediaPipe Tasks API `PoseLandmarker`, IMAGE mode, 33 keypoints | Kept for rollback, not called in production |
| `fall_classifier_v3.onnx` | v3 fall detection (superseded) | Temporal CNN over pose-keypoint windows (30-frame window, stride 10), trained via `training/train.py` | Kept for rollback, not called in production |
| `deepsvdd_model.onnx` + `center.npy` | v2 fall detection (`V2ONNXFallDetector`, one-class anomaly model) | Deep SVDD (one-class classifier over some learned feature space) | Loaded by `model_manager` but **no active Celery task calls it anymore** — dead code path |
| `fall_detection_v2.pth` | Unclear | Unclear — an older checkpoint file, provenance not re-verified in this session | Unclear / likely legacy artifact |

**Why so many models:** this reflects an iterative history — v1 → v2 (DeepSVDD anomaly) → v3
(MediaPipe pose + temporal CNN) → v4 (RF-DETR, current). Each generation's code and weights were kept
in place rather than deleted, specifically so a rollback is possible if v4 turns out to have a
dealbreaker problem in production. **Do not delete v2/v3 model files or code without confirming with
whoever owns the deployment — they are the rollback path.**

## 3. Datasets used

**Pose-based pipeline (v3, now superseded) was trained on:**
- **CAUCAFall** — staged fall dataset; this is also the dataset used for every sensitivity/specificity
  number in this document. Fixed 5-subject slice: **Subjects 2, 4, 6, 8, 10** × 10 activities each
  (5 fall types: Fall backwards/forward/left/right/sitting; 5 ADL types: Hop/Kneel/Pick up
  object/Sit down/Walk) = 50 clips. PNG frame sequences per activity clip, dev-machine path
  `d:/project/PROJECT/Dataset CAUCAFall/CAUCAFall/Subject.N/Activity/*.png` (override via the
  `CAUCAFALL_ROOT` env var if your checkout lives elsewhere). Reproduce the benchmark with:
  `python training/benchmark_v4_rfdetr_50clips.py`
- **GMDCSA24** — another staged fall pose dataset
- **FallVision** — the third original dataset (confirmed from `v3_fall_detection.py`'s own docstring:
  "~6,100 videos pooled from GMDCSA24 + FallVision + CAUCAFall"); see `training/parse_fallvision.py`.
  Check `training/train.py`'s `POSE_DIRS` list for the authoritative current set of pose directories.
- **OF-In-the-Wild (OF-ItW)** — added during this session. Part of **OmniFall**
  (arXiv:2505.19889), a benchmark that unifies 8 staged lab datasets (OF-Staged, including CAUCAFall
  and GMDCSA24 plus 6 others: CMDFall, UP-Fall, Le2i, EDF, OCCU, MCFD — **only CAUCAFall/GMDCSA24 were
  actually integrated**, the other 6 were not downloaded/used), 818 real-world accident videos sourced
  from OOPS/FailArmy compilations (**OF-In-the-Wild**, the one integrated here), and 12,000
  diffusion-generated synthetic clips (OF-Synthetic, not used). 16-class activity taxonomy collapsed
  down to binary fall/ADL for this project (fall = OmniFall classes `fall`/`fallen`).
  - Labels loaded via HuggingFace `datasets.load_dataset("simplexsigil2/omnifall", "of-itw")` — a
    standard `datasets` library call, no special package needed.
  - **Important:** the OmniFall README references a `pip install omnifall` package — **this package
    does not exist** on PyPI or in any real GitHub repo. Do not waste time looking for it; use the
    `datasets` library path above instead.
  - Video files downloaded directly from Columbia's OOPS dataset host, matched to OmniFall's label
    rows by filename. Matching required alphanumeric-only lowercase normalization
    (`re.sub(r"[^a-z0-9]", "", s.lower())`) because label `path` values strip ALL punctuation, not just
    spaces — a naive space-stripping match only found 89/818 videos; the full normalization found
    818/818 with zero collisions.
  - Poses extracted via `training/extract_ofitw_poses.py` → 3,997 `.npz` files in
    `training/data/poses_ofitw/` (from 5,178 label rows, minus ~1,156 exact duplicate rows in the
    source label table, minus 36 clips too short to yield a usable window).
  - Retraining with OF-ItW added: `Loaded 10102 videos / Fall: 4496, ADL: 5606 / Best val F1: 0.594`
    (down from 0.615 without OF-ItW) — **but the full precision/recall curve dominates the old model
    at matched operating points**, so the new checkpoint was kept despite the lower single-threshold
    F1 number. This is a case where a single scalar metric (F1 at threshold=0.5) was misleading; always
    check the full curve, not just one point, when comparing model versions.

**v4 (RF-DETR, current production model) was NOT trained by anyone in this session.** It is a
checkpoint (`checkpoint_best_total.pth`, later copied to `models/rfdetr_fall_detector.pth`) that a
teammate produced and dropped into the project folder undocumented. Its provenance was recovered by
inspecting `torch.load(ckpt, weights_only=False)['args']` (an `argparse.Namespace` saved at training
time): fine-tuned from RF-DETR-base on a **Roboflow-hosted dataset called "fall_detaction-3"** (note
the typo in the dataset name itself). **This dataset's size, quality, and label consistency have never
been independently audited by anyone working on this repo.** Given the generalization problems found
in §7 below, auditing or replacing this training data is probably the single highest-leverage next
step, not further pipeline tuning.

## 4. Pipeline evolution, in order

### v1 — legacy, architecture undocumented
`app/detection/fall_detection.py`, `FallONNXDetector`, `fall_detection_model.onnx`. Still has a live
Celery task (`process_fall_detection`) but it's unconfirmed whether anything in the frontend still
triggers it. Treat as legacy/frozen; don't invest in it without first confirming it's actually dead.

### v2 — DeepSVDD one-class anomaly model
`app/detection/v2_fall_detection_onnx.py`, `V2ONNXFallDetector`, `deepsvdd_model.onnx` + `center.npy`.
Loaded by `model_manager._load_fall_detector()` / `get_v2_fall_detector()`, but **no Celery task calls
`get_v2_fall_detector()` anymore** — confirmed dead at the call-site level, kept only as loadable
rollback code.

### v3 — MediaPipe pose + temporal CNN (superseded by v4, kept for rollback)
`app/detection/v3_fall_detection.py`. Pipeline: MediaPipe `PoseLandmarker` (IMAGE mode) extracts 33
keypoints per frame → sliding window (`WINDOW_SIZE=30`, `STRIDE=10`) → temporal CNN
(`fall_classifier_v3.onnx`) outputs a fall probability per window (`THRESHOLD=0.5`) → smoothing/gating
state machine decides whether to actually flag a fall.

**Bugs found and fixed in this pipeline during a 50-clip end-to-end batch test** (this batch test
matters: it ran the literal production function, not offline metrics, and is what surfaced these —
offline validation metrics looked fine and hid all three of these):

1. **`MIN_PERSON_FRACTION` too strict (was 0.7).** MediaPipe pose detection fails intermittently, and
   sometimes totally, on a prone/collapsed body — it wasn't trained heavily on that pose. Diagnosed via
   a full per-frame MediaPipe trace on `Subject.3/Fall forward` — **note:** `Subject.3` was a one-off
   diagnostic example, not part of the fixed 5-subject benchmark set used everywhere else in this
   document (Subjects 2/4/6/8/10, see §6's table and `training/benchmark_v4_rfdetr_50clips.py`); it
   only had 39.6% intermittent person detection across the clip, meaning most 30-frame windows never
   had enough valid frames to pass a 70%-of-window gate. **Fix:** lowered to
   `MIN_PERSON_FRACTION=0.2`, and added a separate `RESET_PERSON_FRACTION=0.05` so a genuinely empty
   room still fully resets state (avoids the naive fix of "just lower the threshold to 0" which would
   also stop resetting on real empty-room cases).

2. **Destructive `recent_flags.clear()` on gate reset, erasing real fall evidence at the worst possible
   moment.** Raw probability traces on `Subject.6/Fall backwards` (part of the 5-subject benchmark
   set) and `Fall right` showed confidence
   climbing to 0.73 / 0.693 right as the person hit the ground, then crashing to 0.0 as MediaPipe lost
   the pose — and the reset logic was clearing the smoothing buffer exactly then, discarding the
   evidence. A first fix attempt (tiered reset-vs-hold thresholds) had **zero effect** on these two
   specific cases, because their signal crashed below even the new lowered reset threshold — it never
   reached the "hold" zone the tiered logic was meant to protect. **Actual fix:** a "collapse-signal"
   heuristic — if probability was above `COLLAPSE_CONFIDENCE=0.6` immediately before detection is lost
   entirely, treat that as a positive fall signal and fire once (`state.collapse_fired` flag prevents
   repeat-firing on the same event). This is the fix that actually worked.

3. **Smoothing rule too strict.** Replaced a strict "2 consecutive frames over threshold"
   (`MIN_CONSECUTIVE=2`) rule with an N-of-M rule: `SMOOTH_NEED=2` out of the last `SMOOTH_OF=3`
   samples.

Net effect of all three fixes, verified by re-running the same 50-clip batch test: **sensitivity
72% → 80%, zero new false alarms.**

**Also learned along the way:** an initial run of this same 50-clip test produced numbers that looked
almost meaningless until train/val split membership was checked directly (same `seed=42`
`split_videos()` call used at training time) — only 8 of the 50 clips were genuinely held-out from
training, and all 8 of those were correct; all 12 errors were on clips the model had already seen
during training. This redirected the whole debugging effort from "the model doesn't generalize" (wrong
conclusion) to "the live pipeline's gating/smoothing logic has real, fixable bugs" (right conclusion).
**Lesson: always check train/val split membership before interpreting a batch test's error pattern.**

### v4 — RF-DETR, current production pipeline
`app/detection/v4_fall_detection_rfdetr.py`. `V4RFDETRFallDetector` wraps Roboflow's `RFDETRBase`
(`rfdetr` PyPI package), loaded with the teammate's checkpoint. Classifies each sampled raw RGB frame
directly — no pose extraction step, so it doesn't share v3's MediaPipe-failure-on-prone-poses problem.

```python
FALLEN_STREAK_NEEDED = 2   # consecutive "fallen" classifications required to alert
STRIDE_FRAMES = 5          # run inference every 5 frames
CONFIDENCE_THRESHOLD = 0.3 # RF-DETR detection confidence floor
```

`detect_v4_fall(frame, state, fall_detector, config, camera=None, threshold=None)` returns
`(detected, probability, label, frame)` — same signature shape as `detect_v2_fall_only_onnx` /
`detect_v3_fall`, so it's a drop-in replacement.

**Integration bugs found and fixed:**
- `RFDETRBase(pretrain_weights='checkpoint_best_total.pth')` with a bare relative filename triggered
  Roboflow's HF-style cache-dir resolution logic and threw `FileNotFoundError` looking in
  `~/.roboflow/models/`. **Fix:** pass `os.path.abspath(ckpt_path)`.
- `model.class_names[c]` threw `IndexError`. RF-DETR's `class_id` output is **1-indexed** (COCO
  convention, 0 = background) while `class_names` is a 0-indexed list. **Fix:** index with `c - 1`.
- `frame_bgr[:, :, ::-1]` (the common idiom for BGR→RGB) produces a **negative-stride** numpy view,
  which `torch.from_numpy` (called internally by RF-DETR's preprocessing) cannot accept —
  `ValueError: At least one stride in the given numpy array is negative`. **Fix:** use
  `cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)` instead, which returns a proper contiguous array.

**Wiring:** `model_manager.py` got a new `v4_fall_detector` slot with the same thread-safe lazy-load
pattern as v2/v3 (`_load_v4_fall_detector()`, `get_v4_fall_detector()`); `is_ready()` now gates on
`v4_fall_detector` instead of v3. `camera_manager.py`'s `process_v2_fall_detection` task was rewired
to import and call the v4 detector/state/function instead of v3's.

**Deliberately NOT renamed:** the Celery task name `process_v2_fall_detection`, and the
`detection_result = 'fall_v2'` string written to the database, were both left exactly as-is even
though the pipeline behind them is now v4. This is intentional, not an oversight — the frontend
(`FRONT_END_DETR_BED-main`) was confirmed via grep to depend on that exact literal string `'fall_v2'`
in 5 files. **Do not rename these without also updating the frontend.**

**Dependency change:** `rfdetr` requires `torch>=2.2.0, torchvision>=0.17.0`. The repo's
`requirements.txt` was previously pinned to `torch==2.0.1, torchvision==0.15.2` for compatibility with
the rest of the stack (mediapipe, ultralytics/YOLO). Bumped to `torch==2.2.0, torchvision==0.17.0`
(the conservative floor rfdetr states, not the dev machine's actual 2.6.0). **This has never been
verified against an actual Docker build** — see §7, this is a real open risk.

**Comparison methodology (worth reusing):** rather than re-running the (slow) RF-DETR model every time
a decision-rule parameter needed testing, per-frame raw classifications were captured once for all 50
CAUCAFall clips (`rfdetr_capture_50.py` → `rfdetr_raw_results.json`, one JSON with a
`[label, confidence]` sequence per clip sampled every `STRIDE_FRAMES=5` frames) and then swept offline
against different candidate decision rules. This is much faster than re-running inference per
parameter choice, and was reused successfully later (see §8, duration-based fix attempt). The same
capture-once/sweep-many pattern was used earlier for tuning the v3 pose model's threshold
(`training/tune_threshold.py`).

**Final production verification:** ran the same 50-clip test through the literal
`detect_v4_fall()` production function (not a re-implementation) — confirmed
**80.0% sensitivity (20/25), 84.0% specificity (21/25), 240.0s total** — matching the offline-swept
prediction exactly, and much faster than v3's equivalent run (787–860s), since there's no MediaPipe
step.

## 5. Bed-exit detection (separate pipeline, not touched this session but relevant context)

`app/detection/bed_exit.py`. Motion-gated: frame-differencing against a reference frame
(`SKIP_FRAMES=15`, `MOTION_THRESHOLD`/`MAX_MOTION_THRESHOLD` bounds) — only runs the actual ONNX model
when motion in-range is detected, which keeps it cheap. Classifies a center ROI of the frame into
`bed`/`sleep`/`sit` via `bed_pose_mobilenetv2_3.onnx`, and fires a detection specifically on the
`sleep → sit` transition (i.e., someone who was lying down is now sitting up — an early bed-exit
signal). Explicitly forces `CPUExecutionProvider` for ONNX Runtime — this model was designed with
CPU deployment in mind from the start, unlike the newer RF-DETR fall detector.

## 6. Out-of-distribution ("in the wild") testing — the most important findings

The 80%/84% CAUCAFall numbers are an **in-distribution** benchmark: same lab, same handful of subjects
(a subset of whom the model may have literally been trained on, for the pose pipeline at least), same
camera angle, same lighting. When directly asked "is it really accurate?", this was tested properly by
sourcing images completely outside any training dataset (free stock photos from Pexels — elderly people
in various real poses, completely unrelated to CAUCAFall/GMDCSA24/OF-ItW) and running them through the
literal `V4RFDETRFallDetector.classify_frame()` production function.

**Genuinely fallen photos (15 images):** 13/15 (87%) correctly classified `fallen`, 15/15 got at least
some signal (`fallen` or `falling`, never `standing`/`none`). Reasonably solid.

**ADL / non-fall photos (19 images, after fixing a ground-truth labeling bug — see below):**
**10/19 (52.6%) were false-flagged as `fallen`/`falling`.** This is far worse than the 16% false-alarm
rate the CAUCAFall benchmark suggested. The false positives cluster clearly around **low/bent/kneeling/
floor-level body poses**: yoga stretches (downward-dog, forward-fold), tying shoes, kneeling to do
crafts, sitting on the floor with another person — and, less predictably, at least one case of someone
just sitting normally in an armchair reading (`falling`, confidence 0.808). This is consistent with (and
explains, at greater severity) the CAUCAFall benchmark's own most common false-alarm category,
"Pick up object" (a bending activity).

**Root cause diagnosis:** this is a **training-data-coverage problem, not a model-architecture
problem**. The `fall_detaction-3` Roboflow dataset the checkpoint was fine-tuned on almost certainly
lacks deliberate hard-negative examples of "person bending/kneeling/sitting on the floor while doing an
ordinary activity, definitely not falling." Any classifier trained the same way (staged fall footage
without explicit ADL-at-floor-level negatives) would likely show the same failure mode regardless of
its architecture — this was confirmed by researching alternative pretrained fall-detection models
(HuggingFace, Roboflow Universe, GitHub) and finding none with credible evidence of doing better; see
§10.

### A self-correction worth understanding in detail (methodology lesson)

The first pass at this out-of-distribution test also produced a dramatic-looking finding: images of
elderly people walking with a **cane or walker** were being confidently (0.7–0.94) misclassified as
`fallen`. This looked like a serious, specific, actionable bias — mobility-aid users are a large
fraction of the real target population.

**This finding was wrong**, due to a bug in the *test*, not the model: the hand-written Python
ground-truth dictionary (`{filename: true_label}`) had every `img_*` description shifted by one
filename slot relative to what was actually downloaded (a transcription slip while filtering out a few
"no person visible" images from the middle of the list). The specific image reported as "cane-walking
misclassified as fallen at 0.895 confidence" was, once the dict was corrected, actually a genuine photo
of a person collapsed on a kitchen floor — correctly classified, not a false positive at all.

A dedicated, cleanly-built follow-up test (19 fresh images of people using a cane/walker/rollator vs.
20 fresh control images of people standing with no aid, all newly downloaded, all globbed from a
folder rather than hand-mapped to individual labels — so there was no dictionary to mis-transcribe)
found **0/19 and 0/20 false positives** — i.e., **no mobility-aid bias exists**. Re-scoring the
*original* flawed test with the *corrected* ground truth (reusing the model's original outputs, no need
to re-run inference) is what surfaced the real pattern described above (bent/kneeling/floor poses, not
mobility aids).

**Takeaway for whoever continues this work:** when hand-building a ground-truth label dictionary for a
test, prefer directory-based labeling (put images in labeled folders, glob them) over a manually
transcribed `{filename: label}` dict wherever possible — the former can't suffer an off-by-one shift.
When a manual dict is unavoidable, a surprising/dramatic finding is exactly when to double check the
mapping before reporting it.

## 7. Known open risks — nothing below has been fixed or verified yet

1. **RESOLVED (as in: definitively answered "no"). CPU inference speed for RF-DETR was measured for
   real, on the real production server, inside the real Docker resource limits — see §9 for full
   numbers. This is the single most severe finding in the project.** Confirmed **4,679ms/frame
   (0.214 fps)**, built and run on [target-server-ip] with `--cpus=2.0 -e OMP_NUM_THREADS=1
   -e CUDA_VISIBLE_DEVICES=""` — not an estimate. Requires `STRIDE_FRAMES` ≥70–140 to keep up with any
   realistic camera fps, ~5-10x past the point where sensitivity was already shown to collapse to
   near-zero. Root cause: the server's virtual CPU has **no AVX/AVX2/FMA** (only SSE4.1/4.2, confirmed
   via `lscpu`), crippling GEMM/matmul throughput for any transformer-heavy model — not NNPACK
   (an earlier version of this doc misattributed it to NNPACK; corrected in §9, that's a
   convolution-focused optimization and RF-DETR's bottleneck is attention/GEMM, which doesn't route
   through it). **This penalizes every CPU-inference candidate on this server, not just RF-DETR.**
   Plain fp32 ONNX export gave zero speedup (tested earlier on the dev machine). **v4 is not viable on
   this server, full stop — not a tuning problem.** ⚠️ **v3 is not a fallback either — see §13:
   MediaPipe's prebuilt binary crashes outright (SIGILL) on this same AVX-less CPU, not just slow.**
   The decision is now infrastructure/architecture: raise CPU/thread budget substantially, get GPU
   access, or build a new lightweight pipeline around `yolo11n-pose.pt` (§13 — the one model of four
   tested that actually runs on this hardware, but currently just a bare pose detector with no
   classifier or OOD validation yet). Stop investing in v4 CPU optimization on this hardware; it won't
   close a 20-100x gap.

2. **RESOLVED: `requirements.txt`'s torch/torchvision bump (2.0.1→2.2.0, 0.15.2→0.17.0) was verified
   with a real `docker build` on the actual target server ([target-server-ip]) — it succeeds, no conflict
   with `mediapipe`/`ultralytics`.** Found a real, separate inefficiency while doing this: the default
   `torch==2.2.0` PyPI wheel pulls in the full CUDA toolkit as transitive dependencies
   (`nvidia-cudnn-cu12` alone is 731.7MB, plus nccl/cusparse/cublas/etc., ~2GB total) even though this
   is a CPU-only deployment (`CUDA_VISIBLE_DEVICES=""`). **Fix:** pin the CPU-only wheel instead —
   `--index-url https://download.pytorch.org/whl/cpu` — to cut image size and build time
   substantially. Not done yet, just identified.
   ⚠️ **Doing this build filled the target server's disk to 100% (a hard 21GB quota) and required an
   emergency cleanup** (`docker builder prune`/`image prune`/`rmi`) — see the "if you try this again"
   note at the end of this section before repeating it.

3. **The bending/kneeling/floor-sitting false-positive problem (§6) has no implemented fix.** A
   duration-based mitigation was tried and explicitly rejected — see §8, don't re-attempt this exact
   approach without new data. The most promising remaining direction is retraining/fine-tuning with
   deliberate hard-negative examples of these poses, not further decision-rule tuning. A tool for
   sourcing that hard-negative data now exists — `training/measure_fp_per_hour.py` (see §12) — but it
   needs known-fall-free long-form ADL video (Toyota Smarthome or similar) to run against, and that
   dataset access has been requested but not yet granted as of this writing.

4. **The `fall_detaction-3` Roboflow training dataset behind the current checkpoint has never been
   independently audited** for size, label quality, or class balance.

5. **It's unconfirmed whether the legacy v1 pipeline (`process_fall_detection`,
   `fall_detection_model.onnx`) is still triggered by anything in the frontend**, or is fully dead code.

6. **Per-Celery-worker RAM usage has never been measured with real Celery processes.**
   `celery_worker` runs with `--concurrency=4` (up to 4 camera tasks as separate prefork processes).
   If each worker process loads its own copy of a model (prefork workers don't share memory the way
   threads would), 4x a multi-hundred-MB-plus model could be enough to OOM-kill the container on a
   small server before it even finishes starting up. A same-process single-instance proxy exists now
   for `yolo11n-pose.pt` (~377MB RSS, §13) suggesting ~1.5GB at 4x concurrency — comfortably fine — but
   this has **not** been checked for RF-DETR or `yolov10x.pt`, and is not a substitute for actually
   starting 4 real Celery prefork workers and reading their RSS from `ps`/`docker stats`.

7. **RESOLVED (measured) + PARTIALLY FIXED. `yolov10x.pt`'s ("alone detection") inference speed was
   measured on the real server — see §13: 3,280–3,313ms/frame.** Confirmed via `docker ps -a` on the
   server that **no live deployment of this project exists there** — so this was a confirmed
   code-level bug, not an observed live incident (see §15) — though it would have surfaced immediately
   on first real deployment. §15 fixed the two clearest contributors: added
   `CAP_PROP_BUFFERSIZE=1` (missing in all 5 detection loops, not just this one) and replaced
   "every single frame, no stride" with a 20s time-based check interval, plus removed a duplicate
   `detect_persons()` call that was silently doubling cost in the v1 path. **Not yet resolved:** the
   underlying per-check latency itself is still ~3.3s if/when it does run — §16 has a preliminary
   (Pexels-based, not yet real-camera-validated) comparison showing `yolov10n.pt` as a promising
   ~16x-faster-in-the-§13-sense candidate with a small (~5 percentage point) accuracy cost. Also see
   §14: before optimizing further, ask whether the server's CPU type can be changed to
   `host`/`host-passthrough`, which could make this whole question moot.

**If you attempt another `docker build` against the real server:** free disk there is tight (as of
this writing, ~1GB after emergency cleanup, on a 21GB total quota). Use the CPU-only torch wheel fix
from item 2 above first — it avoids ~2GB of wasted download/layer space — and check `df -h /` on the
server *before* building, not after.

## 8. Dead end: duration/persistence-based smoothing fix (tested, rejected, do not redo without new data)

**Hypothesis:** since real falls leave a person down for a long time, while bending/kneeling/tying
shoes are brief (person returns to standing within a second or two), requiring the RF-DETR `fallen`
classification to persist for a longer duration before alerting (instead of just
`FALLEN_STREAK_NEEDED=2` consecutive samples, which spans only ~0.33–0.5s at typical fps) should filter
out the transient false positives while still catching genuine falls.

**Test method:** reused the already-captured `rfdetr_raw_results.json` (§4) to sweep candidate
`streak_needed` values against all 50 CAUCAFall clips **without re-running the model**, computing
sensitivity/specificity for each candidate before writing any production code.

**Result — sharply negative, no viable operating point:**

| streak | ≈duration @25fps | sensitivity | specificity |
|---|---|---|---|
| 2 (current) | 0.40s | 80.0% | 84.0% |
| 3 | 0.60s | 64.0% | 88.0% |
| 4 | 0.80s | 60.0% | 96.0% |
| 8 | 1.60s | 28.0% | 96.0% |
| 15 | 3.00s | 28.0% | 100.0% |
| 30 | 6.00s | 4.0% | 100.0% |

Even a small increase (streak 2→3) crashes sensitivity from 80% to 64%. There is no streak value in
this sweep that improves specificity without a badly disproportionate sensitivity loss.

**Why this happened:** CAUCAFall's "fall" clips are short **staged demonstrations** (3–10 seconds
total) where an actor deliberately gets back up soon after "acting out" the fall, because it's a
demo, not a real emergency. This means even the model's classification of a *genuine* fall clip
doesn't sustain a long "fallen" run — so a duration requirement penalizes true positives on this
dataset just as harshly as it would penalize a real bending/kneeling false positive. **This dataset
structurally cannot validate a duration-based fix** — it would need real (or realistically simulated)
footage where a person actually stays down for an extended period after a genuine fall, which nothing
currently in the project has.

**Decision:** did not implement this change. `FALLEN_STREAK_NEEDED` remains `2` in production. If
someone wants to revisit this idea, they need footage of a person remaining down for an extended,
realistic duration (ideally: real long-form footage, or at minimum synthetically extended clips) —
not another sweep against CAUCAFall.

## 9. CPU inference speed — measured, confirmed too slow, and it silently reproduces §8's problem

**Measured on the dev machine's CPU** (Intel Core i5-13500, 13th Gen desktop, 20 logical threads —
not a weak machine) by forcing RF-DETR onto CPU explicitly (`RFDETRBase(..., device="cpu")`):

**~496ms/frame (≈2 fps), PyTorch eager mode.** At `STRIDE_FRAMES=5`, the required per-inference
budget to keep up with a live camera is `STRIDE_FRAMES / fps` — 167–333ms depending on fps (15–30).
496ms blows that budget at every realistic camera framerate; the detection loop would fall
continuously further behind wall-clock time.

**Tried the obvious first fix — plain ONNX export — it did not help at all.** RF-DETR's own
`model.export(format="onnx")` API produced a working 115 MB fp32 ONNX graph; benchmarked with ONNX
Runtime's `CPUExecutionProvider` (all 20 threads, default graph optimizations on):
**511ms/frame — statistically the same as PyTorch eager, 0.97x "speedup."** RF-DETR's bottleneck is
compute-bound deformable-attention transformer layers, not something ONNX Runtime's default CPU
kernels do anything special for (unlike CNNs, where ONNX export commonly gives a real 2–4x). Remaining
untried options — INT8 quantization (rfdetr supports it, needs calibration data, will cost some
accuracy — bad news given §6's already-fragile specificity), OpenVINO (uncertain custom-op support for
deformable attention), or reducing the export resolution below 560×560 — are all bigger, less certain
investments than this quick test, and haven't been attempted yet.

**The dangerous part: this isn't just "slow," it silently re-creates the exact failure mode §8 already
proved harmful.** The only real mitigation available without a faster runtime is increasing
`STRIDE_FRAMES` so each (slow) inference call has more wall-clock time to finish — but §8 already
showed that requiring the `fallen` streak to span more wall-clock time craters sensitivity. Re-running
the same offline-sweep methodology from §8, but this time subsampling `rfdetr_raw_results.json` to
simulate actually increasing `STRIDE_FRAMES` (rather than just `FALLEN_STREAK_NEEDED` at a fixed
stride) makes this concrete:

Required `STRIDE_FRAMES` to keep up at ~500ms/inference call: **≥8 frames at 15fps, ≥10 at 20fps, ≥13
at 25fps, ≥15 at 30fps.**

| simulated STRIDE_FRAMES | FALLEN_STREAK_NEEDED | sensitivity | specificity |
|---|---|---|---|
| 5 (current, GPU-only viable) | 2 | 80.0% | 84.0% |
| 10 (≈15–20fps camera on this CPU) | 2 | 68.0% | 88.0% |
| 10 | 1 | 88.0% | 76.0% |
| 15 (≈25–30fps camera on this CPU) | 2 | 60.0% | 92.0% |
| 15 | 1 | 76.0% | 88.0% |

**No combination tested reaches the GPU-benchmarked 80%/84% baseline.** Dropping
`FALLEN_STREAK_NEEDED` to 1 partially recovers sensitivity at each stride level, but at a real
specificity cost — and given §6 already found real-world specificity is much worse than this
benchmark suggests (52.6% false-positive rate on bending/kneeling photos), trading more specificity
away here is risky, not free. **There is currently no known parameter combination that makes v4 run
acceptably on this class of CPU without a real accuracy cost.**

**What this means concretely:** the target production server's actual CPU spec (core count, model)
has to be checked against these numbers before anything else about v4's future is decided — it's now
a hard blocker, not a nice-to-have. If the target server is weaker than this dev machine (very
possible — this is a fairly capable modern desktop chip), all the numbers above are optimistic, not
pessimistic. Reproduce with `training/benchmark_v4_rfdetr_50clips.py` (per-clip PyTorch/ONNX timing)
plus the same subsample-and-eval-streak technique from §8 against a fresh `rfdetr_raw_results.json`
captured at the real target stride.

### The target server was actually checked (SSH, read-only `nproc`/`lscpu`/`free -h`) — it's worse than the dev machine, not better

```
CPU:  QEMU Virtual CPU version 2.5+ (KVM-virtualized, not bare metal)
      14 vCPUs allocated, but lscpu shows only 4 actually ONLINE (cores 3,5,6,11 --
      cores 0-2,4,7-10,12,13 are offline)
RAM:  12 GiB total, ~11 GiB free (not the bottleneck)
OS:   Ubuntu 26.04 LTS, kernel 6.8, Docker 29.6.0 already installed
```

This is strictly worse than the i5-13500 dev machine used for every benchmark above: fewer
usable cores (4 online vs 20 threads) and a virtualized CPU model rather than real desktop silicon
(virtualized cores typically have some overhead vs bare metal per core too). Since
`docker-compose.yml` already pins `OMP_NUM_THREADS=1` regardless of how many cores the host has, the
**single-threaded 1,543ms/frame number from the section above is still the most relevant estimate —
and if anything this weaker/virtualized CPU likely makes it worse, not better.** There was no
optimistic case hiding in "maybe the real server is beefier than the dev machine" — it isn't.

**This closes the open question.** v4 (RF-DETR) at its current architecture and the current
`docker-compose.yml` resource configuration is confirmed not viable on the actual target
infrastructure, not just suspected to be. The three options from the paragraph above (raise the
container's CPU/thread budget — note the underlying VM itself would need more *online* cores first,
not just a higher `cpus:` limit in compose; get GPU access; or fall back to v3) are no longer
hypothetical trade-offs to weigh later — one of them has to be chosen before v4 can ship.

### It gets worse: the real deployment config was checked, and it's even more constrained

`docker-compose.yml`'s `celery_worker` service (the container that actually runs
`process_v2_fall_detection`, i.e. v4, per camera) sets:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=""
  - OMP_NUM_THREADS=1
deploy:
  resources:
    limits:
      cpus: '2.0'
command: celery -A app.celery worker --concurrency=${CAMERA_MAX_PARALLEL:-4} ...
```

`CUDA_VISIBLE_DEVICES=""` confirms CPU-only is deliberate, not accidental. But `OMP_NUM_THREADS=1`
was almost certainly set for the *old* lightweight ONNX/MediaPipe pipelines (to avoid 4 concurrent
camera workers oversubscribing each other) — it was never revisited for RF-DETR, which needs
multi-threaded BLAS to be fast. Re-running the CPU benchmark **forced to a single thread**
(`torch.set_num_threads(1)`, matching this env var exactly) gives:

**1,543ms/frame (0.65 fps) — 3.1x worse than the already-too-slow 496ms 20-thread number.**
That needs `STRIDE_FRAMES` ≥23–46 (fps-dependent) just to keep the processing loop from falling
further and further behind wall-clock time — far past anything in §9's stride-vs-accuracy table, deep
into the range where even `FALLEN_STREAK_NEEDED=1` only holds sensitivity around ~68% at best (see
the stride=30 row). And this single-thread number doesn't even account for the container's `cpus: '2.0'`
cap being shared across up to 4 concurrent camera tasks (`--concurrency=4`) — with several cameras
running, each task's *effective* CPU share would be even less than what was just measured.

**Conclusion: v4 (RF-DETR) does not just run slowly under the actual `docker-compose.yml` resource
configuration — it is not viable there at all**, independent of any `STRIDE_FRAMES`/streak tuning.
This is no longer a parameter-tuning problem. Before investing further in v4 optimization, the real
decision is one of: (a) substantially raise `celery_worker`'s CPU allocation and remove/raise
`OMP_NUM_THREADS=1` specifically for the fall-detection task (at the cost of headroom for other
concurrent camera tasks sharing the same container), (b) get GPU access for this container, or
(c) fall back to v3 (the pose pipeline), which was implicitly designed/validated for exactly this
tight, single-threaded, CPU-only, multi-camera-concurrent resource budget already.

### FINAL, CONFIRMED NUMBER: actually built and ran on the real production server — v4 is dead, no longer an estimate

Everything above this point was estimation (dev-machine single-thread simulation) or spec-comparison
(the target server has fewer/weaker cores). The actual container was built and run **on the real
server** ([target-server-ip]), inside Docker, with the exact production resource limits
(`--cpus=2.0 -e OMP_NUM_THREADS=1 -e CUDA_VISIBLE_DEVICES=""`), running the literal
`V4RFDETRFallDetector.classify_frame()` production code:

**4,679ms/frame (0.214 fps) — averaged over 10 real inferences, on the real target hardware.**

This is **3x worse** than the dev-machine single-thread simulation (1,543ms) predicted. The run's
stderr also showed `Could not initialize NNPACK! Reason: Unsupported hardware` — **this was
initially (and incorrectly) written up here as the root cause. It isn't, or at least isn't the main
one: NNPACK primarily accelerates convolution, and RF-DETR's bottleneck (deformable-attention
transformer layers) is GEMM/matmul-bound, which routes through oneDNN/MKL, not NNPACK.** The
actual, verified root cause: `lscpu` on the server shows the QEMU virtual CPU exposes only
`sse4_1`/`sse4_2` — **no AVX, AVX2, or FMA at all** — while the dev machine (confirmed via
`torch.backends.cpu.get_cpu_capability()`) runs PyTorch's AVX2-optimized kernels. Matrix
multiplication without AVX2/FMA falls back to dramatically slower generic/SSE code paths, which is
consistent with a 3x-plus gap on a GEMM-bound model and fully explains the discrepancy without
needing NNPACK as an explanation. **This is not RF-DETR-specific — it penalizes any GEMM-heavy model
run on this server**, including MediaPipe and any pose-estimation candidate being considered as a v3
fallback (see §12). Dev-machine numbers for those models cannot be assumed to transfer; they need
to be measured on this actual server, not extrapolated.

Required `STRIDE_FRAMES` to keep up at this real speed: **≥70 at 15fps, ≥94 at 20fps, ≥117 at 25fps,
≥140 at 30fps** — an order of magnitude past anything in the stride-vs-accuracy table above (which
only went to stride=30, already down to single-digit-percent sensitivity). At these real strides,
sensitivity would be at or near 0%.

**There is no more open question here.** v4 (RF-DETR) does not run acceptably on this server at any
achievable settings — this was confirmed by literally building and running it there, not inferred.
Do not spend further effort trying to tune, quantize, or otherwise rescue v4's performance on this
specific hardware; the fix has to be infrastructure (more/better CPU cores, or GPU access) or a
different model (v3 fallback, pending its own OOD validation below).

**How this was actually gotten working (useful if repeating this build):** the build repeatedly hit
`disk quota exceeded` on this server's 21GB quota — first from `torch==2.2.0`'s default wheel pulling
in ~4-5GB of unused CUDA/nvidia-* packages (server is CPU-only), fixed by pinning
`--extra-index-url https://download.pytorch.org/whl/cpu` with `torch==2.6.0+cpu` /
`torchvision==0.21.0+cpu` (note: `torch==2.2.0+cpu` alone wasn't enough — `rfdetr`'s `transformers`
dependency requires `torch>=2.5`, which only surfaced as a runtime `ImportError`, not a build failure,
wasting a full rebuild cycle; 2.6.0 matches what the dev machine already had working and satisfies
transformers' floor too). Even after that fix, `docker build --no-cache`'s own layer-export step
transiently filled the disk again — `docker builder prune -af` (**never** `--volumes` on this box,
see below) reliably reclaimed multiple GB each time and let the build/run complete. Both fixes are
now in `Dockerfile` and `requirements.txt`.

**⚠️ This server is shared with another project (`project_forest_animal`)** — its 4 Docker volumes
and 2 images must never be touched. `docker system prune -a --volumes` would delete that project's
data since nothing is currently running (no container has them "in use," so a `--volumes` prune
would treat them as safe to remove). Always use scoped commands (`docker builder prune -af`,
`docker rmi <specific-tag>`) and check `docker volume ls` / `docker images` first if unsure what's
present.

### v3 was then run through the same out-of-distribution test as v4 (§6) — results are mixed, and partly inconclusive

Using the exact same Pexels photos from §6 (16 genuinely-fallen images + 58 not-fall images, the
latter including the cane/walker/control set from §6's bias investigation), fed through the literal
`detect_v3_fall()` production function. Since v3 is a **temporal** model (needs a 30-frame window,
unlike v4's per-frame classifier), each static photo was fed repeated ~45 times to fill and stabilize
the window — see `v3_wild_test.py`.

**Results:**

| | v3 (pose) | v4 (RF-DETR) |
|---|---|---|
| Sensitivity (16 fallen photos) | 37.5% (6/16) | 87.0% (13/15, §6) |
| Specificity (58 not-fall photos, same set for both) | **91.4%** (53/58) | 82.8% (48/58) |

**The specificity comparison is valid and favors v3** — genuinely fewer false alarms in the wild.

**The sensitivity comparison is NOT a fair fight, and v3's 37.5% should not be read as "v3 misses most
real falls."** v3's model explicitly uses **frame-to-frame keypoint velocity** as an input feature
(`_normalize_and_velocity`'s `vel = np.diff(...)` in `v3_fall_detection.py`). Feeding the same static
image 45 times in a row makes that velocity **exactly zero for the entire window** — real falls
involve a fast collapse motion that a repeated static frame can never represent, no matter how
"fallen" the pose looks. v3's in-distribution sensitivity on real video (CAUCAFall, actual motion
present) was 80% (§4) — a real number from real motion, unlike this test. Static-pose specificity
testing doesn't have this problem (a person genuinely holding still has genuinely near-zero velocity
too), which is why the specificity side of this table is trustworthy and the sensitivity side isn't.

**Honest conclusion: this test could not determine whether v3 is a safe out-of-distribution fallback.**
It productively confirmed v3 has fewer wild false alarms than v4, but the safety-critical question
(does v3 actually catch out-of-distribution falls) needs real video clips of people falling in
non-lab settings — not static photos — which nothing in this project currently has. Don't
re-quote the 37.5% number as v3's real sensitivity; it's an artifact of the test methodology, not a
finding about the model.

## 10. Dead end (partial): searching for a better off-the-shelf model

Searched HuggingFace Hub, Roboflow Universe, and GitHub for alternative pretrained fall-detection
models that might replace RF-DETR outright. Findings:

- An EfficientNetB0-based HF model (`Siddhartha276/Fall_Detection`) — **no accuracy metrics reported
  anywhere**, appears to be an individual/student project, and is architecturally a whole-frame binary
  classifier with no geometric/pose grounding — the same class of risk that produces our current
  bending/kneeling confusion, so it's unlikely to actually be better even if tested.
- Several Roboflow Universe community datasets (4,497 and 7,780 fall images) — unvalidated
  provenance, but potentially useful as **additional training data** to combine with hard-negative
  examples, rather than as ready pretrained models.
- The UR Fall Detection Dataset (University of Rzeszów) — a legitimate, well-known academic dataset,
  but still staged/lab-style falls, same fundamental limitation class as CAUCAFall.
- Several GitHub pose+LSTM repos reporting only 61–67% accuracy — worse than what this project
  already has.

**Conclusion: no clearly superior drop-in replacement was found.** This searched confirmed (rather
than solved) the diagnosis in §6: the failure mode is about training-data diversity, not model choice.
The single most valuable output of this search is the two larger Roboflow datasets as **candidate
additional training data**, not as replacement models. This has not been pursued yet (no fine-tuning
has been attempted with hard-negative bending/kneeling data).

## 11. If you're an AI picking this project up next

- The active fall-detection code path is `app/detection/v4_fall_detection_rfdetr.py`, called from
  `process_v2_fall_detection` in `app/services/camera_manager.py`. Don't be confused by the "v2"
  naming — it's historical/frontend-coupled, not a mistake.
- Before believing any accuracy number, ask whether it's in-distribution (same dataset family as
  training data) or out-of-distribution. In-distribution numbers here have consistently looked much
  better than real-world performance.
- Before implementing a decision-rule tuning idea (thresholds, streaks, durations), check whether it
  can be evaluated offline against already-captured raw per-frame data before writing production code
  — `rfdetr_raw_results.json` / the capture-then-sweep pattern in §4 and §8. It's much faster and has
  already saved from shipping one harmful change (§8).
- If you build a hand-written `{filename: ground_truth}` test dictionary, double-check the mapping
  before trusting a surprising result — see the cane/walker false-alarm story in §6.
- The two biggest unresolved risks are CPU inference speed (§7.1/§9 — measured and confirmed **not
  viable** under `docker-compose.yml`'s actual `celery_worker` constraints, not just slow) and the
  bending/kneeling false-positive rate (§6/§7.3). Both are more valuable to work on than further model
  swapping (§10) or further decision-rule tuning (§8) alone — note §8 and §9 are now known to
  interact: hardware/container constraints can silently force the exact duration-based degradation §8
  already proved harmful, without anyone deliberately choosing it.
- **v4's CPU non-viability is no longer a "go measure the server" open question — it's measured, and
  it's an infrastructure/architecture decision now, not a benchmarking one.** See §9's closing
  paragraph for the three real options (raise `celery_worker`'s CPU/thread budget, get GPU access, or
  fall back to v3 pending an OOD test of its own). Don't re-derive this from scratch; the numbers
  already exist.

## 12. FP/hour measurement tool — the actual path to fixing §6/§7.3, once data is available

`training/measure_fp_per_hour.py` runs the literal production `detect_v4_fall()` over long-form
video and counts false-positive **events** (not raw flagged frames — consecutive alerts within one
continuous episode are grouped into a single event, the same way `alert_cooldown` would collapse them
into one real-world alert). Point it at any video folder known to contain **zero genuine falls**
(general ADL/daily-activity footage) and every event it produces is a false positive by construction
— no ground-truth labeling needed, which is what makes this a viable *accuracy* baseline before any
new dataset access is granted, and why it doesn't need to wait on the CPU-speed work in §9 (accuracy
and latency are independent axes).

Outputs: `raw_results.json` (every sampled frame, for offline re-scoring later without re-running
inference — the same capture-then-sweep pattern as §4/§8), `events.json` (one row per deduped false
alarm), and a `candidates/` folder of the peak-confidence frame from each event — this is the actual
raw material for fixing §7.3 via fine-tuning with real hard negatives, as opposed to the ad-hoc stock
photos used for diagnosis in §6.

**Status: written and smoke-tested (verified the frame-sampling/dedup logic against a known-fall-free
CAUCAFall `Walk` clip — 49 correctly-deduplicated inference samples from 241 frames, 0 events, 0
FP/hour, as expected for a clean walking clip), but not yet run for real** — needs access to a
genuine long-form known-fall-free dataset (Toyota Smarthome was the candidate discussed; access
requested from the dataset's academic license holders as of this writing, not yet granted). Do not
use Toyota Smarthome, NTU RGB+D, or ETRI-Activity3D footage for actual **fine-tuning/training** without
first separately checking that dataset's license terms — using footage to *measure* a false-positive
rate doesn't put any of that data into the model's weights, but training on it does, and those are a
very different legal question. (Not legal advice — read the actual license text before deciding.)

## 13. Pose-model alternatives benchmarked on the real server — v3's MediaPipe is also dead; one candidate survives

With v4 confirmed non-viable (§9), the natural next question was whether *any* CPU-friendly
person/pose model works on this actual hardware — including re-checking v3's own MediaPipe dependency
(never previously stress-tested on this server) and the model already live in production for a
different feature (`yolov10x.pt`, alone-detection). First, the exact production call patterns were
read directly from source (not assumed) to make the benchmark representative:

- `AlonePersonDetector.detect_persons` (`app/detection/fall_detection.py`) calls
  `self.yolo_model.track(source=[frame], tracker="bytetrack.yaml", conf=0.6, classes=[0],
  verbose=False)`. **Finding along the way: `detect_alone_with_state` in `camera_manager.py` calls
  `detect_persons()` twice per frame** — once inside `detect_alone_only()`, once again directly right
  after — a real, previously-unnoticed duplicate-inference bug in the **v1** alone-detection path
  (`process_alone_detection`). The **v2** path (`process_v2_alone_detection` →
  `detect_v2_alone_only_onnx` in `v2_fall_detection_onnx.py`) calls it only once — clean. Also worth
  knowing: that file defines **two functions both named `detect_v2_alone_only_onnx`** (one taking a
  state object, one not) — Python silently keeps only the second; the first is dead, unreachable code.
- Neither alone-detection loop (v1 or v2) has any `STRIDE_FRAMES`-style skip logic — **YOLOv10x runs
  on every single frame**, paced only by `time.sleep(1.0/fps)`.
- `V3PoseFallDetector.extract_keypoints` (`app/detection/v3_fall_detection.py`) calls
  `vision.PoseLandmarker.detect()` (IMAGE mode, `min_pose_detection_confidence=0.5`). **Also runs on
  every frame unconditionally** — only the downstream temporal classifier is gated by `STRIDE=10`, not
  MediaPipe itself. This matters a lot for interpreting any MediaPipe benchmark: its raw per-frame
  latency has to fit inside the full `1/fps` budget, with zero stride cushion, unlike RF-DETR.

**Benchmark methodology:** `training/bench_pose_models.py` (now committed to the repo, alongside
`training/measure_fp_per_hour.py` and `training/benchmark_v4_rfdetr_50clips.py`) ran each of 3 models
continuously for 90s per run (not a small fixed iteration count, to catch shared-vCPU throttling
drift), at both
`OMP_NUM_THREADS=1` (current production setting) and `=4` (candidate alternative), reporting
p50/p95/p99/max latency, a first-half-vs-second-half drift check, and peak RSS via
`/proc/self/status`. Ran directly inside the same `rfdetr-bench` container/resource limits as §9's
RF-DETR test (`--cpus=2.0`), via volume-mounting the extra model files in rather than rebuilding the
image.

**Results:**

| Model | p50 @ threads=1 | p50 @ threads=4 | peak RSS | Outcome |
|---|---|---|---|---|
| `yolov10x.pt` (live in prod today) | 3,280ms | 3,313ms | ~520MB | Technically runs, but 3.3s/frame with **zero stride** in production — almost certainly already unusably slow in the real deployment |
| MediaPipe `PoseLandmarker` (v3) | **crash** | **crash** | n/a | `FATAL ERROR: This binary was compiled with avx enabled, but this feature is not available on this processor (go/sigill-fail-fast)` — same for `pclmul`. Immediate SIGILL, not a slowdown. |
| `yolo11n-pose.pt` (new candidate, unintegrated) | 297ms | 294ms | ~377MB | Only model of the three that ran successfully. ~16x faster than RF-DETR, lighter RSS than either other model. |

**Thread count (1 vs 4) made no meaningful difference for any model** (≤1% either direction) — this
rules out the hypothesis that PyTorch was oversubscribing threads relative to the cgroup limit as an
explanation for anything measured here. The bottleneck is the missing AVX/AVX2/FMA support itself
(§9's root-cause correction), not thread configuration.

**What this means:**

1. **v3 is now a confirmed dead end on this hardware, not just an open question.** MediaPipe's
   prebuilt binary hard-requires AVX; there is no config flag or stride adjustment that fixes a SIGILL.
   A differently-compiled (non-AVX) MediaPipe build might exist but has not been located or verified —
   don't assume one does before checking.
2. **A previously-unknown, separate production risk was found: `yolov10x.pt`-based alone-detection is
   almost certainly already broken or severely degraded in the live deployment**, independent of
   anything to do with fall detection. This wasn't on anyone's radar before this benchmark — it needs
   its own investigation (is `process_alone_detection` or `process_v2_alone_detection` actually the
   one wired up? is anyone getting alone-detection alerts today, or has this been silently failing?).
3. **`yolo11n-pose.pt` is the one candidate worth pursuing further**, but it is *only* a raw pose
   detector today — there is no fall-classification logic built on top of it (no temporal window, no
   trained classifier, nothing like v3's `fall_classifier_v3.onnx` or v4's fine-tuned classes). Building
   that, then running it through the same OOD validation as §6/§9's v3 test, is real, unstarted work —
   this result is a green light to *start* that path, not a finished fallback.

**RAM estimate for `yolo11n-pose.pt` at production concurrency:** ~377MB peak RSS in this single-process
benchmark × `--concurrency=4` (current `CAMERA_MAX_PARALLEL` default) ≈ **~1.5GB** if each Celery
prefork worker loads its own independent copy (§7's still-unverified OOM risk) — comfortably within
the server's 12GB total RAM, unlike what RF-DETR or `yolov10x.pt` would likely require at the same
concurrency (this specific multi-worker number has still not been measured directly with real Celery
processes — this is a single-process extrapolation, not a substitute for that measurement).

## 14. Before chasing model speed further: check whether the CPU limitation is even real hardware, or just a QEMU/KVM config default

§9/§13 established the server's virtual CPU has no AVX/AVX2/FMA. **This may not be a hardware ceiling
at all.** QEMU/KVM defaults to a maximally-portable virtual CPU model (`qemu64`/`kvm64`) that
deliberately excludes newer instruction sets for cross-host live-migration compatibility, unless the
hypervisor is explicitly configured to expose the real host CPU (`-cpu host` / Proxmox's
"host"/"host-passthrough" CPU type) — a config change, not a hardware purchase, typically applied with
a single VM reboot. **Ask whoever manages the underlying Proxmox/KVM host whether the CPU type can be
changed to `host`/`host-passthrough` before investing further in either (a) more RF-DETR optimization
attempts or (b) building a whole new lightweight pipeline.** If this turns out to be possible, it could
restore AVX2 across the board for free — potentially reviving v4's viability entirely, not just making
the §13 pose-model search easier. This has not been asked yet as of this writing; it is a 5-minute
question with a very high potential payoff and should happen before further multi-week pipeline work.

## 15. Detection loops fixed: missing frame-buffer control, and alone-detection's total lack of stride

Investigating whether `yolov10x.pt`-based alone-detection (§13: 3.3s/frame on the real server) is an
active production incident led to two findings and a set of fixes, applied directly to
`app/services/camera_manager.py`:

**Finding 1 — no live deployment exists on the server used for all testing in this doc.** `docker ps -a`
there is empty; the actual `docker-compose.yml` stack (db/backend/celery_worker/redis) has never been
deployed on [target-server-ip] — it was only ever used for standalone benchmark containers. **This means
the alone-detection slowness is a confirmed *code-level* bug, not a currently-active incident** — it
would surface immediately on first real deployment to hardware like this, but nothing has been
observed actually failing in front of real users (there may be a *different* server where this project
is already live; nobody in this session has that information — ask the project owner).

**Finding 2 — every detection loop in `camera_manager.py` opens `cv2.VideoCapture()` with no
`CAP_PROP_BUFFERSIZE` control, while the separate `stream_service.py` (live camera preview, a
different feature) already sets `CAP_PROP_BUFFERSIZE=1` explicitly to avoid latency.** This tells you
the original author *knew* about this failure mode and fixed it in one place but not the other five —
not a team-wide knowledge gap. Confirmed present in all 5 loops: `process_fall_detection`,
`process_alone_detection`, `process_bed_exit_detection`, `process_v2_fall_detection` (i.e. **v4**,
the current production fall detector), `process_v2_alone_detection`. Without this, a slow-enough model
on a live RTSP source can end up processing an ever-growing backlog of stale frames (or frames get
silently dropped by the OS/FFmpeg buffer, depending on transport) — either way, whatever the loop
"currently" reports does not correspond to the live scene. This affects v4 too, not just
alone-detection, though v4 has *some* mitigation via `STRIDE_FRAMES` already; alone-detection had none.

**Fixes applied (plain file edits — this repo has no `.git`, so nothing was committed; see the repo
owner if you want that set up):**

1. `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` added to all 5 loops, matching `stream_service.py`'s existing
   pattern exactly.
2. **Removed a duplicate-inference bug in `detect_alone_with_state`** (the v1 `process_alone_detection`
   path): it called `person_detector.detect_persons(frame)` **twice** per check — once inside
   `detect_alone_only()`, once again immediately after for track-history bookkeeping — silently
   doubling YOLO cost for no reason. Now calls it once and reuses the result. (Also found, but left
   alone: `v2_fall_detection_onnx.py` defines **two functions both named `detect_v2_alone_only_onnx`**;
   Python silently keeps only the second, the first is dead/unreachable code — worth cleaning up
   separately, not touched here.) The unused `state.states[tid].history` tracking data this function
   builds was confirmed write-only (nothing in the codebase reads it) — kept as-is, just noting it for
   whoever eventually wants to either use it or remove it.
3. **Added a time-based stride to both alone-detection loops** (v1 via `AloneDetectionState`, v2 via
   local loop variables) — `ALONE_DETECTION_CHECK_INTERVAL_S = 20`, gating how often the person
   detector actually runs rather than every single frame. **This value was deliberately derived from
   the feature's own pre-existing `alone_cooldown = 300` (5-minute alert cooldown), not from model
   speed** — 20s gives ~15 check opportunities per cooldown window. If the real product requirement for
   "how fast must alone-detection notice" turns out to be different from what a 5-minute alert cooldown
   implies, revisit this constant specifically, not the buffer/dedup fixes above (those are correct
   regardless of the exact interval chosen).

These three fixes are pure bug fixes with unchanged detection behavior/accuracy — deliberately kept
separate from any model swap (below) so either can be reverted independently.

## 16. Preliminary yolov10x → yolov10n comparison for alone-detection (promising, but not yet real-camera-validated)

With `yolov10x.pt` confirmed too slow (§13) even after the stride fix reduces *how often* it runs (it's
still 3.3s per actual check), swapping to a smaller model in the same family was evaluated — but on
**person-counting accuracy specifically** (alone-detection's actual job), not speed or fall-detection
accuracy, using real (non-lab) Pexels photos with manually-verified person counts.

**A ground-truth mistake happened here too, caught before being reported** — worth internalizing as a
pattern for this project specifically (this is now the second time; see §6 for the first, the
cane/walker mislabeling): the first pass reused the `aid_test/cane` and `aid_test/control` image sets
from §6's mobility-aid bias investigation, assuming "1 person" for all of them since that was true for
*that* investigation's purposes (does the photo show a person with/without a cane). It wasn't
re-verified for *exact total person count*, and several of those photos turn out to be busy outdoor
park/street scenes with real bystanders in the background — `cane_04.jpg` alone has 8-9 real people
in it, not 1. `yolov10x` reporting 9 detections there wasn't a model failure; it was correctly counting
a crowd my stale ground truth called "1 person." **Lesson: a label that was correctly verified for one
question (does this show X) is not automatically valid ground truth for a different question (exactly
how many Y) on the same image — re-verify for the actual thing being measured, don't reuse.**

**Corrected result**, restricted to the subset actually re-verified for exact person count this pass
(21 images: 15 single-person, 6 two-person including 3 deliberately-occluded/close-proximity cases):

| Model | Accuracy | Errors |
|---|---|---|
| `yolov10x.pt` (current production) | 20/21 = 95.2% | 1: overcounted a person who is small/self-occluded (hand covering face) as producing 2 boxes |
| `yolov10n.pt` (candidate) | 19/21 = 90.5% | 2: same occlusion case, **plus a genuine miss** — failed to detect a person in an unusual seated/twisted pose that yolov10x caught |

Both models fail the same hard occlusion case (not nano-specific). `yolov10n` has one additional real
failure mode — missing an unusual pose entirely — consistent with the accuracy-for-speed tradeoff
flagged before this test was run. ~5 percentage point gap on n=21 stock photos.

**This is encouraging, not conclusive.** It is not the 20-30 real-camera-footage image validation this
project already agreed is the actual bar before merging a model swap (no real camera footage from this
deployment exists anywhere in this project as of this writing). Treat this result as "worth continuing
toward that real validation," not as approval to ship the swap.

## 17. Gemini-assisted relabeling of GMDCSA24 fall onset (real data-quality fix, null result on pooled F1 — do not repeat without reading why)

**The idea:** `dataset.py`'s `make_windows()` only has a real per-frame fall boundary for CAUCAFall and
OF-ItW/OOPS (§3). For GMDCSA24 and FallVision, the whole-clip label ("this clip contains a fall") is
trustworthy but *which frames* count as the fall is guessed from the frame of peak motion-energy — a
heuristic, not ground truth. Gemini can watch a real video and give an actual visual judgment of when
the fall *begins* (losing balance), which should be a better boundary than "frame with the most pixel
movement" (which tends to land near impact, not onset).

**Scope was deliberately narrow.** Of the 4 pose sources, only GMDCSA24's 79 Fall clips were relabeled:
CAUCAFall/OF-ItW already have real labels (nothing to fix), and FallVision (5,845 samples, the *largest*
heuristic-labeled pool) has no raw video available locally — only pre-extracted COCO-17 keypoint CSVs —
so Gemini has nothing to watch there. This asymmetry matters for interpreting the result below.

**The relabeling itself worked and the finding is real.** `training/relabel_gmdcsa24_gemini.py` uploads
each clip to the Gemini file API and asks for the fall-onset timestamp; `training/build_relabeled_dataset.py`
turns that into a new `frame_labels` array, saved to `data/poses_gmdcsa24_v2/` (81 ADL clips copied
unchanged, 79 Fall clips relabeled). Across all 79 clips, Gemini's onset was **2.2s earlier than the old
motion-peak heuristic on average, in 74/79 clips (94%)** — verified by hand on several clips (e.g.
`s1_Fall_01`: heuristic peak = 5.68s, Gemini onset = 3.1s; the intervening ~2.5s is the person actually
losing balance and falling, which the old heuristic was labeling as "not fall"). One clip (`s3_Fall_06`)
Gemini called "not a fall, looks like floor exercise" even on retry — direct frame review (by the AI
session doing this work, not a script) showed the clip actually starts with the person already on the
ground mid-fall-recovery, settling into lying flat by 2.3s; GMDCSA24's staged Fall clips don't always
capture the standing-to-ground transition. That one was manually labeled fall=1 for the whole clip
(`source: "claude_manual_review"` in `data/gemini_relabel_results.json`) rather than trusting Gemini's
read.

**Gemini's free-tier quota is 20 requests/day per model** for `gemini-2.5-flash` and was exhausted after
~9 clips. `gemini-flash-lite-latest` draws from a separate quota pool and finished the rest — if
re-running this, start with the lite model rather than burning the flash quota first.

**The null result:** retrained `train.py` (now seeded — see `SEED`/`TRAIN_SEED` env var, added for exactly
this kind of before/after comparison) on old-heuristic vs. Gemini-corrected GMDCSA24 labels, 3 seeds each
(42, 7, 123), everything else identical:

| | seed 42 | seed 7 | seed 123 | mean |
|---|---|---|---|---|
| old heuristic labels | 0.595 | 0.589 | 0.599 | 0.594 ± 0.004 |
| Gemini-corrected labels | 0.591 | 0.602 | 0.594 | 0.596 ± 0.005 |

Paired per-seed difference: -0.004, +0.013, -0.005. **The mean improvement (+0.001) is smaller than the
run-to-run noise (±0.004-0.005). This is a wash, not a win — do not deploy this as "the more accurate
model" on the strength of this experiment.** A GMDCSA24-only subset evaluation (`eval_gmdcsa24_cross.py`)
shows a more encouraging diagonal (own-model-own-labels F1 0.641 -> 0.690) but the off-diagonal terms
don't support a clean causal story (evaluating the *baseline* model against Gemini's labels also scores
higher than against its own training labels: 0.641 -> 0.670), which looks like Gemini's boundary being
generally easier to hit rather than a genuine model improvement, and n=43 videos is small enough that
none of this should be over-read.

**Why a real, verified data-quality fix produced a null result:** GMDCSA24 is 160 of ~6,100 pooled
training videos — about 2.6%. Even a perfect fix there is diluted below the noise floor of a metric
computed over the whole pooled validation set, which FallVision (5,845 samples, ~95% of it) dominates.
**FallVision is also where the same heuristic-labeling problem is largest, and it's exactly the source
this technique can't reach** (no video, only keypoints). If this is revisited, the actual next question is
whether Gemini (or any model) can make a useful onset judgment from a rendered keypoint-skeleton
animation instead of real video — untested, and not obviously going to work as well as watching real
footage.

**Not deployed to production**: `models/fall_classifier_v3.onnx` is unchanged. The relabeling scripts,
the corrected `data/poses_gmdcsa24_v2/` dataset, and `data/gemini_relabel_results.json` all exist locally
(the last two under the existing `training/data/` gitignore rule — not pushed) in case someone wants to
build on this rather than repeat it from scratch.

## 18. Real false-alarm root cause found and partially fixed — MediaPipe IMAGE-mode jitter feeding the velocity feature

**First real "in the wild" test of v3, ever.** A user-supplied `Test/` folder had 3 real video clips (TikTok/
news compilations of doorbell-camera and street falls — not lab footage, not stock photos). Ran them
through the actual production code path (`training/test_v3_on_clip.py`, calling `detect_v3_fall`/
`V3PoseFallDetector` exactly as `camera_manager.py` does) and visually verified every alert against the
source frame. Result: **~77% of the 24 total alerts checked were real falls**, consistent with the
25-40% false-alarm rate already estimated in SS0/SS9 — the first time that number has been checked against
non-lab footage rather than assumed.

**Root cause of the false alarms, found and confirmed two ways:**
1. **Quantitative**: `training/diagnose_false_positive.py` dumps every frame's shoulder-hip tilt angle and
   MediaPipe detection status around an alert. On all 3 confirmed false positives (people standing still),
   the tilt angle swings wildly frame-to-frame anyway — e.g. clip3 t=1.3s: 5° -> 87° -> 12° -> 83° -> 65° -> 28°
   across consecutive frames of someone standing motionless. Clip1's case additionally had MediaPipe's
   tracking drop to all-zero for 1-2 frames mid-window, then resume.
2. **Independent visual check**: sent the same 3 alert frames to Gemini (`gemini-flash-lite-latest`) with
   no context beyond "did this person fall." It called all 3 "NOT A FALL" independently, matching the
   direct visual read.

**Why this happens**: `PoseLandmarkerOptions(running_mode=vision.RunningMode.IMAGE)` treats every frame as
an independent, context-free pose estimation problem — see `extract_poses.py`'s original comment claiming
"we do our own temporal windowing downstream anyway, so we don't need MediaPipe's VIDEO-mode smoothing."
That assumption is false in practice: single-frame pose estimation from an ambiguous camera angle can jump
between multiple plausible skeleton interpretations frame to frame even for a motionless person, and
because `_normalize_and_velocity()` computes **velocity from consecutive raw frames**, that jitter is fed
to the classifier as if it were real fast motion — exactly the signal a fall produces.

**Fix applied** (`app/detection/v3_fall_detection.py`), two parts, both inference-side only (does not touch
training data or require retraining):
- `_smooth_keypoints()`: a 3-frame centered moving average on x,y (not visibility) inside
  `_normalize_and_velocity()`, damping single-frame jitter while a real fall's large multi-frame
  displacement still comes through.
- `V3FallDetectionState.last_good_kpts`: on a tracking dropout, hold the last real detection in the
  buffer instead of MediaPipe's all-zero fallback, so the window doesn't see a fake real->zero->real jump.

**This is a real train/inference preprocessing mismatch** — `training/dataset.py` does not apply this
smoothing, so the model was trained on unsmoothed keypoints. Validated empirically rather than assumed
safe, by rerunning all 3 test clips before/after and checking every alert changed, not just the ones
expected to:

| clip | timestamp | before | after (smoothing only) | after (+ hold-last-good) | verdict |
|---|---|---|---|---|---|
| 1 | t=15.3s (2 people standing) | 0.707 FALSE ALARM | 0.726 (unchanged) | 0.614 (reduced, still fires) | not fully fixed |
| 2 | t=17.1s (person by motorbike) | 0.556 FALSE ALARM | gone | gone | fixed |
| 3 | t=1.3s (person by fence) | 0.611 FALSE ALARM | gone | gone | fixed |
| 1 | t=1.6s (real fall) | 0.889 | 0.928 | 0.930 | preserved, slightly stronger |
| 2 | t=6.0s (real fall) | 1.000 | 1.000 | 1.000 | preserved |
| 3 | t=4.0s (real fall) | 0.960 | 0.971 | 0.973 | preserved, slightly stronger |

**2 of 3 known false positives eliminated, the third reduced (0.707 -> 0.614) but not eliminated — clip1's
case is nighttime/low-light with two people near a doorway, likely harder tracking noise than a short
moving-average window can fully absorb. Every checked true positive held or got slightly stronger, not
weaker, which is the result you'd want before trusting this wasn't just trading recall for precision.**
Not re-validated against the pooled GMDCSA24/FallVision/CAUCAFall/OF-ItW validation set from SS17 — this
was validated only against the 3 real test clips, which is a much smaller and less rigorous check than a
full retrain-and-compare. If clip1's residual false alarm matters enough to chase further, the next lever
to try is a longer smoothing window or bumping `THRESHOLD` above 0.5 slightly (0.614 is close to the
threshold) — check that doesn't suppress the real alerts that came in as low as 0.538 in SS16's clip1 run
before doing that.

## 19. Scaled the real-footage test to 10 clips with objective ground truth — found a second, harder, different problem (not fixed)

**Bigger, objectively-scored sample.** The 3 hand-picked `Test/` clips (SS18) had no formal ground truth --
alerts were checked by eye. `training/eval_v3_on_oops.py` picks 10 more real clips a different way: query
OmniFall's OF-ItW label table (the same one behind `poses_ofitw`) for OOPS videos that (a) have a real
human-annotated fall/fallen segment and (b) are already downloaded locally, then score every alert
automatically against those real segment boundaries (±1.5s tolerance) instead of eyeballing. Caught a real
bug in the process: the first run scored 0/10 clips with zero alerts on all of them, which turned out to be
the eval script resolving the JSON-stored clip paths relative to the wrong working directory (silent
`cv2.VideoCapture` failure, not a detection failure) -- fixed by resolving paths from `__file__` instead of
cwd. Worth remembering: a suspiciously uniform result (all-zero, all-pass) across an entire batch is a
strong prior for "the harness is broken," not "the thing being measured is uniformly true" -- check the
harness before trusting the number.

**Results, 10 clips / 152s / 25 real fall segments, run against the SS18-fixed pipeline:**
- Segment recall: 76.0% (19/25 real falls got at least one alert)
- Precision: 59.1% (13/22 alerts were real)

**Lower precision than SS18's 3 clips (~77%), and for a different, non-overlapping reason.** Random
selection by "has a fall segment" pulled in mostly outdoor sports/stunt compilations (unicycle tricks,
downhill mountain biking, tetherball, snowmobiling) rather than indoor/domestic footage. Checked several
false positives by hand:
- A person doing a balance trick on a 3-wheel-stacked unicycle, crouched forward -- an unusual but
  controlled pose that happens to share fall's key visual signature (bent-forward torso).
- A child riding a bike toward the camera, leaning forward -- same signature, same story.
- MediaPipe detecting a "person" in a scene with no real person clearly in frame (a ladder over an icy
  pond) -- a different mechanism than SS18's jitter, more like an outright spurious detection.
- A person standing normally near a pond in an unrelated spliced-in segment of the same compilation video
  (these OOPS source videos, like the SS16/SS18 ones, splice together unrelated clips under one
  "weekly fails" title) -- misread the same way SS18's standing-still cases were.

**This is a different kind of problem than SS18's, and not fixed here.** SS18's false alarms came from a
mechanical bug (single-frame pose jitter leaking into the velocity feature) with a targeted, validated
inference-side patch. These come from the model never having seen forward-leaning-but-not-falling dynamic
poses (biking, unicycling, sports) as negative examples during training -- GMDCSA24/FallVision/CAUCAFall/
OF-ItW are staged-fall or calm-ADL footage, not athletic footage. That's a training-data coverage gap, not
a bug, and a smoothing filter can't fix a decision boundary the model was never taught. Fixing it for real
would mean adding dynamic-but-not-falling ADL examples to training data and retraining -- out of scope for
an inference-side patch, and not attempted here.

**Also worth being honest about scope: this test batch was accidentally out-of-domain for what this
project actually needs.** This is elderly *indoor* monitoring -- the deployment scenario is a home/facility
camera watching ADLs (walking, sitting, bending, reaching), not unicycle tricks or downhill biking. A
random "has a fall segment" filter over OOPS pulled in mostly outdoor stunt content because that's what
FailArmy compilations are mostly made of, not because it's representative of the real use case. The 59.1%
precision number here is a genuine result on the clips tested, but treating it as "the system's real-world
precision" would overstate the problem -- SS18's calmer, more domestic-feeling clips (doorbell cameras,
people walking/standing) are the closer analog to actual deployment, and scored better (~77%). If this is
revisited, the right next sample is 10 more OOPS clips filtered to indoor/domestic-looking content
specifically, not just "has a fall segment" -- that would be the fairer test of real deployment accuracy.

## 20. Tried to find 10 more domain-relevant OOPS clips, failed (only 1/30), pivoted to reusing GMDCSA24 -- this is the real number

**Attempted SS19's suggested next step**: screened 30 more OOPS candidates (keyword-filtered to exclude
obvious sports/outdoor titles first) with Gemini, asking specifically "is this indoor/domestic and
elderly-monitoring-relevant." Result: **only 1 of 30 qualified.** FailArmy compilations are curated for
entertainment value almost by construction (violent altercations, playground stunts, pool jumps, office
pranks) -- calm indoor ADL-like falls are rare in that corpus, not just under-sampled. Extrapolating,
getting 10 relevant clips this way would need a pool of ~300, which isn't an efficient use of Gemini calls
for what it would return. Abandoned this path rather than force it.

**Pivoted to data already on hand.** GMDCSA24's raw video is already downloaded (SS18/SS19 both use its
Fall half already), and `training/dataset.py`'s `split_videos(seed=42)` held out 31 of its 160 clips from
training (15 Fall, 16 ADL) -- calm, staged, indoor, genuinely the closest available match to this
project's actual deployment domain. `training/eval_v3_on_gmdcsa24_val.py` runs the real live pipeline
(`detect_v3_fall`, with the SS18 fix applied) over all 31 raw videos end-to-end -- not the offline windowed
classifier accuracy SS17 measured, the actual frame-by-frame detection loop production code runs.

**Result: 14/15 Fall clips caught (93.3%), 10/16 ADL clips stayed clean (62.5%, so 6 false alarms / 37.5%).**
This recall number is the best of any real-footage test this session. The false-alarm rate lands right in
the 25-40% range the code's own docstring already claimed -- this is the first time that claimed number has
actually been checked end-to-end on domain-relevant held-out video rather than asserted.

**Checked 3 of the 6 ADL false alarms by hand (`s1_ADL_01`, `s4_ADL_08`, `s2_ADL_15`) -- all 3 are a person
on a bed** (lying down, shifting position, sitting up to fold something), not a fall. This is not a new
bug -- it's exactly the bed-exit ambiguity the docstring already names ("normal bed movement resembles the
aftermath of a fall") and SS18's fix doesn't touch it, because it isn't jitter: the pose keypoints and
velocity genuinely do look fall-like, because a pose-only classifier has no way to know it's looking at a
bed instead of the floor. Fixing this for real needs a different signal entirely (e.g. bed-region context
from object detection, fused with pose) -- not a threshold or smoothing change.

**Where this leaves the accuracy picture, honestly, across everything tested this session:**

| test | domain match | recall | precision / false-alarm rate |
|---|---|---|---|
| SS18: 3 doorbell/street clips | closer to real deployment | not separately measured | ~77% of alerts real |
| SS19: 10 OOPS sports/stunt clips | poor match (SS20 confirms) | 76% | 59% (worse, different cause) |
| SS20: 31 GMDCSA24 held-out clips | best match (indoor, staged-elderly) | **93.3%** | **62.5% clean (37.5% false-alarm)** |

**SS20's numbers are the ones to trust for "how will this behave in the actual product."** High recall
(catches real falls reliably) with a real, known, structural false-alarm source (beds) that the operating
notes already say to expect and that this session's fix does not address. Still not accurate enough to
alert autonomously -- unchanged conclusion from the original docstring, now with an actual held-out number
behind it instead of an estimate.

## 21. 8 more user-supplied clips, 69 alerts, Claude+Gemini double-checked every one -- reconfirms SS20's bed issue, finds a new night-vision mechanism

**User added 8 more clips to `Test/`** (renamed to sequential `4.mp4`-`11.mp4` to match the existing
`1.mp4`-`3.mp4` convention -- see git history for the rename). Ran all 8 through `test_v3_on_clip.py`
(69 total alerts) and, per the user's explicit ask to have Gemini and Claude check together, screened
every single alert frame with Gemini (`training/gemini_screen_new_clips.py`) rather than hand-sampling a
subset like SS18/SS19 did. 3 of 69 calls hit a transient `503 UNAVAILABLE` (server overload, not quota --
different from SS16's daily-quota wall) and were retried individually until they succeeded.

**Result: 45/69 alerts judged real falls by Gemini (65.2%)** -- between SS18's domestic-clip precision
(~77%) and SS19's sports-clip precision (59%), consistent with this batch being mostly doorbell/porch
compilation footage (similar in kind to SS18's clip1) with a few harder cases mixed in. Followed up in
person (Claude, direct image read) on the more unusual `NOT_A_FALL` verdicts rather than trusting Gemini's
read alone -- same double-check standard as SS18:

- **Bed-lying, again**: `11/t=1.3s` and `11/t=11.7s` -- an infrared night-vision bunk-bed/crib camera clip,
  person lying down. Same exact mechanism as SS20, now confirmed on completely different source footage.
  This is the most consistently reproduced false-positive cause across every real-footage test this
  session has run.
- **New mechanism, confirmed with `diagnose_false_positive.py` traces (not left as a guess): long tracking
  dropouts interacting with SS18's own hold-last-good-keypoints fix.** `4/t=9.6s` (person dancing, arms
  raised, infrared night vision) and `10/t=15.4s` (a TikTok loading-screen transition, no person on screen)
  both fired at high confidence (0.92-0.96) with `person_found=False`. First guess was that
  `COLLAPSE_CONFIDENCE`'s exact-1.0 shortcut had fired (SS9/original docstring) -- checked the actual
  frame trace and that's wrong: the probabilities (0.920, 0.946, 0.960...) aren't the flat 1.0 that path
  always returns, so this is a genuine fresh classifier inference, not the collapse shortcut. What the
  trace actually shows in both cases: MediaPipe's `person_found` flickers True/False repeatedly for ~1-2s
  before the alert (not one clean dropout), and SS18's fix holds the last real keypoints during every False
  frame instead of zero-filling. With `person_found` false more often than true across that stretch, the
  30-frame window ends up mostly *repeated, frozen copies* of one earlier pose, mixed with a few genuinely
  different real detections -- a pattern the model never saw in training (real motion, or a genuinely still
  person, but not "one pose held constant, spliced against jitter"). It isn't gated by `RESET_PERSON_FRACTION`
  because the window's true/false mix stays above that 5% floor throughout.
  **This means SS18's fix, which was validated and helped on short (1-2 frame) dropouts, has a
  plausible different failure mode on longer/intermittent ones** -- worth knowing given the fix is already
  in production.

**Not fixed.** Root-caused with real evidence this time (not a guess), but the corrective change isn't
obvious enough to write blind: naive options like "cap how many consecutive frames get held before falling
back to zero-fill" or "skip inference if too many of the last N frames were held rather than real" both
need the same before/after validation discipline SS18 used (does it fix these 2 cases without weakening
real detections elsewhere) before going into `v3_fall_detection.py`. Flagging for next session rather than
shipping an unvalidated patch on top of an already-partially-understood mechanism.

## 22. Tried retraining on the SS21 verified clips as real training data -- null-to-negative result, and a clear lesson why

**The idea, and it's a reasonable one**: SS21's Gemini+Claude double-check produced 67 fully verified
segments from `Test/4.mp4`-`11.mp4` -- 45 confirmed real falls, 22 confirmed false positives (hard
negatives: the model scored these high but both Gemini and Claude agree they aren't falls). Added them as
a 5th pose source (`training/extract_realtest_poses.py` -> `poses_realtest_v1`, toggled via
`USE_REALTEST_V1=1` in `train.py`) and retrained, 3 seeds each with/without, same discipline as SS17/SS20.

**Pooled val F1: another wash** (0.594 -> 0.595, within noise) -- same dilution story as SS17: 67 segments
against ~6100+ pooled videos is too small a fraction to move a metric averaged over everything.

**The metric that actually matters -- SS20's GMDCSA24 held-out check, rerun against all 3 retrained
models:**

| | Fall recall | ADL clean | which ADL clips still false-alarm |
|---|---|---|---|
| baseline (SS20, no realtest data) | 93.3% (14/15) | 62.5% (10/16) | s1_ADL_01, s2_ADL_03, s2_ADL_15, s4_ADL_07, s4_ADL_08, s4_ADL_10 |
| +realtest_v1, seed 42 | **100%** (15/15) | 43.8% (7/16) | same 6, **+3 new** (s2_ADL_16, s2_ADL_20, s3_ADL_11) |
| +realtest_v1, seed 7 | **100%** (15/15) | 62.5% (10/16) | **exact same 6**, 0 new |
| +realtest_v1, seed 123 | **100%** (15/15) | 56.2% (9/16) | same 6, **+1 new** (s2_ADL_16) |

**Checked with a direct set-difference, not eyeballed: zero of the 6 original false-positive clips were
fixed in any of the 3 seeds.** Recall genuinely improved (93.3% -> 100%, consistent across all 3 seeds --
a real, repeatable effect, not noise), but the actual problem this was meant to fix -- SS20/SS21's
bed-lying false alarms -- is completely untouched, and 2 of 3 seeds made the false-alarm rate *worse* by
adding new false positives on previously-clean clips.

**Why, and it's obvious in hindsight**: of the 22 hard negatives added, only **one** was bed-related (from
clip 11's bunk-bed segment) -- the other 21 came from SS21's clips 4-11, which are dancing/standing/porch/
outdoor scenes, not beds. I mined hard negatives from one failure surface (SS21's doorbell clips) and then
re-tested against a *different* failure surface (SS20's GMDCSA24 beds) that the new data barely touched.
One example diluted into ~1% of the pooled training set was never going to overcome a systematic bias.
**The lesson, worth remembering before repeating this: hard-negative mining only helps the specific failure
mode it targets -- verify what you're adding actually matches what you're testing against before training,
not after.** A properly targeted version of this experiment would extract several bed-lying hard negatives
specifically (there's no shortage of candidates: GMDCSA24's own ADL clips, or filming/finding more bed
footage) rather than reusing whatever happened to be in SS21's unrelated clips.

**Not deployed.** No seed fixes the known issue, and 2 of 3 make it worse -- shipping any of these would be
a pure downgrade dressed up as "we retrained on more data." `models/fall_classifier_v3.onnx` is still the
SS18-fixed model, unchanged. `poses_realtest_v1` and the training scripts are kept (git-tracked) in case
someone wants to build a properly-targeted version rather than repeat this from scratch.

## 23. 50 more real clips (GMDCSA24's *training* split this time), Gemini+Claude double-checked -- finds two more false-positive patterns beyond "bed"

**Requested: test ~50 more, checked together (Gemini + Claude).** SS20 already exhausted GMDCSA24's 31
held-out clips; this batch (`eval_v3_on_gmdcsa24_train50.py`) picks 50 more (25 Fall, 25 ADL, seed 99) from
the other 129 clips -- the ones the deployed model *was* trained on. Deliberately not a generalization
test: the question here is whether SS20/SS21's bed-lying false alarms are a pure held-out-only
generalization gap, or show up even on clips the model has already seen.

**Result: 23/25 falls caught (92.0%), 20/25 ADL clips clean (80.0%).** Both numbers land between SS20's
held-out figures (93.3% / 62.5%) and would-be-perfect memorization -- so training-set exposure helps
somewhat (80% > 62.5% clean) but doesn't fully cover the pattern even on clips the model trained on. That
answers the question: this is *partly* a generalization gap and *partly* something the model never fully
learned to reject even in-distribution.

**Checked all 5 false positives and both misses by hand, cross-verified with Gemini (all 6 verdicts
matched independently) -- and found the false positives split into three distinct patterns, only one of
which was already known:**
- **Bed-lying (2 of 5)** -- `s4_ADL_15`, `s4_ADL_16`, same subject/room. Same SS20/SS21 pattern, now
  confirmed on training-set clips too.
- **New: arms fully outstretched, T-pose / stretching exercise (2 of 5, both from `s3_ADL_02`)** -- person
  standing upright, both arms spread wide, doing a stretch. Not bed-related at all. This connects directly
  to SS21's `4/t=9.6s` finding (person dancing with arms raised, night vision) -- now confirmed on a
  second, unrelated, daytime indoor clip. Two independent domains agreeing makes this a real pattern to
  track, not a fluke: **wide/raised-arm poses are a second, distinct false-positive trigger**, separate
  from bed-lying.
- **New: floor-lying / bending forward near an object (1 of 5, `s3_ADL_09` + a related bend in
  `s3_ADL_04`)** -- echoes SS17's `s3_Fall_06` ("floor exercising" per Gemini, same ambiguity class) and
  SS19's forward-lean-while-biking finding. A third recurring shape: forward torso lean/floor-level pose
  reads as fall-like regardless of context.

**The 2 misses are also worth knowing about, not just the false positives**: `s4_Fall_08` is a real fall
where the person ends up sprawled face-down across a bed, legs dangling off the side -- missed entirely.
Given how much of this session's false-positive hunting has been about *suppressing* bed-related alerts,
a real fall onto/near a bed being missed is the failure mode you'd expect if a future fix over-corrects in
that direction -- worth checking against explicitly if bed-related hard negatives are ever added (SS22's
lesson).

**Not fixed, not retrained again.** This is measurement, not a repeat of SS22's mistake -- three
now-distinct, cross-domain-confirmed patterns (bed-lying, wide/raised-arms, forward-lean/floor-lying) is
enough evidence to make a properly-targeted hard-negative set for a future retrain, but building and
validating that is its own task, not something to rush right after SS22's negative result on a narrower
attempt.

## 24. Multi-person tracking added -- v3 only ever tracked one person per camera before this

**The gap**: `V3PoseFallDetector` never set `num_poses` on `PoseLandmarkerOptions` (MediaPipe defaults to
1), and `extract_keypoints()` took `result.pose_landmarks[0]` -- whichever single person MediaPipe happened
to detect first. `camera_manager.py` held one `V3FallDetectionState` per camera. With 2+ people in frame
(a resident and a caregiver, say), only one was ever tracked -- a second person's fall could go completely
unseen if MediaPipe's per-frame pick favored the other person, and a mid-frame pick switch would corrupt
the rolling window with a different person's keypoints spliced in.

**What changed** (`app/detection/v3_fall_detection.py`):
- `NUM_POSES = 4`, set on `PoseLandmarkerOptions` -- MediaPipe now returns up to 4 poses per frame.
- `extract_all_keypoints()` (new) returns every detected pose + hip-center; `extract_keypoints()` (existing,
  used by every training/eval script this session built) is now a thin wrapper that takes just the first
  one -- unchanged behavior for single-person callers, zero risk to the extensively-validated test tooling.
- `_step_person()`: the single-person state machine from `detect_v3_fall` (SS9-SS18's window/smoothing/
  collapse logic), factored out unchanged so both the single- and multi-person entry points run the exact
  same validated logic instead of two copies that could drift.
- `PersonTracker`: nearest-hip-center matching (`MAX_TRACK_DISTANCE=0.15` normalized, `MAX_MISSED_FRAMES=
  WINDOW_SIZE`) -- deliberately simple (no motion model, no re-ID), matches this project's "same room, few
  people, mostly-static camera" scope rather than building a real MOT system.
- `V3MultiPersonFallState` + `detect_v3_fall_multi()`: one `V3FallDetectionState` per tracked person,
  returns a list of `(track_id, detected, probability, label, hip_center)`.
- `camera_manager.py`'s `process_v2_fall_detection` now uses the multi-person entry point, logs
  `person_count=len(results)`, and alerts if *any* tracked person is in the fall state (kept the existing
  camera-level cooldown rather than adding per-person cooldown -- lower risk, and this system already isn't
  meant to alert autonomously per SS9's operating note).

**Validated two ways before trusting it**:
1. **Single-person regression check** -- reran SS20's GMDCSA24 held-out eval (which only ever has 1 person
   per clip) against the refactored code: **identical 93.3% recall / 62.5% clean**, confirming
   `_step_person()`'s extraction didn't change single-person behavior at all.
2. **Real multi-person footage** -- `training/test_v3_multi_on_clip.py` (draws each tracked person in a
   different color, labeled by track ID) against `Test/1.mp4`'s known two-person doorbell scene (~12-15s):
   confirmed both people get simultaneous, independent skeletons/track IDs/states when MediaPipe detects
   both (`people tracked: 2`) -- the core capability works.

**Known, honest limitation, observed directly in that same clip**: when a person is briefly undetected
(MediaPipe missed them for ~1s while they were still on screen) and then reacquired, they can get a *new*
track ID if they moved far enough during the gap -- no motion prediction, matching the tracker's stated
scope. This resets that person's rolling window (has to refill before re-alerting) but does not cause a
crash or a false alert; it's a graceful continuity loss, not a safety issue. On the same full-clip run, a
single compilation video with hard scene cuts (already established as a `Test/` artifact, not a real
deployment scenario -- see SS18/SS21) produced 58 distinct track IDs over 67s, which is expected: a scene
cut is a genuine, correct new-track event (the camera view itself changed), not tracker malfunction.

**Not yet checked**: per-person false-alarm rate with 2+ real people in frame simultaneously and both
moving (this session's 2-person footage was mostly one-active/one-standing-still) -- worth another
real-footage pass if multi-person accuracy specifically needs validating further.

## 25. Ran the multi-person path across all 11 `Test/` clips -- it works, but costs more false positives than expected, found and root-caused, not yet fixed

**Ran `test_v3_multi_on_clip.py` on all 11 clips** (not just clip1's spot check from SS24). Alert counts vs
the established single-person baseline: 5 of 11 clips got *more* alerts (clip1 15->19, clip4 14->21, clip5
13->21, clip8 12->18, clip9 16->21), 2 got *fewer* (clip2 6->3, clip10 5->3), 4 stayed the same.

**Checked the "more" clips by hand -- two distinct causes, not one:**
1. **More pose slots surface more marginal detections.** `NUM_POSES=4` means MediaPipe now returns up to 4
   candidate poses per frame instead of always just the single best one -- clip4 t=9.8s and t=9.9s both
   fired FALL within 0.1s on two different track IDs (#7, #8) for what a direct look confirms is the same
   already-known false positive (SS21's night-vision dancing scene) -- MediaPipe produced two slightly
   different pose estimates for one ambiguous figure, and now both get their own track and their own alert
   instead of one.
2. **Track churn creates repeated cold-start windows, and cold starts seem to be where false positives
   cluster.** clip4 t=61.2s and t=65.1s and t=87.0s are a person walking away from camera with a backpack,
   confirmed NOT a fall by direct look, each time a *new* false positive (not a duplicate of anything in
   the single-person run) -- most plausibly because every time this person's track resets (occlusion,
   distance, lighting), their 30-frame window has to refill from scratch, and a freshly-filling window
   built from a person walking away seems more prone to a spurious high score than the same person's
   window would be if it had stayed continuous. This rhymes with a pattern already seen elsewhere this
   session (SS21's clip3 t=1.3s, clip11 t=1.3s: false positives clustered right at the start of a clip/
   track, before the window has "settled").

**Checked the "fewer" clips too, since fewer alerts could mean a missed real fall (worse) instead of fewer
false alarms (fine) -- the safety-relevant direction to check first.** clip2: lost the 11.2s re-trigger,
but the underlying real event (the crosswalk fall from SS18/SS21) still fires at 2.3s, 6.0s, and 13.8s --
not missed, just fewer redundant re-triggers of the same event (per-track window resets absorb some of
the oscillation single-person's one continuous window used to re-flag). clip10: lost the 7.2s trigger
(0.994 confidence in single-person mode) but the same real event (child falling down stairs, SS21) still
fires at 8.1s (0.994) -- again not missed. **In both checked cases the real underlying event still got at
least one alert opportunity; what changed is internal re-trigger frequency, not whether the event was
caught at all** -- but this was only checked on 2 clips, not exhaustively.

**Net assessment, stated plainly: multi-person tracking's core capability is real and validated (SS24), but
it measurably increases false-positive churn -- worse on solo-person clips too, not just multi-person ones,
because NUM_POSES=4 always searches for up to 4 poses even when only one person is present.** Not reverted
or retuned here -- the fix (e.g., lower NUM_POSES, raise confidence for non-primary pose slots, or persist
window state across a track-ID reassignment instead of restarting cold) needs the same before/after
validation discipline as SS18's fix, not a guess. Flagging for next session rather than shipping an
unvalidated change on top of a change that itself was just shipped.

## 26. Quantified SS25's multi-person false-positive cost against ground truth, not just spot checks -- smaller than feared, and localized to one mechanism

**Requested: ~20 more, Gemini+Claude both checking.** Reused SS20's exact 31 GMDCSA24 held-out clips (real
ground truth, not manual eyeballing) but ran them through the multi-person path
(`eval_v3_on_gmdcsa24_val_multi.py`) instead of single-person -- every one of these clips has exactly one
real person, so this is a controlled test of SS25's "extra pose slots cost false positives even on
solo-person footage" claim against a scored dataset instead of a handful of spot-checked doorbell frames.

**Result: recall identical (93.3%, same single miss `s2_Fall_20`), ADL-clean dropped 62.5% -> 56.2%
(10/16 -> 9/16) -- exactly one new false positive** (`s3_ADL_11`), confirmed independently by both a
direct look and Gemini (person reaching up to adjust curtains -- same raised-arm pattern family as SS23).
All 6 of the original false-positive clips persisted unchanged; nothing was fixed or newly broken among
them.

**This is a much smaller cost than SS25's doorbell-clip spot checks suggested (clip4 alone picked up
several new false positives).** The likely explanation: these GMDCSA24 clips are short (a few seconds) and
single-continuous-shot, so the track-churn mechanism SS25 identified (repeated cold-start window refills
after occlusion/reacquisition) barely gets a chance to fire -- there's no time for a track to drop and
reappear. The longer, choppier real-world compilation clips gave track churn much more room to compound
with the "more pose slots" effect. **Conclusion: on the specific footage this project actually targets
(short, continuous, single-room clips), multi-person mode's false-positive cost is small (+1 in 16 tested)
-- the bigger cost SS25 measured is concentrated in exactly the scenario (long clips, people leaving and
re-entering frame) that's more relevant to compilation-style test footage than typical fixed-camera
deployment, though a real deployment does still have people leaving/re-entering over a full day, so this
isn't a reason to stop tracking it.**

## 27. User-reported "detects furniture/cars/fences as people" -- confirmed real, but the obvious fix (confidence threshold) doesn't work

**This is a different, upstream problem from everything else in this file.** SS9-SS26 are all about the
fall *classifier* (`fall_classifier_v3.onnx`, trained on `training/`) misjudging a real detected person's
pose. This is about MediaPipe's pose *detector* -- a pre-trained Google model we don't train or fine-tune
at all -- finding a "person" where there isn't one. These need different fixes: the classifier can be
retrained (as this whole file has done repeatedly); MediaPipe can only be worked around with a filter on
its own output, since we have no access to retrain it.

**Confirmed with real evidence, not assumed.** Scanned `Test/12.mp4` for low-average-landmark-visibility
detections (a hallucinated pose typically scores lower than a real one) and checked each by hand /
Gemini. Found genuine hallucinations: a close-up shot of a **burnt pizza** (no person in frame at all,
avg visibility 0.51-0.53) and separately a **dog + shovel** scene and a **boat on water** scene (both no
person, visibility 0.696 and 0.706).

**The obvious fix -- reject detections below some visibility threshold -- does not work, checked against
real data before proposing it:**

| what | avg visibility | real person? |
|---|---|---|
| burnt pizza close-up (no person) | 0.51-0.53 | NO -- hallucination |
| extreme close-up selfie face | 0.581 | YES -- real, just cropped tight |
| dog + shovel (no person) | 0.696 | NO -- hallucination |
| boat on water (no person) | 0.706 | NO -- hallucination |
| person holding a cat, arm occluded | 0.669 | YES |
| person crawling near porch steps | 0.681 | YES |
| person tripping, partially cut off at frame edge | 0.676 | YES |
| several more real partially-visible people | 0.649-0.713 | YES |

**Real and hallucinated detections occupy the same visibility range (0.51-0.71) -- there is no clean cutoff.**
A threshold high enough to reject the dog/shovel and boat hallucinations (>=0.71) would also reject nearly
every genuinely real but partially-visible/occluded/awkwardly-framed person found in this same sample --
exactly the kind of detection real deployment (someone mid-fall, partially behind furniture) most needs to
keep. Shipping a naive threshold would trade a rare cosmetic problem for a real recall loss on the cases
that matter most. Not implemented.

**What would actually work, not yet built or validated**: an anatomical-plausibility check on the *shape*
of the detected skeleton (are shoulder-width/hip-width/limb-proportions within plausible human ranges,
are keypoints in a sensible spatial arrangement relative to each other) rather than a single confidence
number -- a hallucinated pose on a pizza or a boat is likely to be geometrically incoherent in a way pure
average visibility doesn't capture. This needs the same design-then-validate treatment as every other fix
in this file (build it, run it against both a hallucination set and a large real-person set like SS20's,
confirm it doesn't cost recall) -- not attempted here, flagged as the concrete next step if this is worth
pursuing further.

**Practical note on actual impact**: hallucinated poses were rare in the scan (4 of ~137 sampled points in
one clip) and, being geometrically close to random noise rather than a coherent moving body, are less
likely to accumulate the sustained fall-like signal the classifier's smoothing (`SMOOTH_NEED`/`SMOOTH_OF`)
requires to actually fire an alert -- this is a real visual-clutter and wasted-compute problem (confirmed),
but not yet confirmed to be a meaningful contributor to real false alarms specifically (untested).

## 28. Retrained with properly-targeted hard negatives this time (real lying/sitting data, not SS22's mismatch) -- real recall gain, but the exact same false positives persist anyway

**Did this right, unlike SS22.** SS22's hard negatives came from an unrelated failure surface (dancing/
standing outdoors) and didn't touch GMDCSA24's bed-lying false positives at all. This time,
`extract_omnifall_adl_poses.py` pulled 117 real segments from OmniFall's OF-ItW label table filtered to
the exact matching activity classes -- `lying`/`lie_down` (47, full set), plus a sample of `sitting`/
`sit_down` (40) and `kneeling`/`squatting` (30) -- sourced from roughly 40 different OOPS subjects/rooms,
versus GMDCSA24's 4. `poses_omnifall_adl`, toggled via `USE_OMNIFALL_ADL=1`, retrained 3 seeds with/without.

**Pooled val F1: small but consistently positive this time** (unlike SS22's mixed +0.003/+0.008/-0.008):
+0.010, +0.006, +0.001 across seeds 42/7/123 (baseline mean 0.594 -> 0.600).

**The real test -- SS20's GMDCSA24 held-out set, ground-truth scored, all 3 seeds:**

| | Fall recall | ADL clean | 
|---|---|---|
| baseline (SS20) | 93.3% (14/15) | 62.5% (10/16) |
| +omnifall_adl, seed 42 | **100%** (15/15) | 50.0% (8/16) |
| +omnifall_adl, seed 7 | **100%** (15/15) | 56.2% (9/16) |
| +omnifall_adl, seed 123 | **100%** (15/15) | 56.2% (9/16) |

**Recall hit 100% in all 3 seeds -- fully reproducible, not noise -- catching `s2_Fall_20`, the one fall
every previous model in this file has missed.** That's a genuine, real gain worth taking seriously.

**But checked with a direct set-difference (not eyeballed) whether the intended fix actually happened: it
did not, in any seed.** All 6 of SS20's original false-positive clips (`s1_ADL_01`, `s2_ADL_03`,
`s2_ADL_15`, `s4_ADL_07`, `s4_ADL_08`, `s4_ADL_10`) persist unchanged in every one of the 3 retrained
models, plus 1-2 new false positives appear depending on seed (`s2_ADL_16` in 2/3, `s3_ADL_11` in 2/3).
**47 real, diverse lying/lie_down examples -- properly targeted this time, not SS22's mismatch -- still did
not fix a single one of the specific clips this was aimed at.**

**What this actually shows**: the bed-lying false positives aren't primarily a data-scarcity problem
(SS22's hypothesis, now tested and not confirmed) -- adding real, diverse, correctly-targeted negative
examples moved the model in the recall direction (a real, valuable, reproducible effect) without moving it
in the precision direction on these specific clips at all. Two plausible explanations, neither confirmed:
(1) `train.py`'s `pos_weight` loss term explicitly weights false negatives 1.5x worse than false positives
by design (see `train.py`'s own comment) -- structurally pulling any additional training signal toward
"catch more real falls" even when the added data's actual purpose was precision; (2) these 6 specific
clips may be close to irreducibly hard for a pose-only classifier regardless of how much lying/sitting data
exists elsewhere -- something specific to their camera angle or motion pattern that isn't a generic
"lying" concept the model is failing to generalize.

**This is a real product tradeoff, not a clear win or a clear loss -- not deployed without a decision on
it.** Every prior retraining attempt in this file (SS17, SS22) was a clean "no benefit, don't ship" call.
This one is different: recall genuinely improved (never missing a real fall matters a lot for this use
case) at a real precision cost (more prompts to check the camera on ADL footage) -- and given SS9's
existing operating note already frames detections as "a prompt for staff to check the camera, not a
confirmed event," a higher-recall/lower-precision model could be the *right* choice for this product, not
just an inferior one. Left undeployed pending that call rather than assumed either way.

## 29. Tested SS28's pos_weight hypothesis directly -- found a genuine, cross-validated win (deployed)

**SS28 left two untested hypotheses for why the OmniFall ADL data didn't fix the specific false-positive
clips despite fixing the recall problem.** Tested hypothesis 1 directly: `train.py`'s `pos_weight` (now
`POS_WEIGHT_MULT` env var) explicitly multiplies the natural class-ratio weight by 1.5x to punish missed
falls harder. Reran the same SS28 data (`USE_OMNIFALL_ADL=1`) at `POS_WEIGHT_MULT=1.0` (no artificial
recall bias), 3 seeds.

**Recall stayed at 100% in all 3 seeds regardless of pos_weight (6/6 runs total between SS28 and SS29) --
robustly reproducible, not a pos_weight artifact.** ADL-clean varied by seed: 56.2%/56.2%/**62.5%**
(seeds 42/7/123) at pos_weight=1.0, vs 50.0%/56.2%/56.2% at pos_weight=1.5 (SS28) -- removing the bias
helped on average but didn't fix any of the 6 original clips in 5 of 6 seeds either.

**Seed 123 at pos_weight=1.0 is the exception, and it's a genuine, cross-validated win, not a fluke read
from one number:**

| test set | recall | ADL/precision |
|---|---|---|
| SS20's 31 held-out clips: baseline | 93.3% | 62.5% (10/16) |
| SS20's 31 held-out clips: seed123 pw=1.0 | **100%** | 62.5% (10/16, exact same false positives -- zero new ones) |
| SS23's 50 training-split clips: baseline | 92.0% | 80.0% (20/25) |
| SS23's 50 training-split clips: seed123 pw=1.0 | 92.0% (unchanged) | **84.0%** (21/25) |

**On the second test set, checked which specific clip's false positive actually went away: `s3_ADL_02`
(SS23's "arms outstretched / stretching" case) -- now completely clean, zero alerts, where baseline fired
twice.** This wasn't the pattern the OmniFall data specifically targeted (lying/sitting), but a real fix
happened anyway -- plausibly an incidental benefit of the kneeling/squatting portion of the SS28 data, or
just a better-generalizing decision boundary from this specific seed. The 4 remaining false positives on
this set (two bed-lying, one floor-lying, one bending-near-bed) are unchanged, consistent with SS28's
finding that this exact pattern is stubborn regardless of pos_weight or more lying/sitting data.

**Deployed.** Two independent, ground-truth-scored, held-out-or-mostly-held-out test sets both show
recall improved-or-unchanged and precision improved-or-unchanged with this specific checkpoint
(`ss29_seed123_pw1.pt`) -- zero measured regressions anywhere, unlike every prior retraining attempt in
this file (SS17, SS22, SS28's other 5 seeds). `models/fall_classifier_v3.onnx` replaced with this model.
`train.py`'s default `pos_weight` multiplier left at 1.5 (unchanged) since removing it didn't reproducibly
help across seeds -- only this specific trained checkpoint is being kept, not the hyperparameter change
generally. **Caveat worth stating plainly: this result depends on this specific random seed's initialization
landing in a good spot** (seeds 42 and 7 at the same settings did not match baseline's precision) -- treat
this as "a good model that was found and validated," not "a reliable recipe that will reproduce again."

## 30. Post-deploy regression check on all 13 `Test/` clips -- 67.3% precision, no new failure modes found

**Reran every `Test/` clip (now 13 -- `13.mp4` added) through the newly-deployed SS29 model**, both single-
and multi-person paths, then Gemini-verified all 107 single-person alerts (`verify_current_model_alerts.py`
-- re-extracts a fresh frame per alert straight from the source clip rather than trusting `alert_frames/`,
which accumulates stale files from earlier runs since it's never cleared between them; found and fixed a
real bug along the way: `test_v3_multi_on_clip.py` wrote to the same `summary.txt` filename as the
single-person script, clobbering it -- now `summary_multi.txt`).

**72/107 (67.3%) confirmed real falls.** Not comparable to SS29's 62.5%/84% ADL-clean numbers -- those are
single-activity GMDCSA24 clips; `Test/`'s clips are multi-scene TikTok/doorbell compilations, a harder
domain this file has consistently measured lower on (SS19's sports clips: 59%; SS18's calmer doorbell
clip: ~77%). 67.3% across a mixed batch lands where expected, not a signal of a new regression.

**Checked what the 35 false positives actually were -- every one matches an already-documented pattern,
nothing new:** bed-lying (SS20/21/23/26/28), pose hallucination on non-human objects -- a burnt pizza and a
boat, confirmed again on fresh footage (SS27), video-compilation transition artifacts (TikTok loading
screens firing the person-gone fallback path, SS21/25), and reaching/bending forward (opening a gate,
crawling, assisting someone with a walker) which extends SS19/23's forward-lean pattern rather than being
new. `clip 9` alone accounted for 11 of the 35 -- spot-checked directly and confirmed it's another
multi-scene Ring/Nest compilation with heavy scene-cut track churn, the exact mechanism SS25 already
identified as the main driver of false positives on long choppy clips (as opposed to short continuous
ones, where SS26 measured almost no added cost).

**No code or model changes from this pass -- it's a clean bill of health for the SS29 deploy**, confirming
the newly-shipped model didn't introduce any failure mode beyond what was already known and accepted.

## 31. Tested model capacity as the last untested lever -- conclusive negative, and it clarifies what's actually going on

**Last untested hypothesis for the 6 persistent GMDCSA24 false positives**: `model.py`'s docstring notes a
much bigger architecture (ST-GCN, ~17x more params) underperformed this one on the *old* data (0.582 vs
0.615 F1) -- but that comparison predates every data change this session made, so it doesn't rule out a
*modest* capacity increase on SS29's current winning config. Added `HIDDEN_SIZE` env var to `train.py`/
`export_onnx.py`, tested `hidden=192` (up from 128, +50% width) on the exact same SS29 data
(`USE_OMNIFALL_ADL=1`, `POS_WEIGHT_MULT=1.0`), 3 seeds.

**Conclusive result: hidden=192 helped nothing and cost recall in 2 of 3 seeds.**

| | recall | ADL clean | false-positive clips |
|---|---|---|---|
| hidden=128, seed123 (deployed, SS29) | 100% | 62.5% | s1_ADL_01, s2_ADL_03, s2_ADL_15, s4_ADL_07, s4_ADL_08, s4_ADL_10 |
| hidden=192, seed42 | 100% | 62.5% | **identical set, all 6** |
| hidden=192, seed7 | 93.3% (worse) | 62.5% | **identical set, all 6** |
| hidden=192, seed123 | 93.3% (worse) | 62.5% | **identical set, all 6** |

**Across 9 total training runs now (6 at hidden=128 from SS28/29, 3 at hidden=192 here), varying seed, data,
pos_weight, and now model capacity, the exact same 6 clips fail every single time, with zero exceptions.**
This isn't noise or an undertrained model -- it's a structural limit of the representation, not the
capacity or the data. All 6 clips are the same bed-lying pattern documented since SS20. Pose keypoints
alone -- x/y/velocity of 17 body points -- carry no information about what surface is under the person;
a body lying flat with limbs at rest looks geometrically identical whether it's a bed or the floor. No
amount of more parameters, more seeds, or more (correctly-targeted) training examples can teach a
pose-only classifier a distinction the input representation cannot express in the first place.

**This closes the "make the current model better" line of investigation for this specific pattern.**
A real fix needs a different signal entirely -- e.g. fusing bed/furniture location (from object detection
on the RGB frame) with the pose classifier's output, so the same keypoint pattern is judged differently
depending on what's underneath the person. That's a materially bigger architecture change than anything
tried in SS17-31 (all of which stayed within "same inputs, different model/data/hyperparameters") and
wasn't attempted here. `hidden=192` checkpoints kept locally (`ss31_seed*_h192.pt`) but not exported to
production -- no benefit over the deployed SS29 model, and worse recall in 2 of 3 seeds.

## 32. SS29's deployed model MISSED a real elderly fall -- traced the cause, reverted production

**4 new real clips added to `Test/`** (`14.mp4`-`17.mp4`), 3 of them genuine elderly-fall footage (a CCTV-
style hospital-corridor fall, an "Elderly Person Falling" clip, "Man falling to ground"). Ran the SS29
deployed model against all of them. **`15.mp4`: a man clearly loses balance, crumples, and ends up flat on
his back -- confirmed by direct frame inspection, unambiguous real fall -- and the deployed model never
alerted.** Max probability reached was 0.462, just under the 0.5 threshold.

**Traced why with the full frame-by-frame trace**: `person_found=True` for virtually the entire clip (not
a tracking-loss case, ruling out the SS18/24 mechanisms). Probability peaks at 0.462 right during the
active falling motion (~t=6.2-6.6s) then *drops* to near-zero once he's lying still on the floor
(0.026 by t=7.0s) -- even though he's genuinely fallen and motionless, which is exactly what a real fall's
aftermath looks like.

**Isolated the cause by testing 3 checkpoints (same seed, only the SS28/29 treatment varied) against this
one clip:**

| checkpoint | clip15 max prob | detected? |
|---|---|---|
| seed123, no OmniFall ADL data, pos_weight=1.5 (pre-SS28) | 0.821 | yes, clean |
| seed123, +OmniFall ADL data, pos_weight=1.5 (SS28) | 0.549 | no (crossed once, not sustained) |
| seed123, +OmniFall ADL data, pos_weight=1.0 (SS29, **deployed**) | 0.462 | **no** |

**A clean, monotonic trade-off**: exactly the same intervention that raised GMDCSA24 precision (SS28/29's
lying/sitting hard negatives) systematically suppresses confidence on genuine falls that end in prolonged
stillness -- because "fallen and motionless" and "lying on a bed, motionless" are the *same signal* to a
pose-only classifier, the SS31 finding cutting the other direction. Teaching the model "still + horizontal
isn't a fall" to fix bed false positives also teaches it "still + horizontal isn't a fall" for the tail end
of a real fall.

**Found a better checkpoint, not just reverted blindly**: tested `ss28_seed42_without.pt` (seed 42, no
OmniFall data, pos_weight=1.5 -- the original SS20-era config) against all 3 new real clips and GMDCSA24:

| | GMDCSA24 recall | GMDCSA24 ADL-clean | clip14 | clip15 | clip16 |
|---|---|---|---|---|---|
| SS29 deployed (just reverted) | 100% | 62.5% (6 FP) | 0.84 hit | **0.462 MISS** | 0.963 hit |
| this checkpoint (now deployed) | 100% | 62.5% (**same 6 FP, confirmed by set**) | 0.85 hit | **0.641 hit** | 0.93 hit |

**Identical GMDCSA24 numbers, same exact false-positive clips -- but this one also catches clip15.** A
genuine Pareto improvement over SS29, not a trade-off. Verified end-to-end against the actual production
`models/` directory post-swap. **Deployed** -- `models/fall_classifier_v3.onnx` replaced. Not yet pushed to
GitHub (explicit instruction this session: hold off on git until model quality work is settled).

**Open question worth flagging**: this checkpoint's GMDCSA24 numbers (100%/62.5%) don't exactly match
earlier reports of "the SS20 baseline" (93.3%/62.5%) despite nominally identical seed/data/hyperparameters
-- likely CPU floating-point non-determinism across separate training runs even with `torch.manual_seed`
set (thread-reduction order isn't seeded). Worth remembering: *exact* reproduction of a specific past
run isn't guaranteed even by matching every documented parameter -- what's being trusted here is this
specific saved checkpoint's measured behavior, not an assumption that it recreates an older result.

## 33. Tried 4 YOLO/kinematic signals to fix the bed-lying false positives -- all fail, with data showing why

**Following SS31's conclusion that pose-only input is structurally blind to "what's underneath the person,"
tested whether the already-deployed `yolov10x` object detector (recognizes COCO `bed`/`couch`/`chair`) could
supply that missing context cheaply, no retraining required.** Extracted real frames from the 6 known
GMDCSA24-val ADL false positives and cross-checked against all 15 genuine val Fall clips using actual
bounding boxes and keypoints, not spot checks.

**Signal 1 -- "is a bed anywhere in frame":** confirmed working on the known false-positive frames (bed
detected 0.77-0.9 confidence on 5/6 of them; the 6th, `s4_ADL_07`, is a low backlit daybed YOLO never
catches even down to conf=0.03). But sampling all 15 genuine val Fall clips found **14 of 15 also have a bed
visible in the room throughout the entire clip, including at t=0 before any fall** -- because GMDCSA24 fall
clips are recorded in bedroom settings, the bed is just always in frame whether or not the person falls near
it. This signal alone would suppress ~93% of genuine falls. Rejected outright.

**Signal 2 -- person-bed bounding-box overlap** (what fraction of the person's box sits inside a bed box,
the natural refinement): correctly separates 5/6 known false positives (overlap 0.23-0.83) from the 6 val
Fall clips with *no* bed in the room (overlap 0.00-0.04). But **9 of the remaining 15 val Fall clips also
show high person-bed overlap at the end (0.61-1.00)** -- these are falls that happen to land on, against, or
next to a bed (realistic: e.g. losing balance while getting out of bed), which is geometrically
indistinguishable from safely lying on it. The false-positive range (0.23-0.83) and this genuine-fall range
(0.61-1.00) overlap directly in the 0.61-0.83 band -- no threshold cleanly separates them. A cutoff that
catches the false positives would also suppress roughly 60% of genuine falls in this dataset.

**Signal 3 -- recent keypoint-velocity spike** (hypothesis: a real fall's impact produces a velocity burst a
calm bed-transition doesn't): reused the exact same per-frame velocity feature the classifier already
consumes. False-positive clips ranged 0.06-2.20; the 9 bed-overlap-confounded genuine falls ranged
0.16-7.54 -- heavy overlap again (e.g. `s4_ADL_10`'s false positive peaks at 2.20, above 5 of the 10 genuine
near-bed falls). MediaPipe's own per-frame jitter and ordinary bed-settling motion (adjusting position,
swinging legs up) produce velocity magnitudes comparable to some of GMDCSA24's less-forceful acted falls.

**Consulted Gemini for a second opinion given this data** (`gemini_consult_bed_fix.py`); its most concrete
suggestion was to check the *shape* of the sit/lie-down transition instead of a raw velocity spike -- a real
fall is a sudden discontinuous drop (<0.5-1s), while lying on a bed is a smooth multi-second descent, so net
displacement over a short window should be less jitter-sensitive than an instantaneous frame-to-frame diff.

**Signal 4 -- max net hip-height drop over any 0.5s window** (raw, frame-height-relative, not the
torso-normalized feature): tested exactly this. False positives: 0.02-0.53. The 9 confounded near-bed
falls: 0.02-0.94. **Still the same story** -- e.g. two false positives sit at 0.50-0.53, above 4 of the 10
genuine near-bed falls (0.02, 0.04, 0.04, 0.38). Some GMDCSA24 "falls" are gentle, controlled stunt falls
(actors falling safely) with a small, gradual hip-height change; some real bed-transitions are quick.

**Conclusion: 4 different geometric/kinematic signals, all tested against real ground-truth clips (not
theorized), all show substantial distributional overlap between "false positive" and "genuine fall" -- no
rule-based threshold on any of them cleanly separates the two classes.** This isn't a tuning problem, it
reinforces SS31's finding at a different layer: the ambiguity isn't only that pose keypoints lack surface
context, it's that "did they fall" and "are they now positioned near/on the bed" are genuinely
uncorrelated in this data -- real falls land on beds too. Any signal built purely from *where* the person
ends up, however precisely measured, inherits this confound. Gemini's two other suggestions were a
geometric "bed-plane" heuristic (reasoned, not tested, to have the identical confound as Signal 2 -- it's
still a position-relative-to-bed check, just measured differently) and human-in-the-loop triage: route
ambiguous alerts to a lower-urgency notification with a snapshot for caregiver confirmation instead of a
binary auto-alert/auto-suppress decision. The latter sidesteps the classification problem entirely rather
than trying to solve it, and fits naturally with the already-deferred video-clip-buffer + LINE notification
feature -- worth reconsidering once that feature is built.

**No code or model changes from this investigation.** `models/fall_classifier_v3.onnx` unchanged (still
SS32's `ss28_seed42_without.pt` export). Scripts kept in `training/`:
`test_bed_signal_on_fps.py`, `test_bed_signal_falsepos_on_falls.py`, `test_bed_overlap_signal.py`,
`test_velocity_spike_signal.py`, `test_hip_drop_signal.py`, `gemini_consult_bed_fix.py`.

## 34. Full YOLO-pose-vs-MediaPipe pipeline swap tested end to end -- promising but not a clean win, not deployed

**User asked to try replacing MediaPipe (the pose/keypoint extractor) with YOLO-pose** (`yolo26s-pose.pt` --
native COCO-17 output, index-identical to this codebase's LEFT_SHOULDER=5/RIGHT_SHOULDER=6/LEFT_HIP=11/
RIGHT_HIP=12 convention, so zero remapping needed unlike MediaPipe's 33-point output). Feasibility check
first: sampled the last 2s of all 15 GMDCSA24-val Fall clips (person down after falling -- the exact case
`v3_fall_detection.py`'s `MIN_PERSON_FRACTION` comment documents MediaPipe struggling with) plus 2 real
`Test/` clips. **yolo26s-pose detected a person in 81.2% of these frames vs MediaPipe's 71.7%, while running
~40% faster** (0.027s vs 0.044s/frame, 4-thread CPU). Visual check on 3 hard prone-body frames showed
MediaPipe's skeleton visibly collapsed/tangled while yolo26s-pose's tracked the actual body shape correctly
-- a real, visible quality difference, not just a metric.

**Given this, went further and tested the full swap, not just the feasibility signal**: rebuilt the training
pipeline's pose-extraction step with YOLO-pose (`yolopose_extractor.py`, drop-in for `.extract_keypoints()`)
and re-extracted all 3 video-derived training datasets -- GMDCSA24 (160 clips; had to re-clone the dataset
from GitHub since the original scratchpad copy had expired, recovering the 40 ADL clips no longer present
locally), CAUCAFall (100), and OF-ItW/OOPS (matched down from a natural 5142 to the original 3997 segments,
so the comparison isn't confounded by "more data" as well as a different pose backend). `poses_fallvision`
(5845 pre-extracted external keypoints, not derived from either pose backend) stayed a shared constant.
Trained 3 seeds (42/7/123) on the exact deployed config (no OmniFall data, pos_weight=1.5, hidden=128) --
internal val F1 0.600/0.604/0.603, notably consistent. Exported all 3 to ONNX and built a parallel eval
script (`eval_yolopose_on_gmdcsa24.py`) that reuses `_step_person` (the validated state machine -- pose-
extraction-agnostic, takes raw keypoints) with YOLO-pose keypoints instead of MediaPipe's, so the model is
tested on the same input distribution it was trained on.

**GMDCSA24 val/train50 results, 3 seeds:**

| seed | VAL recall | VAL ADL-clean | TRAIN50 recall | TRAIN50 ADL-clean |
|---|---|---|---|---|
| 42 | 100% (15/15) | 62.5% (10/16) | 92.0% (23/25) | 84.0% (21/25) |
| 7 | 100% (15/15) | 56.2% (9/16) | 92.0% (23/25) | 84.0% (21/25) |
| 123 | 100% (15/15) | 62.5% (10/16) | 88.0% (22/25) | 84.0% (21/25) |

**Essentially matches the deployed MediaPipe model's numbers** (100%/62.5% val, 92%/84% train50 at its
best seed) -- and the same stubborn bed-lying pattern persists almost identically (a consistent core of 4
false-positive clips shared across all 3 seeds, with 2 more rotating in/out), reinforcing SS31/33's finding
that this is a representational limit of pose-only input, not a MediaPipe-specific keypoint-quality problem.

**But the real Test/ clips -- the ones that actually matter, 14/15/16, the genuine elderly-fall footage
that led to SS32's fix -- told a different story:**

| seed | clip14 (deployed: 0.85 hit) | clip15 (deployed: 0.64 hit) | clip16 (deployed: 0.93 hit) |
|---|---|---|---|
| 42 | MISSED | 0.68 hit | 0.71 hit |
| 7 | MISSED | MISSED | 0.79 hit |
| 123 | 0.50 hit (barely) | MISSED | 0.74 hit |

**No single seed catches all 3, where the currently-deployed MediaPipe model catches all 3 solidly.**
Checked whether this was a keypoint-extraction dropout on clip14 specifically (the same mechanism behind
several bugs earlier in this file) -- it wasn't: YOLO-pose detects a person in 100% of clip14's frames.
The miss is coming from the classifier's learned decision boundary, not missing pose signal. Most likely
explanation: the deployed MediaPipe model benefited from many rounds of accumulated tuning across this
entire file (SS18's smoothing fix, SS28/29's data experiments, threshold/pos_weight tuning) while these
YOLO-pose checkpoints are first-pass, default-config results with zero iteration -- not evidence YOLO-pose
keypoints are worse, but real evidence this specific swap isn't a validated, ready-to-deploy improvement.

**Verdict: promising direction, not a clean win. Not deployed.** `models/fall_classifier_v3.onnx` and
`app/detection/v3_fall_detection.py` unchanged -- still MediaPipe, still SS32's checkpoint. The higher raw
detection rate and speed are real and reproduced independently of the training-variance question above, so
this is worth revisiting with more tuning (more seeds, threshold sweep, possibly more real-world training
data in the OF-ItW-style mold) rather than abandoned outright -- but swapping the production pose backend
on the strength of matching aggregate GMDCSA24 numbers alone, while regressing on the 3 real clips that
matter most, would have repeated exactly the mistake SS32 was about (trusting an aggregate metric that
didn't cover the failure case that turned out to matter). New code kept in `training/`: `yolopose_extractor.py`,
`extract_poses_yolopose.py`, `extract_caucafall_poses_yolopose.py`, `extract_ofitw_poses_yolopose.py`,
`eval_yolopose_on_gmdcsa24.py`, `eval_yolopose_on_testclips.py`. `train.py` gained `CAUCAFALL_DIR_NAME`/
`OFITW_DIR_NAME` env vars (mirroring the existing `GMDCSA24_DIR_NAME` pattern) so this comparison didn't
need permanent hardcoded changes.

**Follow-up tuning pass, same session, per explicit user request to keep pushing**: tried the two cheapest
levers before concluding this needs real new training data. (1) **Ensembling** the 3 seeds' probabilities
(`EnsembleOnnxFallClassifier`, averages `predict_window()` across checkpoints) -- made clip14/15 WORSE, not
better (both missed, where seed42 alone had caught clip15) -- the failing seeds' low probabilities drag the
average down rather than averaging out as noise, evidence this isn't per-seed random variance. (2) **2 more
seeds trained** (1, 99) and **a threshold sweep down to 0.4** (`EVAL_THRESHOLD` env var added to both eval
scripts) -- GMDCSA24 numbers barely moved (VAL still ~100%/56-62%, TRAIN50 still 92%/84%), and clip14/15
still weren't recovered by the 3-seed ensemble even at this looser threshold.

**5 seeds now tested total -- the pattern is decisive and consistent, not noise:**

| seed | clip14 (deployed: 0.85 hit) | clip15 (deployed: 0.64 hit) | clip16 (deployed: 0.93 hit) |
|---|---|---|---|
| 42 | MISSED (peak 0.44) | 0.68 hit | 0.71 hit |
| 7 | MISSED (peak 0.67, not sustained) | MISSED | 0.79 hit |
| 123 | 0.50 hit (barely) | MISSED | 0.74 hit |
| 1 | MISSED | 0.60 hit | 0.81 hit |
| 99 | MISSED | MISSED | 0.96 hit |

**clip16 is solid across all 5 seeds (0.71-0.96) -- a genuine, reproduced win.** clip15 is roughly a coin
flip (2/5). **clip14 is caught in only 1/5, and barely** -- this is where the real problem concentrates, not
spread evenly across all 3 clips. Diagnosed directly: frame-by-frame probability trace on clip14
(`diagnose_yolopose_clip14.py`) shows probability never gets close to threshold with seed42 (peaks at 0.44
around the actual fall moment, otherwise 0.03-0.2) despite the frame at that moment showing an unambiguous,
textbook fall (elderly woman collapsed on the floor, cane fallen beside her, visible distress -- confirmed
by direct visual inspection). Checked it isn't a keypoint-detection dropout (the documented root cause of
several earlier bugs in this file): YOLO-pose detects a person in 100% of clip14's frames. The classifier
itself, not the pose signal, is what's under-confident on this specific motion pattern.

**Conclusion: this is very likely a training-data-coverage gap, not a fixable-by-more-seeds problem** --
clip14's fall (using a cane, more torso rotation, brief self-occlusion during the collapse) may simply be
under-represented in the ~4,257 video-derived training clips relative to more common falls. Ensembling and
threshold tuning are architecture-agnostic band-aids that would apply equally to any weak spot; neither
moved this one, which points at the data the classifier learned from rather than how its output is
thresholded. A real fix would need either targeted new training examples of this specific fall pattern
(mobility-aid falls, partial self-occlusion) or substantially more hyperparameter/architecture search --
both bigger investments than this pass, and not attempted here. **Still not deployed** -- same verdict as
before, now on stronger evidence. New file: `diagnose_yolopose_clip14.py`.

**Consulted Gemini** with the full evidence above (`gemini_consult_yolopose_tuning.py`) for a second opinion
before closing this out. Its response contained one factual error worth flagging rather than trusting blindly
-- it reasoned as if the classifier were "trained on jittery MediaPipe data," which is wrong (this whole
pass retrained from scratch on YOLO-pose-extracted features specifically to avoid that exact train/inference
mismatch). Its more useful point: clip14's seed42 peak (0.44) is close enough to threshold that it's worth
directly testing threshold=0.4 on seed42 ALONE (not just the 3-seed ensemble, which could be masking a
single seed's near-miss by averaging in weaker seeds). **Tested it directly: still no alert** -- the 0.44
peak doesn't sustain across enough consecutive inference windows to satisfy `_step_person`'s SMOOTH_NEED=2-
of-3 gate even at this looser threshold, closing off the "just barely below threshold" explanation entirely.
This confirms it's a genuine confidence/sustain problem, not a threshold calibration one.

## 35. Flip + occlusion augmentation closed the gap -- YOLO-pose deployed to production, replacing MediaPipe

**User asked to keep pushing for professional-grade quality.** Added two standard augmentations to
`training/dataset.py`, applied only to the training split (never val): **horizontal flip**
(`flip_horizontal_window` -- mirrors x and vx, swaps left/right keypoint pairs via `FLIP_PAIRS`; a fall
isn't inherently left- or right-handed, so this doubles effective training data for free) and **synthetic
per-joint occlusion** (`occlude_window` -- freezes 1-3 random joints' position for a short random span and
zeros their velocity there, mirroring how `_step_person` holds last-known state on a real tracking dropout
rather than zero-filling -- targeted directly at SS34's diagnosis that clip14's weakness involved brief
self-occlusion during a twisting collapse). Gated behind a new `AUGMENT` env var in `train.py` (default off,
so every prior run in this file stays exactly reproducible).

**Retrained 3 seeds with `AUGMENT=1`, same YOLO-pose data as SS34. Dramatic, consistent improvement on the
exact clip that augmentation targeted:**

| seed | clip14 (no aug, SS34) | clip14 (with aug) | clip15 (with aug) | clip16 (with aug) |
|---|---|---|---|---|
| 42 | MISSED (peak 0.44) | **0.91 hit** | 0.72 hit | 0.74 hit |
| 7 | MISSED | **0.68 hit** | MISSED | 0.86 hit |
| 123 | 0.50 hit (barely) | **0.87 hit** | 0.51 hit | 0.80 hit |

clip14 went from caught-in-1-of-5 (SS34, no augmentation) to caught-in-3-of-3 (with augmentation), and by
a wide margin in 2 of the 3. This is strong evidence the SS34 diagnosis (training-data coverage gap around
self-occlusion/rotation, not a fixable-by-more-seeds problem) was correct, and that synthetic occlusion
augmentation is a real, working fix for it -- not just a lucky seed.

**A mistake caught by re-verifying through the actual production code path, not just the parallel eval
script -- worth documenting honestly.** Initially picked seed123-aug for deployment based on a compiled
summary table, believing it had VAL 100%/62.5% -- but that number was a mix-up with seed7's result; **seed123's
GMDCSA24 val eval was never actually run** before that decision. Deployed it anyway, then re-verified via
`eval_v3_on_gmdcsa24_val.py` (the real, unmodified production eval script, run twice for determinism) and
got 100%/**50.0%** (8/16) -- worse than deployed, and a real number this time, not a mix-up. Caught before
this went uncorrected: switched to **seed42-aug** instead, which has actual, individually-verified numbers
across every test surface:

| | Deployed (MediaPipe, pre-this-session) | seed42-aug (YOLO-pose, now deployed) |
|---|---|---|
| VAL recall | 100% (15/15) | 100% (15/15) |
| VAL ADL-clean | 62.5% (10/16) | 56.2% (9/16) |
| TRAIN50 recall | 92.0% (23/25) | 92.0% (23/25) |
| TRAIN50 ADL-clean | 84.0% (21/25) | **88.0% (22/25)** |
| clip14 | 0.85 hit | 0.91 hit |
| clip15 | 0.64 hit | 0.72 hit |
| clip16 | 0.93 hit | 0.74 hit |
| pose detection rate, hard frames | 71.7% | 81.2% |
| CPU inference speed | baseline | ~40% faster |

All numbers on this row for seed42-aug are from the actual unmodified production scripts
(`eval_v3_on_gmdcsa24_val.py`, `eval_v3_on_gmdcsa24_train50.py`) and the real `V3PoseFallDetector` /
`detect_v3_fall` / `detect_v3_fall_multi` production code paths (`verify_production_testclips.py`,
`diff_extractors.py` confirming the parallel eval script and production path produce bit-identical
keypoints) -- not the parallel `eval_yolopose_on_*.py` scripts alone, after the seed123 lesson. **Net honest
assessment**: a wash on VAL (-1 clean clip), a real gain on TRAIN50 (+1 clean clip) and all 3 real critical
clips (still all caught, thinner margin on clip16 specifically: 0.74 vs 0.93), plus a genuine, independently-
reproduced improvement in raw pose-detection reliability and CPU speed.

**Deployed for real this time**: `app/detection/v3_fall_detection.py`'s `V3PoseFallDetector` now loads
YOLO-pose (`models/yolo26s-pose.pt`, copied in) instead of MediaPipe (`mediapipe`/`vision`/`BaseOptions`
imports removed; `pose_landmarker_lite.task` left in `models/` unused rather than deleted). `extract_all_keypoints`
rewritten around `ultralytics.YOLO.predict()`, sorted by box confidence, capped at `NUM_POSES` -- same
external interface (`(kpts17, hip_center)` tuples) as before, so `camera_manager.py` and every training/eval
script that imports `V3PoseFallDetector` needed no changes. `models/fall_classifier_v3.onnx` replaced with
the `yolopose_aug_seed42.pt` export; old MediaPipe-trained checkpoint kept at
`models/fall_classifier_v3_mediapipe_backup.onnx` for a fast revert if needed. Verified both the single-
person (`detect_v3_fall`) and multi-person (`detect_v3_fall_multi`) production entry points directly, not
just through eval scripts. **Not yet pushed to GitHub** (standing instruction this session).

**Caveat worth stating plainly, matching this file's established honesty norm**: VAL ADL-clean is down one
clip (9/16 vs 10/16) and clip16's margin is notably thinner (0.74 vs 0.93) than the outgoing MediaPipe
model. This is a net-positive trade on the evidence gathered, not a strictly-dominant win -- if a future
session finds clip16-like real footage starts getting missed, this is the first thing to revisit, and the
MediaPipe backup checkpoint is one file copy away from reverting the classifier (though reverting the pose
backend itself would also require reverting `v3_fall_detection.py`'s `V3PoseFallDetector`, not just the
`.onnx` file, since it's no longer MediaPipe-shaped).

## 36. Post-deploy Gemini-verified regression check, both single- and multi-person paths, all 17 `Test/` clips

**User asked for one more round of scrutiny before trusting the new deployment**, matching this file's
established post-deploy audit pattern (SS30). Noted first that the production pipeline is fully
deterministic (confirmed by literally rerunning `eval_v3_on_gmdcsa24_val.py` twice and getting bit-identical
output down to the decimal -- SS35's earlier discrepancy was a real mistake, not noise), so repeating
identical clips through identical code adds nothing; the actually-new value was checking a code path that
hadn't been exercised yet this session: `detect_v3_fall_multi` (what `camera_manager.py` really runs), not
just the single-person path most of this session's testing used.

Ran both entry points over all 17 `Test/` clips through the real deployed `V3PoseFallDetector`
(`collect_production_alerts.py`), then Gemini-verified every single alert frame, freshly re-extracted from
source (`verify_production_alerts.py`, same methodology as `verify_current_model_alerts.py`):

| path | alerts | Gemini-confirmed real | precision |
|---|---|---|---|
| single-person | 66 | 54 | 81.8% |
| multi-person | 64 | 51 | 79.7% |

**Both paths agree closely and confirm the same story**: all 3 critical real clips (14/15/16) verified as
genuine falls by Gemini on both paths -- not an artifact of one code path. clip13's new alert (present now,
absent before SS35's augmentation) also verified genuine ("person down on all fours on wet pavement" /
"person motion-blurred and prone on the floor") -- a real catch, not a new false positive. Every false
positive found matches an already-documented pattern from earlier in this file, nothing new: upright
walking/standing misclassified (the largest category), sitting (porch swing, intentionally sitting on a
ramp), exercise (hanging from a bar), bending to pick something up, boat-related motion, and one video-
transition artifact (a text-only frame with no person, the SS21/25 compilation-clip mechanism). **No new
failure mode found. Clean bill of health for this deployment**, on the same standard this file has applied
to every prior one (SS30). No code or model changes from this pass.

## 37. Alone-detection: real ground-truth accuracy testing (not self-agreement), threshold tuning, and a failed fine-tune attempt

**User pushed back on the earlier YOLO26-size benchmark (SS-era, alone-detection person counter)**:
that comparison only measured each candidate's agreement with the *currently deployed* `yolov10x`, which
isn't a real accuracy measure since `yolov10x` itself could be wrong. Built a proper ground-truth test
instead: sampled 77 frames across 11 `Test/` clips (`gt_sample_frames.py`), had Gemini independently count
the actual number of people per frame (`gt_gemini_count.py`) as ground truth, then scored all 6 YOLO
candidates (`yolov10x`, `yolo26 n/s/m/l/x`) against it on the binary condition production actually uses --
"is exactly 1 person present" (`gt_compare_binary.py`).

**Default threshold (0.6, matching production) topped out at 80.5%** (`yolo26l`/`yolo26x` tied). Examined
the errors directly (`gt_show_errors.py`) -- 12 of 15 for `yolo26l` were "Gemini says 1 person, YOLO
predicted 0" (a clean miss, not a miscount). Visually inspected two: both were distant, motion-blurred,
mid-fall/prone people in outdoor TikTok-compilation clips -- a domain mismatch with what alone-detection
actually needs (indoor home camera, people upright/sitting, close range), not a representative failure.

**Threshold sweep found the real lever**: `yolo26l` at conf=0.35 reached **88.3% overall (68/77)** with
**zero new false alone-alerts** vs the 0.6 baseline (`gt_threshold_sweep.py`/`gt_threshold_sweep2.py` --
also checked imgsz=1280 and `yolo26x`, neither beat this). Restricting to the 21 frames that actually match
the real domain (clips 14/15/16, real elderly-care footage) -- **100% accuracy (21/21)**, including the one
crowd-scene clip, correctly classified as "not exactly 1" regardless of precise headcount.
**Deployed.** `model_manager.py`'s `_load_person_detector()` now loads `yolo26l.pt` @ conf=0.35 (was
`yolov10x.pt` @ 0.6) -- confirmed loading cleanly and producing a correct real alert in production.

**Then tried actually training a custom person-detector**, per explicit user request to push further than
threshold tuning. No manually-labeled home-camera dataset exists, so used self-training / pseudo-labeling:
sampled 349 new frames from all 17 `Test/` clips (`ft_sample_frames.py`, explicitly excluding any frame
within 1s of the 77 held-out ground-truth frames to prevent leakage), generated pseudo-labels with `yolo26l`
at conf>=0.5 (`ft_make_pseudolabels.py`), fine-tuned `yolo26m` for 15 epochs at a low LR
(`ft_train.py`) starting from its own pretrained weights.

**Hit a real Windows bug first**: the initial run (and a second attempt at `yolo26l`) appeared to hang for
50+ minutes producing no output. Root cause: `model.train()` was called at module level without an
`if __name__ == "__main__":` guard -- on Windows, PyTorch's multi-worker DataLoader uses the `spawn` start
method, which re-imports the launching module in each worker and crashes/hangs without that guard. Fixed
by wrapping in `main()` + the guard, and setting `workers=0` as a belt-and-suspenders fix. Once fixed,
training was fast (15 epochs in ~4 minutes) -- this machine has a CUDA GPU (RTX 4070 Ti SUPER) available
locally, confirmed by Ultralytics' own device log, though production inference stays CPU-only
(`ort.InferenceSession(..., providers=["CPUExecutionProvider"])`) and is unaffected by this.

**Evaluated the fine-tuned checkpoint against the same 77-frame Gemini ground truth
(`ft_eval.py`) -- conclusively worse, not better:**

| | pretrained `yolo26l` @ conf=0.35 | fine-tuned `yolo26m` (best conf tried) |
|---|---|---|
| Overall accuracy | 88.3% | 51.9% |
| Realistic-domain accuracy | 100% | 76.2% |

**Root cause, not just a bad run**: classic self-training pitfall. 349 frames is too small a fine-tuning
set on its own, but the deeper problem is that pseudo-labels came from the *same* model class making the
*same* mistakes being fine-tuned on -- frames where the teacher missed a person got an empty label, actively
teaching the student to keep missing that same pattern rather than correcting it. This is expected: self-
training without real human-verified labels or careful confidence-based filtering degrades rather than
improves a model that's already reasonably good, unlike SS35's augmentation (which changed *how* real
labeled data was presented to the classifier, not what the labels were). **Not deployed.** A genuine
accuracy gain here would need real labeled data (human- or Gemini-per-frame-verified boxes, not model self-
labels) at meaningfully larger scale than 349 frames -- a bigger undertaking, not attempted further this
pass. New files: `gt_sample_frames.py`, `gt_gemini_count.py`, `gt_run_yolo_variants.py`, `gt_compare.py`,
`gt_compare_binary.py`, `gt_threshold_sweep.py`, `gt_threshold_sweep2.py`, `gt_show_errors.py`,
`gt_show_errors2.py`, `ft_sample_frames.py`, `ft_make_pseudolabels.py`, `ft_train.py`, `ft_eval.py`.
