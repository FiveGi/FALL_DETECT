# DAY3-MEDIA — Codex, 2026-10-05

Owner-assigned illustrations for the sibling `../report/report_day3_outline.md`.
No HTML built. No training, deployment, cache jobs, or existing processes changed.
New media uses CPU inference with one PyTorch/ONNX thread; final clip command printed
`pose backend on cpu, 1 CPU thread(s)` for both models and `torch threads 1`.
Claude review is pending before publishing these new renderer's illustrations/numbers.

## Outputs (staged inside repository)

| File | Content | Evidence |
|---|---|---|
| `report/img/day3_t2_paired_deltas.png` | 3 matched seeds, 4 count deltas, threshold 0.65 | [chart.json](chart.json) |
| `report/img/day3_coco_false_person_grid.jpg` | 6 COCO val examples, nightaug_s44 detections and posneg_s44 zero detections | [grid.json](grid.json) |
| `report/videos/day3_compare_near.mp4` | Test/7.mp4#2, source 5.97–9.93 s, 32 frames / 4.0 s playback | [clips.json](clips.json) |
| `report/videos/day3_compare_multi.mp4` | Test/1.mp4#15, source 51.77–57.23 s, 44 frames / 5.5 s playback | [clips.json](clips.json) |

All output hashes, dimensions, decoded counts and codec checks: [verification.json](verification.json).
Sampling rounds onto source frames/8-fps slots, hence output duration differs slightly from source interval.

## Commands actually run

From the repository root, using installed Python 3.11:

```powershell
python tools/day3_media.py chart
python tools/day3_media.py grid
python tools/day3_media.py clips
python tools/day3_media.py verify
```

The final commands completed with exit 0. Final clip run explicitly uses Windows Media Foundation
H.264 encoding. The first run's automatic encoder selection printed missing OpenH264 errors,
then fell back successfully; `clips.log` belongs to that first attempt. Re-rendered both clips
using explicit `CAP_MSMF`, without these errors, and decoded every final frame again.
The verification JSON, not that initial log, identifies the final files.

## Provenance and limits

- Chart implementation: `tools/day3_media.py:46`. Dataset: owner's 75 fall segments,
  25 multi-person falls, 25 frozen near-bucket falls; D3: 14 scored faller-labelled segments
  (3 excluded from the 17-segment manifest). Each metric is a mean over 8 crop phases, not
  independent additional clips. Seeds 45/46/47 are matched to their own t1_ctrl.
  Both sides use nightaug_s44 at **0.65**; this chart is not the production-vs-finalist comparison.
  Exact row/line references and all raw means/deltas are in `chart.json`.
  Pinned source lines: `results_pinned.jsonl:22,24,26,35,48,50`;
  D3 sources `track_metric_t2.txt:2,3,4` and `track_metric_step1b.txt:4,5,6`.
  Near counts were recomputed from all 48 corresponding owner phase JSON files using
  `owner_size_buckets_v4.json`; they are absent from the pinned summary rows.
  Exact deltas, ordered owner/multi/near/D3:
  s45 `[6,2.75,3.125,2.125]`, s46 `[7.5,4.5,2,4.625]`, s47 `[3.875,2,1,0.625]`.
  Bar labels round to 2 decimals; evidence retains exact eighths. Gains do not imply all gates pass.
- Grid implementation: `tools/day3_media.py:109`; imports the actual
  `training/measure/object_false_person.py:26-36` test-set function and uses the same
  full-frame CPU inference settings as its line 48: imgsz 320, conf 0.30, classes=[0].
  COCO `instances_val2017.json`: person-free means **no person annotation**, with an indoor-clutter
  category present and image locally available. There are 1,176 eligible images. Scanned 66
  in sorted image-ID order, selected the first 6 where nightaug has >=1 detection and posneg has 0.
  Posneg was only run on nightaug-positive images. This is selected illustration, not a rerun
  of the whole 1,176-image aggregate. IDs, boxes, confidence values and model hashes in `grid.json`.
  Viewed the complete grid: food, cats/dog, shoes, and bathrooms; no obvious real people visible.
- Clips implementation: `tools/day3_media.py:159`; **fresh CPU inference**, not painted cached alerts.
  Uses `cache_owner_segments.segment_frames` (`training/measure/cache_owner_segments.py:58`)
  and the actual `detect_v3_fall_multi` pipeline. Stock pose + `models/` at 0.65 vs
  `nightaug_s44_p2/weights/best.pt` + `t2_truncfull_s45` at 0.70. Profile:
  full-frame 320, ROI 256 every 8, pose conf 0.30, preprocess auto, 8 fps, ROI phase 0,
  separate fresh track/crop states at segment start; no audio. Model/source/pipeline hashes in `clips.json`.
  Two selected gains, **not representative performance**, no held-out Le2i inference.
  Near identity: `owner_size_buckets_v4.json:237`; multi identity comes from
  `../.ai_evidence/multi_person_genuine.txt`; start/end from `test_result/incidents/incidents.json`.
  Every alert timestamp on both sides exactly equals the respective pinned phase-0 JSON:
  production `parity_deployed_pinned_owner0.json`, candidate `T2_truncfull_s45_rule_pinned_owner0.json`.
  Skeletons yellow; red circles identify actual alert-track centroids, which can be stale if the
  detector's disappearance rule fires. These are detector outputs, not hand-labelled faller identity.
- Verification implementation: `tools/day3_media.py:263`. Both H.264 files decoded in full,
  32+44 frames, 960x1022, 8 fps; both image files decoded. Visually inspected source preview,
  both complete still images and each clip's first-alert frame (near frame 23, multi frame 35).
  Did not play the clips in a browser or independently remeasure aggregate benchmarks.
  No disagreement with the agreed finalist choice is asserted; chart uses unrounded source arithmetic.

## Remaining placement action for Claude

This session's writable root is the repository only. The intended sibling report directory
is outside that root; no attempt was made to bypass the filesystem restriction.
Copy these four finished assets using a session with write access to `D:/project/PROJECT/report`:

```powershell
Copy-Item -LiteralPath report/img/day3_t2_paired_deltas.png -Destination ../report/img/
Copy-Item -LiteralPath report/img/day3_coco_false_person_grid.jpg -Destination ../report/img/
Copy-Item -LiteralPath report/videos/day3_compare_near.mp4 -Destination ../report/videos/
Copy-Item -LiteralPath report/videos/day3_compare_multi.mp4 -Destination ../report/videos/
```

These copy commands were **not run**. Review script/figures, compare copied hashes with
`verification.json`, retain selection/threshold captions, and wait for t3b before building HTML.
