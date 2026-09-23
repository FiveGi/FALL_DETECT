# The scripts every number in SKILL.md came from

These lived in a session scratchpad, outside the repository, while SKILL.md and `docs/` cited
them by name. Anyone who cloned this could read the claim and not the thing that produced it.
They are here now so a result can be reproduced or argued with.

They write their per-clip output as JSON, and most of them **resume**: a full sweep is 220
clips and a machine going down mid-run has cost this project real time more than once.

## Where the results go

Every script writes into whatever `RESULTS_DIR` points at, defaulting to this directory.
`perclip/*.json` holds one file per measured configuration, and the workbook generator
(`training/build_comparison_xlsx.py`) reads them.

## The measurement scripts

| Script | What it answers |
|---|---|
| `rule_sweep_perclip.py` | **The main one.** One (model, input size, alerting rule, frame rate) over all 220 lab clips, clip by clip. Every accuracy number in SKILL.md from SS51 onwards came from this. |
| `urfd_window_fill.py` | Which URFD clips a given (window, frame rate) can score *at all*. Runs the pose pass once per frame and answers every pair by arithmetic afterwards. |
| `testclips_current.py` | The 17 `Test/` clips through the deployed profiles, with peak scores. |
| `original_no_collapse.py` | The original detector with its "the person vanished" rule disabled, which is how SS66 showed that rule contributed nothing to its recall. |

## The comparison scripts

| Script | What it answers |
|---|---|
| `bench_original_vs_deployed_speed.py` | Time per frame for the original and the current detector on the same 1080p frames. |
| `diff_s_vs_n_pose.py` | Whether two pose models actually produce different keypoints, or only different speeds. |
| `person_found_s_vs_n.py` | How often each pose model finds the person at all — which is what separated them, not keypoint accuracy. |
| `test_camera_motion.py` | Background optical flow per clip, i.e. is the camera fixed. This is how `Test/13` was shown to be a moving, zooming shot. |
| `gemini_verify_test_clips.py` | A second opinion on what is in a clip. Needs `GEMINI_API_KEY` in `.env`; the free tier is 20 requests a day. |

## Test fixtures and verification helpers

| Script | What it is for |
|---|---|
| `mjpeg_server.py` | Serves a clip as MJPEG over HTTP inside the worker container, so a stream can be taken away and given back. The reconnect fix was tested against this. |
| `recalc_excel.ps1` | Recalculates a workbook with Excel and reports formula errors. The xlsx skill ships a LibreOffice version that cannot run on Windows (`socket.AF_UNIX`). |
| `export_sheet_png.ps1` | Screenshots a worksheet range through Excel, so a sheet's layout can be looked at rather than assumed. |

## Two rules these exist to enforce

**Say which frame rate.** The classifier's window is a fixed number of frames, so the same
model scores very differently at different rates. A number without its rate means nothing here.

**Choose on one half of URFD, confirm on the other.** The split takes sequences two at a time
(`((index - 1) // 2) % 2`) — *not* odd and even, because URFD alternates its fall types by
sequence number and an odd/even split compares two different tasks (SKILL.md SS60).
