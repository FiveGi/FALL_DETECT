# Test clips

17 real-world video clips used throughout this project's testing and validation (see
`SKILL.md` and `report.md` for the numbers they produce). Referenced by scripts via the
`TEST_DIR` environment variable (defaults to `D:\project\PROJECT\Test`; set it to this
folder's path to run those scripts here).

Provenance is mixed:

- **Clips 1-12**: multi-scene compilation-style footage (doorbell/security-camera fall
  and near-fall clips), originally sourced from social media. Some of this content is
  third-party copyrighted material, not owned by this project -- see
  `training/sample_results/clip1_summary.txt` for the earlier note on this before these
  clips were added to the repo. Included here for reproducibility at the project owner's
  explicit request; treat accordingly if you redistribute this repository further.
- **Clips 13-16**: real fall footage added later in the project specifically to validate
  against genuine falls the earlier synthetic/lab datasets (GMDCSA24, CAUCAFall) didn't
  cover well (e.g. mobility-aid falls, hospital-corridor settings). One fall each,
  confirmed by watching them and by a second opinion from Gemini.
- **Clip 17**: **no fall happens in it.** It is a group doing an outdoor exercise routine,
  filmed from across a courtyard. It was grouped with 13-16 for some time and reported as a
  fall every configuration missed, which made the detector look worse than it is. The
  correct result on this clip is **no alert**, and it is a useful negative precisely
  because a crowd repeatedly dropping into deep squats is the kind of motion that invites
  a false alarm.
- **Clips 10 and 11** contain a fall each (a toddler down a staircase at ~3s, a child off
  a bunk bed at ~11s), and the detector misses both. Both are phone re-uploads: small
  subjects, heavy letterboxing, sticker overlays, night vision in 11, and several seconds
  of end card. They are out of the domain this system targets rather than evidence about
  it, and are recorded here so nobody re-investigates them a third time.
