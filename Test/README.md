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
- **Clips 13-17**: real elderly-fall footage added later in the project specifically to
  validate against genuine falls the earlier synthetic/lab datasets (GMDCSA24, CAUCAFall)
  didn't cover well (e.g. mobility-aid falls, hospital-corridor settings).
