# Old vs new detection settings

Same trained model in both columns -- `models/fall_classifier_v3.onnx`
(md5 `194614047877dc8e9ff896e5331170f7`) on the `yolo26s-pose` backbone. What
changed is the inference configuration, both of which were previously unmeasured
library defaults:

| setting | old | new |
|---|---|---|
| pose input size (`V3_IMGSZ`) | 640 | **960** |
| pose confidence (`V3_POSE_CONF`) | 0.50 | **0.30** |

Every number below comes from the multi-person entry point
(`detect_v3_fall_multi`), the one the camera loop actually runs.

## Results

| what is measured | old | new |
|---|---|---|
| GMDCSA24 val - falls caught | 15/15 (100%) | 15/15 (100%) |
| GMDCSA24 val - normal clips with no false alarm | 9/16 (56%) | 10/16 (62%) |
| train50 - falls caught | 22/25 (88%) | 24/25 (96%) |
| train50 - normal clips with no false alarm | 22/25 (88%) | 22/25 (88%) |
| two people, one falls - fall caught | 12/15 (80%) | 13/15 (87%) |
| two people, nobody falls - no false alarm | 4/4 (100%) | 4/4 (100%) |
| small-person control - fall caught | 10/15 (67%) | 14/15 (93%) |
| speed on GPU | 30.5 ms/frame (32.7 fps) | 37.5 ms/frame (26.6 fps) |

## Reading this

A camera needs about 25 fps to be followed without dropping frames, so both
columns are fast enough on the GPU and the new column costs a few ms.

The two-person and control rows exist because every other dataset here has
exactly one person in frame, which is not the situation this system is for.
The control is the same fall in the same widened frame with *nobody* beside it:
it separates "a bystander broke detection" from "the person is now rendered at
half the pixels", which the composite changes at the same time.

Reproduce with `python training/compare_old_vs_new.py` (needs the GMDCSA24 clips
in `training/data/` and the composites from
`python training/make_multiperson_testset.py`).
