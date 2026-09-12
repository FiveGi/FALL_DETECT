"""Build a two-person test set with real ground truth, by pairing GMDCSA24 clips.

The multi-person path is what production runs, but nothing here has ever tested it on footage
that actually contains two people AND has ground truth. GMDCSA24 is all single-person;
Test/ has multiple people but no labels (SS36 had to have Gemini verify each alert one by
one). So a claim like "multi-person mode is less accurate" has had nothing to measure
against.

This composites one Fall clip beside one ADL clip into a single frame. Both halves are real
footage with known labels, so the composite has known labels too:

  fall+adl  -> exactly one person falls. The detector must alert, and the ADL person must
               not add a second alert. Tests whether a bystander breaks detection of the
               person who fell -- the case that matters in a care home.
  adl+adl   -> nobody falls. Any alert is wrong. Tests whether two people who are both fine
               produce alerts that neither would produce alone.

Side-by-side rather than alpha-blended: blending creates ghost limbs that are not a real
pose, so a detector failing on them would prove nothing. Two real people in two halves of
one frame is geometrically honest -- each is a real person at a real scale, and the tracker
has to keep them apart, which is exactly the mechanism under test.

Usage:
    python training/make_multiperson_testset.py
"""
import os
import random

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FALL_DIR = os.path.join(ROOT, 'training', 'data', 'gmdcsa24_fall_raw')
ADL_DIR = os.path.join(ROOT, 'training', 'data', 'gmdcsa24_adl_raw_val')
OUT_DIR = os.path.join(ROOT, 'training', 'data', 'multiperson_composite')

# The same held-out clips every other eval in this project uses, so results are comparable.
VAL_FALL = ["s1_Fall_02", "s1_Fall_06", "s1_Fall_16", "s2_Fall_02", "s2_Fall_04",
            "s2_Fall_09", "s2_Fall_14", "s2_Fall_20", "s3_Fall_02", "s3_Fall_09",
            "s3_Fall_12", "s3_Fall_13", "s3_Fall_16", "s4_Fall_06", "s4_Fall_17"]
VAL_ADL = ["s1_ADL_01", "s1_ADL_05", "s1_ADL_11", "s1_ADL_13", "s2_ADL_03", "s2_ADL_07",
           "s2_ADL_13", "s2_ADL_15", "s2_ADL_16", "s2_ADL_18", "s2_ADL_20", "s3_ADL_07",
           "s3_ADL_11", "s4_ADL_07", "s4_ADL_08", "s4_ADL_10"]

# ADL clips that already false-alarm on their own (measured, see
# training/data/multi_diagnosis_gmdcsa.json). Excluded from the ADL side so a composite's
# alert can be attributed to the pairing rather than to a false positive that would have
# happened anyway.
NOISY_ADL = {"s1_ADL_01", "s2_ADL_03", "s2_ADL_16", "s2_ADL_20", "s4_ADL_07", "s4_ADL_08",
             "s4_ADL_10"}


def side_by_side(path_a, path_b, out_path):
    """Write a clip with path_a's frames on the left and path_b's on the right.

    The shorter clip freezes on its last frame rather than looping: a loop would restart a
    person mid-motion, which reads as a teleport to the tracker and would inject exactly the
    track-churn artefact this test is supposed to measure honestly.
    """
    cap_a, cap_b = cv2.VideoCapture(path_a), cv2.VideoCapture(path_b)
    fps = cap_a.get(cv2.CAP_PROP_FPS) or 30.0
    ok_a, frame_a = cap_a.read()
    ok_b, frame_b = cap_b.read()
    if not ok_a or not ok_b:
        cap_a.release(); cap_b.release()
        return False

    h = min(frame_a.shape[0], frame_b.shape[0])

    def fit(f):
        scale = h / f.shape[0]
        return cv2.resize(f, (int(f.shape[1] * scale), h))

    left, right = fit(frame_a), fit(frame_b)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (left.shape[1] + right.shape[1], h))
    if not writer.isOpened():
        cap_a.release(); cap_b.release()
        return False

    n = 0
    while ok_a or ok_b:
        writer.write(cv2.hconcat([left, right]))
        n += 1
        ok_a, f = cap_a.read()
        if ok_a:
            left = fit(f)
        ok_b, f = cap_b.read()
        if ok_b:
            right = fit(f)
        if n > 3000:
            break
    writer.release()
    cap_a.release(); cap_b.release()
    return n > 0


def blank_beside(path_a, out_path):
    """Control: the fall clip in the same 2x-wide frame, but the other half black.

    Without this, a fall missed in a composite could just as easily be caused by the frame
    being twice as wide -- YOLO resizes the whole frame to a fixed input, so each person is
    rendered at about half the pixels -- as by the presence of the second person. Same
    geometry, no second person: a miss here means resolution, not tracking.
    """
    import numpy as np
    cap = cv2.VideoCapture(path_a)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return False
    h, w = frame.shape[:2]
    blank = np.zeros((h, w, 3), dtype=frame.dtype)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 2, h))
    if not writer.isOpened():
        cap.release()
        return False
    n = 0
    while ok:
        writer.write(cv2.hconcat([frame, blank]))
        n += 1
        ok, frame = cap.read()
        if n > 3000:
            break
    writer.release()
    cap.release()
    return n > 0


def main():
    random.seed(42)  # fixed so the set is reproducible and re-runs compare like for like
    os.makedirs(OUT_DIR, exist_ok=True)
    clean_adl = [a for a in VAL_ADL if a not in NOISY_ADL]

    made = {'fall_adl': [], 'adl_adl': []}

    for i, fall in enumerate(VAL_FALL):
        adl = clean_adl[i % len(clean_adl)]
        fp = os.path.join(FALL_DIR, fall + '.mp4')
        ap = os.path.join(ADL_DIR, adl + '.mp4')
        if not (os.path.exists(fp) and os.path.exists(ap)):
            continue
        out = os.path.join(OUT_DIR, f'falladl_{fall}__{adl}.mp4')
        if side_by_side(fp, ap, out):
            made['fall_adl'].append(os.path.basename(out))
            print('made', os.path.basename(out), flush=True)

    for i in range(0, len(clean_adl) - 1, 2):
        a, b = clean_adl[i], clean_adl[i + 1]
        ap, bp = os.path.join(ADL_DIR, a + '.mp4'), os.path.join(ADL_DIR, b + '.mp4')
        if not (os.path.exists(ap) and os.path.exists(bp)):
            continue
        out = os.path.join(OUT_DIR, f'adladl_{a}__{b}.mp4')
        if side_by_side(ap, bp, out):
            made['adl_adl'].append(os.path.basename(out))
            print('made', os.path.basename(out), flush=True)

    for fall in VAL_FALL:
        fp = os.path.join(FALL_DIR, fall + '.mp4')
        if not os.path.exists(fp):
            continue
        out = os.path.join(OUT_DIR, 'fallblank_' + fall + '.mp4')
        if blank_beside(fp, out):
            made.setdefault('fall_blank', []).append(os.path.basename(out))
            print('made', os.path.basename(out), flush=True)

    print('controls:', len(made.get('fall_blank', [])))
    print(f"\n{len(made['fall_adl'])} fall+adl composites, {len(made['adl_adl'])} adl+adl")
    print('in', OUT_DIR)


if __name__ == '__main__':
    main()
