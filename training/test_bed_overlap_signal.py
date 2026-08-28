"""
Refined hypothesis: suppression signal should be PERSON-BED BOUNDING BOX OVERLAP
(what fraction of the person's box sits inside a bed box), not mere co-occurrence
of a bed anywhere in the room.

Tests:
  A) The 6 known ADL false-positive frames -> expect HIGH overlap (person is on the bed)
  B) The last frame of each of the 15 genuine val fall clips (person down on floor,
     often with a bed elsewhere in the room) -> expect LOW overlap
"""
import os
import cv2
from ultralytics import YOLO

model = YOLO("yolov10x.pt")

def person_bed_overlap(frame):
    r = model.predict(frame, verbose=False, conf=0.15)[0]
    person_boxes = []
    bed_boxes = []
    for box in r.boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        xyxy = box.xyxy[0].tolist()
        if name == "person":
            person_boxes.append(xyxy)
        elif name in ("bed", "couch"):
            bed_boxes.append(xyxy)
    if not person_boxes or not bed_boxes:
        return 0.0, len(person_boxes), len(bed_boxes)
    best = 0.0
    for px1, py1, px2, py2 in person_boxes:
        p_area = max(0, px2 - px1) * max(0, py2 - py1)
        if p_area <= 0:
            continue
        for bx1, by1, bx2, by2 in bed_boxes:
            ix1, iy1 = max(px1, bx1), max(py1, by1)
            ix2, iy2 = min(px2, bx2), min(py2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            frac = inter / p_area
            best = max(best, frac)
    return best, len(person_boxes), len(bed_boxes)


print("=== A) Known ADL false-positive frames (expect HIGH overlap) ===")
ADL_DIR = "data/gmdcsa24_adl_raw_val"
ALERTS = {
    "s1_ADL_01": [2.0], "s2_ADL_03": [9.4], "s2_ADL_15": [6.6],
    "s4_ADL_07": [1.3], "s4_ADL_08": [1.3, 6.7], "s4_ADL_10": [4.7],
}
for clip, timestamps in ALERTS.items():
    path = os.path.join(ADL_DIR, f"{clip}.mp4")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ts * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        overlap, np_, nb = person_bed_overlap(frame)
        print(f"  {clip} @ {ts}s: overlap={overlap:.2f}  (persons={np_}, beds={nb})")
    cap.release()

print("\n=== B) Last frame of each genuine val FALL clip (expect LOW overlap) ===")
FALL_DIR = "data/gmdcsa24_fall_raw"
VAL_FALL = ["s1_Fall_02", "s1_Fall_06", "s1_Fall_16", "s2_Fall_02", "s2_Fall_04",
            "s2_Fall_09", "s2_Fall_14", "s2_Fall_20", "s3_Fall_02", "s3_Fall_09",
            "s3_Fall_12", "s3_Fall_13", "s3_Fall_16", "s4_Fall_06", "s4_Fall_17"]
for clip in VAL_FALL:
    path = os.path.join(FALL_DIR, f"{clip}.mp4")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # sample last 1.5s, take the max overlap seen (worst case for suppression)
    duration = n_frames / fps
    max_overlap = 0.0
    t = max(0.0, duration - 1.5)
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            overlap, np_, nb = person_bed_overlap(frame)
            max_overlap = max(max_overlap, overlap)
        t += 0.3
    cap.release()
    flag = "  <-- WOULD BE SUPPRESSED" if max_overlap > 0.5 else ""
    print(f"  {clip}: max_overlap(last1.5s)={max_overlap:.2f}{flag}")
