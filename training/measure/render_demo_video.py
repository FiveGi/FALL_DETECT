# -*- coding: utf-8 -*-
"""A demo video of the detector at work: what it saw, frame by frame, at the rate it ran.

Same pipeline and sampling as eval_incidents_cpu.py (8 fps by source time, crop state reset at
the start), so the video shows the configuration that was measured, not a nicer one. Each person
the detector saw gets a skeleton -- yellow while watching, red while their track is alerting --
and their fall score. A top bar shows the time and the state; a bottom bar a Thai caption.
Played back at the analysis rate (8 fps = real time), so it looks as choppy as the camera loop
really is.

Usage:
  V3_DEVICE=cpu V3_IMGSZ=320 V3_ROI_IMGSZ=256 V3_ROI_FULL_EVERY=8 V3_POSE_CONF=0.3 \
  START_S=0 END_S=10 CAPTION="..." python training/measure/render_demo_video.py Test/15.mp4 out.mp4
"""
import importlib.util
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location('v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

FPS = float(os.environ.get('TARGET_FPS', 8))
WIDTH = int(os.environ.get('WIDTH', 960))
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
FONT = 'C:/Windows/Fonts/tahoma.ttf'


def text_bar(img, txt, top, colour):
    h, w = img.shape[:2]
    bar = 40
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    y0 = 0 if top else h - bar
    d.rectangle([0, y0, w, y0 + bar], fill=colour)
    d.text((14, y0 + 7), txt, font=ImageFont.truetype(FONT, 22), fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def night_alt():
    """NIGHT_ALT=1: show and detect on the test-only night degradation (cache_pose_streams.to_night_alt),
    imported rather than copied so the demo and the measurement cannot drift apart."""
    if os.environ.get('NIGHT_ALT') != '1':
        return None
    os.environ['SIMULATE_NIGHT_ALT'] = '1'
    sp = importlib.util.spec_from_file_location('cps', os.path.join(ROOT, 'training/measure/cache_pose_streams.py'))
    cps = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(cps)
    return cps


def main(src, out):
    cps = night_alt()
    rng = cps.clip_rng(src) if cps else None
    rgb_half = os.environ.get('RIGHT_HALF') == '1'   # URFD files hold depth | colour side by side
    det = v3.V3PoseFallDetector(model_dir=os.path.join(ROOT, 'models'))
    seen = {'people': []}
    real = det.extract_all_keypoints

    def spy(frame):
        seen['people'] = real(frame)
        return seen['people']
    det.extract_all_keypoints = spy
    det.reset_roi_state(int(os.environ.get('ROI_PHASE', 0)))
    state = v3.V3MultiPersonFallState()
    caption = os.environ.get('CAPTION', '')
    start_s, end_s = float(os.environ.get('START_S', 0)), float(os.environ.get('END_S', 0))
    cap = cv2.VideoCapture(src)
    sf = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_s:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * sf))
    i, last_slot, writer, n_alert = int(start_s * sf), -1, None, 0
    while True:
        ok, frame = cap.read()
        if not ok or (end_s and i >= int(end_s * sf)):
            break
        slot = int(i * FPS / sf)
        i += 1
        if slot == last_slot:
            continue
        last_slot = slot
        if rgb_half:
            frame = frame[:, frame.shape[1] // 2:]
        if cps:
            frame = cps.to_night_alt(frame, rng=rng)
        seen['people'] = []
        results = v3.detect_v3_fall_multi(frame, state, det, config=None)
        h, w = frame.shape[:2]
        k = WIDTH / float(w)
        img = cv2.resize(frame, (WIDTH, int(h * k) // 2 * 2))
        H, W = img.shape[:2]
        alert_now = any(r[1] for r in results)
        n_alert += alert_now
        for kp, hip in seen['people']:
            best = min(results, key=lambda r: float(np.linalg.norm(np.asarray(r[4]) - hip)), default=None)
            matched = best is not None and float(np.linalg.norm(np.asarray(best[4]) - hip)) < 0.1
            red = matched and best[1]
            col = (40, 40, 230) if red else (0, 210, 255)
            pts = [(int(x * W), int(y * H)) if c > 0.3 else None for x, y, c in kp]
            for a, b in EDGES:
                if pts[a] and pts[b]:
                    cv2.line(img, pts[a], pts[b], col, 3, cv2.LINE_AA)
            for p in pts:
                if p:
                    cv2.circle(img, p, 4, col, -1, cv2.LINE_AA)
            if matched and pts[0]:
                cv2.putText(img, 'score %.2f' % best[2], (pts[0][0] - 40, max(20, pts[0][1] - 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        t = (i / sf) - start_s
        img = text_bar(img, '%5.1f s   %s' % (t, 'แจ้งเตือน: ตรวจพบการล้ม' if alert_now else 'กำลังเฝ้าดู'),
                       True, (200, 30, 30) if alert_now else (40, 44, 52))
        if caption:
            img = text_bar(img, caption, False, (25, 25, 28))
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'avc1'), FPS, (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer:
        writer.release()
    print('wrote %s, alert frames %d' % (out, n_alert))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
