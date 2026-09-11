"""CPU-vs-GPU control on one clip: same code, same model, only the device differs.

Needed because SS36's baseline alert counts were measured on CPU and this run was the first
on GPU -- a difference in totals could be the device's floating-point behaviour rather than
any real change, and the two are worth telling apart before trusting either number.
"""
import importlib.util, json, os, sys
import cv2

os.environ['V3_DEVICE'] = sys.argv[2]
spec = importlib.util.spec_from_file_location('v3', 'app/detection/v3_fall_detection.py')
v3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3)

clip = sys.argv[1]
det = v3.V3PoseFallDetector(model_dir='models')
state = v3.V3MultiPersonFallState()
cap = cv2.VideoCapture(f'Test/{clip}.mp4')
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
i, alerts, last = 0, [], None
while True:
    ok, f = cap.read()
    if not ok:
        break
    res = v3.detect_v3_fall_multi(f, state, det, config=None)
    label = 'fall' if any(r[1] for r in res) else 'no_fall'
    if label != last:
        if label == 'fall':
            alerts.append(round(i / fps, 1))
        last = label
    i += 1
print(f'clip {clip} on {sys.argv[2]}: {len(alerts)} alerts at {alerts}')
