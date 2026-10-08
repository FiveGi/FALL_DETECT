# -*- coding: utf-8 -*-
"""Contact sheets of the CLS-ADAPT OF-Syn clips for a realism audit (Codex: audit both classes before
training). 4 frames per clip spread over the clip, 5 clips per row, numbered; sheets of 30 clips.
Writes D:/project/PROJECT/.ai_evidence/ofsyn_audit/{lie,fall}_NN.jpg + index.txt"""
import os
import random

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, 'training', 'data', 'omnifall_syn')
OUT = 'D:/project/PROJECT/.ai_evidence/ofsyn_audit_all'
os.makedirs(OUT, exist_ok=True)
L = pd.read_csv(os.path.join(SYN, 'labels', 'of-syn.csv'))
clips = [l.strip() for l in open(os.path.join(ROOT, 'training', 'data', 'clsadapt_syn_clips.txt')) if l.strip()]
lie = {p for p, d in L.groupby('path') if 5 in set(d.label)}
rng = random.Random(3)
groups = {'lie': sorted(p for p in clips if p in lie),
          'fall': sorted(p for p in clips if p not in lie)}
index = []
for name, paths in groups.items():
    for s in range(0, len(paths), 30):
        rows = []
        for k, p in enumerate(paths[s:s + 30]):
            cap = cv2.VideoCapture(os.path.join(SYN, 'videos', p + '.mp4'))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 80
            tiles = []
            for f in np.linspace(0, n - 1, 4).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, im = cap.read()
                im = cv2.resize(im if ok else np.zeros((180, 320, 3), np.uint8), (160, 90))
                tiles.append(im)
            cell = np.hstack(tiles)
            tag = '%s%d' % (name[0].upper(), s + k)
            cv2.putText(cell, tag, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            rows.append(cell); index.append('%s %s' % (tag, p))
        rows += [np.zeros_like(rows[0])] * (-len(rows) % 2)
        sheet = np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows), 2)])
        cv2.imwrite(os.path.join(OUT, '%s_%02d.jpg' % (name, s // 30)), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
open(os.path.join(OUT, 'index.txt'), 'w').write('\n'.join(index))
print('ok', len(index))
