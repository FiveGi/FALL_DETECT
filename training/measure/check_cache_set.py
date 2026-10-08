# -*- coding: utf-8 -*-
"""Fail unless CACHE_ROOT holds exactly N cache dirs, each with every clip the stock CPU day cache
has, all loadable. Guards the POSE-IR screening against evaluating partial caches (Codex P2)."""
import glob
import os
import sys

import numpy as np

root, n = sys.argv[1], int(sys.argv[2])
ref = {os.path.basename(f) for f in glob.glob('training/data/pose_cache/d647cd5294c*/*.npz')}
dirs = [d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d)]
bad = []
for d in dirs:
    have = {os.path.basename(f) for f in glob.glob(os.path.join(d, '*.npz'))}
    if have != ref:
        bad.append('%s: %d/%d clips' % (d, len(have & ref), len(ref)))
        continue
    for f in have:
        try:
            np.load(os.path.join(d, f))['counts']
        except Exception as e:  # truncated by an interrupted run
            bad.append('%s/%s: %s' % (d, f, e))
if len(dirs) != n or bad:
    sys.exit('INCOMPLETE %s: %d dirs (want %d); %s' % (root, len(dirs), n, '; '.join(bad[:5])))
print('complete %s: %d dirs x %d clips' % (root, n, len(ref)))
