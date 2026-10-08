# -*- coding: utf-8 -*-
"""Print stage1_eval's stock cache lists (DAY 8 / IR 3 / OWNER 8) for pinning."""
import importlib.util
import os
import sys

sys.argv = ['x']
s = importlib.util.spec_from_file_location('s', 'training/measure/stage1_eval.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
fix = lambda c: c.replace(os.sep, '/')
print('DAY', ' '.join(fix(c) for c in m.day_caches()))
print('IR', ' '.join(fix(c) for c in m.ir_caches()))
print('OWNER', ' '.join(fix(c) for c in m.owner_caches()))
