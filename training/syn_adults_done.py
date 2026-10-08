# Exit 0 when every adult OF-Syn clip (the ages train.py uses by default) has its offset-0 pose file.
import os, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADULT = {'elderly_65_plus', 'middle_aged_35_64', 'young_adults_18_34'}
m = pd.read_csv(os.path.join(ROOT, 'training/data/omnifall_syn/labels/of-syn.csv')).groupby('path').first()
want = [p for p in m.index if m.loc[p, 'age_group'] in ADULT]
have = sum(os.path.exists(os.path.join(ROOT, 'training/data/poses_omnifall_syn', p.replace('/', '__') + '_o0.npz')) for p in want)
print('adult clips extracted %d / %d' % (have, len(want)))
sys.exit(0 if have == len(want) else 1)
