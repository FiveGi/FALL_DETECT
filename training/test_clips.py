# -*- coding: utf-8 -*-
"""What each clip in `Test/` actually contains, in one place.

These seventeen files are the only footage in this project that is not a lab dataset, and how
they are scored has been got wrong before: `Test/17` sat in the "real elderly falls" group for
days as a clip every configuration missed, which made the deployed detector look worse than it
is and made a 4-of-5 configuration look better. Watching it settles it -- a crowd doing an
outdoor exercise routine, filmed from across a courtyard, with a lot of deep-squat motion and
nobody falling. Gemini reads it the same way. Silence is the right answer there, so it is scored
as a clip that must NOT alert.

Three kinds, because they are not scored the same way:

  REAL          one real fall each, so a configuration simply catches it or misses it.
  NEGATIVE      no fall at all. An alert is wrong.
  COMPILATION   social-media compilations with several incidents per file and no per-incident
                ground truth. A count, never a score -- reporting these as accuracy would be
                inventing a denominator.

`Test/10` and `Test/11` are children, filmed small and at night; they are out of the domain
this system is built for and are listed with the compilations for that reason, not because
nobody falls in them.

The Thai one-line description of every clip lives beside the clips in `Test/clips.json`.
"""
import json
import os

TEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Test')

REAL = ['Test/%d.mp4' % n for n in range(13, 17)]
NEGATIVE = ['Test/17.mp4']
COMPILATION = ['Test/%d.mp4' % n for n in range(1, 13)]
ALL = ['Test/%d.mp4' % n for n in range(1, 18)]


def kind(clip):
    """-> 'real' | 'negative' | 'compilation' for a path like 'Test/13.mp4'."""
    name = 'Test/' + os.path.basename(clip)
    if name in REAL:
        return 'real'
    if name in NEGATIVE:
        return 'negative'
    return 'compilation'


def descriptions():
    """-> {'13.mp4': 'ล้มจริง 1 ครั้ง ...'} from Test/clips.json, or {} if it is missing."""
    path = os.path.join(TEST_DIR, 'clips.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)
