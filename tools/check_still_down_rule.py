"""The still-down rule must not change its answer because the machine was slow.

Runs without Flask, a database or a GPU, the same way `check_alert_rules.py` does, so the rule
can be checked on the CPU server it deploys to.

The defect this guards against was live for a while and is worth stating plainly, because it
failed in the direction that matters. The rule converted a down-FRAME count into seconds by
dividing by `target_fps or achieved`, and every deployed profile sets `target_fps` -- so it
divided by the rate the loop was *asked* for, not the rate it reached. A loop managing 4 fps
against a target of 8 reported half the real seconds, so a person who had been on the floor for
the whole window read as having been down for half of it, and the follow-up concluded they had
got back up.

The slower the machine, the more confidently it was wrong. The production server is CPU-only
with four cores and is the machine most likely to miss its target, which is to say: the
deployment where this mattered most was the one where it was most broken.

Usage:
    python tools/check_still_down_rule.py
"""
import ast
import inspect
import os
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_rule():
    """Execute notification_service.py directly, the way check_alert_rules.py does.

    Importing the package would pull in Flask, SQLAlchemy and the LINE client, none of which
    this rule needs -- and needing them would stop this running on the server it protects.
    """
    path = os.path.join(ROOT, 'app', 'services', 'notification_service.py')
    src = open(path, encoding='utf-8').read()
    src = src.replace('from app.services.line_service import send_line_message_async',
                      'send_line_message_async = None')
    ns = {}
    exec(compile(src, path, 'exec'), ns)
    return ns


_ns = load_rule()
still_down_confirmed = _ns['still_down_confirmed']
STILL_DOWN_FRACTION = _ns['STILL_DOWN_FRACTION']
STILL_DOWN_SECONDS = _ns['STILL_DOWN_SECONDS']

failures = []


def check(name, got, want):
    ok = got == want
    print('%-4s %s' % ('PASS' if ok else 'FAIL', name))
    if not ok:
        failures.append('%s: got %r, wanted %r' % (name, got, want))


STILL_DOWN_SEEN_FRACTION = _ns['STILL_DOWN_SEEN_FRACTION']
print('STILL_DOWN_SECONDS = %g   not-upright fraction = %g   seen-down fraction = %g'
      % (STILL_DOWN_SECONDS, STILL_DOWN_FRACTION, STILL_DOWN_SEEN_FRACTION))
print()
sd = still_down_confirmed   # (frames_since_upright, frames_seen_down, frames_elapsed)

# The answer depends on FRACTIONS of the window, so the achieved rate cannot change it.
print('seen lying down for the whole window, at every achieved rate:')
for n in (8, 20, 40, 100, 200):
    check('  %3d frames elapsed, all seen down' % n, sd(n, n, n), True)
print()

print('a person who genuinely got up halfway through:')
for n in (20, 40, 100):
    check('  %3d frames, upright seen at the midpoint' % n, sd(n // 2, n // 2, n), False)
print()

# Gemini's R5/R6 review: unseen frames must not read as "got up". A person on the floor and
# partly hidden, or lost in night-vision frames, is still down.
print('down, but not always visible (must still confirm):')
check('  night-vision loss, seen 88% of the window', sd(100, 88, 100), True)
check('  half hidden behind the bed, seen 40%', sd(100, 40, 100), True)
print()

# Codex REVIEW-2 reproduced this: absence alone confirmed "still down". Somebody who leaves the
# picture has not been seen lying there, so it must not confirm -- at any processing rate,
# including the 0.8 fps case Codex used (8 frames over 10 seconds).
print('gone from the picture, never seen down (must NOT confirm):')
check('  100 frames, none seen', sd(100, 0, 100), False)
check('  Codex repro: 8 frames at 0.8 fps, none seen', sd(8, 0, 8), False)
check('  seen down only briefly, 10%', sd(100, 10, 100), False)
print()

print('the boundaries are fractions, not frame counts:')
f, g = STILL_DOWN_FRACTION, STILL_DOWN_SEEN_FRACTION
check('  both exactly at their fractions', sd(int(100 * f), int(100 * g), 100), True)
check('  not-upright one frame short', sd(int(100 * f) - 1, 100, 100), False)
check('  seen-down one frame short', sd(100, int(100 * g) - 1, 100), False)
print()

print('degenerate windows cannot answer yes:')
check('  no frames elapsed', sd(0, 0, 0), False)
check('  negative elapsed (clock or counter reset)', sd(5, 5, -1), False)
check('  nobody ever down', sd(0, 0, 50), False)
print()

# A rate must not appear in the rule at all. This is the check that would have caught the
# original defect, and a check that cannot fail is worse than no check, so it reads the source.
# The docstring is stripped with `ast`, not by filtering lines that contain a quote: the first
# version of this check did the latter, kept the whole body of a multi-line docstring -- which
# explains the defect at length, in the words being searched for -- and failed every token.
src = inspect.getsource(still_down_confirmed)
_fn = ast.parse(textwrap.dedent(src)).body[0]
if (_fn.body and isinstance(_fn.body[0], ast.Expr)
        and isinstance(_fn.body[0].value, ast.Constant)
        and isinstance(_fn.body[0].value.value, str)):
    _fn.body = _fn.body[1:]
body = ast.unparse(_fn)
print('the rule does not mention a frame rate:')
for token in ('fps', 'target_fps', 'rate', 'seconds'):
    check('  no %r in the executable body' % token, token in body, False)

print()
if failures:
    print('FAILED:')
    for f in failures:
        print('  ' + f)
    sys.exit(1)
print('all checks passed -- the still-down answer is independent of the achieved frame rate')


# ---------------------------------------------------------------------------------------------
# Lifecycle: drive the REAL per-person state machine frame by frame, then apply the follow-up
# exactly as the camera loop does. Codex asked for this (REVIEW-2 delta): the pure-function checks
# above cannot see that frames_seen_down carries sightings from BEFORE the alert.
# Needs the detector module (numpy/cv2/onnxruntime), which the server has; skipped with a note
# if it cannot be imported, never silently passed.
# ---------------------------------------------------------------------------------------------
def lifecycle():
    import importlib.util as _u
    import numpy as np
    spec = _u.spec_from_file_location('v3', os.path.join(ROOT, 'app/detection/v3_fall_detection.py'))
    v3 = _u.module_from_spec(spec)
    spec.loader.exec_module(v3)
    seen_since = _ns['seen_down_since_alert']

    class NoModel:                       # the classifier is not what is under test here
        def predict_window(self, _w):
            return 0.0

    def body(upright):
        k = np.zeros((17, 3), np.float32)
        k[:, 2] = 0.9
        k[v3.LEFT_HIP, :2], k[v3.RIGHT_HIP, :2] = (0.45, 0.60), (0.55, 0.60)
        if upright:
            k[v3.LEFT_SHOULDER, :2], k[v3.RIGHT_SHOULDER, :2] = (0.45, 0.30), (0.55, 0.30)
        else:
            k[v3.LEFT_SHOULDER, :2], k[v3.RIGHT_SHOULDER, :2] = (0.15, 0.58), (0.15, 0.62)
        return k

    def run(before, after):
        """before: frames up to and including the alert; after: the follow-up window.
        Each item is 'U' upright, 'D' seen lying down, '-' not seen."""
        st, det = v3.V3FallDetectionState(), NoModel()
        blank = np.zeros((17, 3), np.float32)

        def step(c):
            v3._step_person(blank if c == '-' else body(c == 'U'), c != '-', st, det, 0.65)
        for c in before:
            step(c)
        seen_at_alert = st.frames_seen_down
        for c in after:
            step(c)
        return sd(st.frames_since_upright, seen_since(st.frames_seen_down, seen_at_alert),
                  len(after))

    print('lifecycle, through the real state machine:')
    check('  Codex repro: seen falling (3 down), then 8 frames gone', run('UUUDDD', '-' * 8), False)
    check('  seen falling, then gone for a long window', run('UUUDDD', '-' * 80), False)
    check('  seen falling, then seen lying there 80 frames', run('UUUDDD', 'D' * 80), True)
    check('  lying there, briefly hidden (25% unseen)', run('UUUDDD', ('D' * 3 + '-') * 20), True)
    check('  lying there, night-like loss (12% unseen)', run('UUUDDD', ('D' * 7 + '-') * 10), True)
    check('  got up mid-window', run('UUUDDD', 'D' * 30 + 'U' * 10 + '-' * 40), False)
    print()


try:
    lifecycle()
except ImportError as exc:
    print('SKIP lifecycle checks: detector module not importable here (%s)' % exc)
    print()

if failures:
    print('FAILED:')
    for f in failures:
        print('  ' + f)
    sys.exit(1)
print('lifecycle included -- all checks passed')
