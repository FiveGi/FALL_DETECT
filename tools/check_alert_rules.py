"""Check the alert-wording rule, and that the web UI still agrees with the backend about it.

The rule lives in two places by necessity -- `app/services/notification_service.alert_tier`
decides the wording of the LINE message, and `getAlertTier` in
`frontend/src/utils/detectionType.js` decides the badge shown for the same alert on screen. If
one moves and the other does not, the same event reads as urgent in chat and "please check" on
the dashboard. Nothing catches that: the Vue build passes either way, and the browser smoke
test only asserts that a badge renders.

It also guards the finding in SKILL.md SS47: the confidence score must not come back into the
decision. Measured across 147 alerts, a false alarm is *more* likely to clear a high score than
a real fall is (`training/measure_alert_tier.py`), so a score-based tier tells families the
opposite of the truth.

Runs without Docker, a database or a GPU.

Usage:
    python tools/check_alert_rules.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, 'frontend', 'src', 'utils', 'detectionType.js')

# (detection_type, escalation_level, expected tier)
CASES = [
    ('fall_red', 0, 'check'),        # a fresh fall alert asks a human to look
    ('fall_red', 1, 'confirmed'),    # nobody acknowledged it and it was re-sent
    ('fall_red', 3, 'confirmed'),
    ('fall_yellow', 0, 'check'),
    ('bed_exit', 0, 'check'),        # advisory by nature
    ('bed_exit', 2, 'check'),        # ... even when escalated
    ('alone', 0, 'check'),
]

failures = []


def check(label, ok, detail=''):
    print(f'{"PASS" if ok else "FAIL"}  {label}{"  " + detail if detail else ""}')
    if not ok:
        failures.append(label)


def backend_rule():
    sys.path.insert(0, ROOT)
    # Import the module directly: importing the package would pull in Flask, the database and
    # the LINE client, none of which this check needs.
    import importlib.util
    path = os.path.join(ROOT, 'app', 'services', 'notification_service.py')
    src = open(path, encoding='utf-8').read()
    # The one import at the top reaches the LINE client, so stub it out rather than install it.
    src = src.replace('from app.services.line_service import send_line_message_async',
                      'send_line_message_async = None')
    ns = {}
    exec(compile(src, path, 'exec'), ns)
    return ns


def main():
    ns = backend_rule()
    alert_tier = ns['alert_tier']

    for detection_type, escalation, expected in CASES:
        got = alert_tier(detection_type, escalation)
        check(f'backend  {detection_type:12s} escalation={escalation} -> {expected}',
              got == expected, '' if got == expected else f'got {got}')

    js = open(JS, encoding='utf-8').read()

    # The frontend cannot be imported from Python, so assert the shape of the rule instead:
    # it must branch on the escalation count and must not mention a confidence threshold.
    m = re.search(r'export function getAlertTier\(([^)]*)\)\s*\{(.*?)\n\}', js, re.S)
    check('frontend getAlertTier exists', m is not None)
    if m:
        args, body = m.group(1), m.group(2)
        check('frontend rule takes the escalation count, not a confidence',
              'escalation' in args.lower() and 'confidence' not in args.lower(),
              f'signature: ({args.strip()})')
        check('frontend rule returns confirmed only when escalated',
              re.search(r'escalationCount\s*>\s*0\s*\?\s*[\'"]confirmed[\'"]', body) is not None)
        check('frontend rule keeps non-fall alerts in the lower tier',
              "includes('fall')" in body and "return 'check'" in body)

    check('no confidence threshold constant survives in the frontend',
          'CONFIRMED_CONFIDENCE' not in js)

    backend_src = open(os.path.join(ROOT, 'app', 'services', 'notification_service.py'),
                       encoding='utf-8').read()
    decision = backend_src[backend_src.index('def alert_tier'):]
    decision = decision[:decision.index('def notify_alert')]
    check('no confidence threshold survives in the backend rule',
          'confidence' not in decision.lower())

    print()
    if failures:
        print(f'{len(failures)} check(s) failed')
        return 1
    print(f'{len(CASES) + 6} checks passed -- backend and web UI agree on the alert wording')
    return 0


if __name__ == '__main__':
    sys.exit(main())
