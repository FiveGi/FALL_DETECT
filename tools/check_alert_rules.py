"""Check the alert-wording rule, and that the web UI still agrees with the backend about it.

The rule lives in two places by necessity -- `app/services/notification_service.alert_tier`
decides the wording of the LINE message, and `getAlertTier` in
`frontend/src/utils/detectionType.js` decides the badge shown for the same alert on screen. If
one moves and the other does not, the same event reads as urgent in chat and "please check" on
the dashboard. Nothing catches that: the Vue build passes either way, and the browser smoke
test only asserts that a badge renders.

It also guards the finding in SKILL.md SS47: the confidence score must not come back into the
decision. At the bar the system used to split on, a genuine-fall alert and a false alarm were
about equally likely to clear it, so the wording carried no information while sounding like it
did. Re-derive with `training/measure_alert_tier.py` before ever reintroducing a score-based
tier -- and re-run it after any change to the model, the window or the frame rate, since the
scores move with all three.

Runs without Docker, a database or a GPU.

Usage:
    python tools/check_alert_rules.py
"""
import os
import re
import sys

# The alert wording this checks is Thai, and it gets printed. A Windows console defaults to the
# system codepage -- cp874 here -- which cannot encode the emoji in the escalated message, so
# the script died on its own output with a UnicodeEncodeError before reporting anything. Inside
# the container stdout is already UTF-8 and this is a no-op.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

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

    # The second fact that may raise the tier. It has to raise, never lower, and it must not
    # reach a non-fall alert -- bed-exit and alone are advisory whatever the camera saw.
    still = ns['STILL_DOWN_SECONDS']
    for detection_type, seconds, expected in (
        ('fall_red', None, 'check'),
        ('fall_red', still - 1, 'check'),
        ('fall_red', still, 'confirmed'),
        ('fall_red', still * 10, 'confirmed'),
        ('alone_yellow', still * 10, 'check'),
        ('bed_exit', still * 10, 'check'),
    ):
        got = alert_tier(detection_type, 0, seconds)
        check(f'backend  {detection_type:12s} still_down={seconds} -> {expected}',
              got == expected, '' if got == expected else f'got {got}')
    check('backend  still-down can only RAISE the tier, never lower one',
          alert_tier('fall_red', 1, 0) == 'confirmed', 'escalated + they got up must stay confirmed')

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
        check('frontend rule confirms on escalation',
              re.search(r'escalationCount\s*>\s*0\).*?[\'"]confirmed[\'"]', body, re.S) is not None)
        check('frontend rule confirms on still-down',
              re.search(r'stillDownSeconds\s*!=\s*null\s*&&\s*stillDownSeconds\s*>=', body) is not None)
        check('frontend takes the threshold from the server, not only its own constant',
              'threshold' in args, f'signature: ({args.strip()})')
        check('frontend rule keeps non-fall alerts in the lower tier',
              "includes('fall')" in body and "return 'check'" in body)

    check('no confidence threshold constant survives in the frontend',
          'CONFIRMED_CONFIDENCE' not in js)

    # Two copies of a number that decides what a family is told. If they drift, the dashboard
    # and the LINE message disagree about the same alert and nothing else would catch it.
    m_js = re.search(r'export const STILL_DOWN_SECONDS\s*=\s*([0-9.]+)', js)
    check('the web UI default still-down threshold matches the backend default',
          m_js is not None and float(m_js.group(1)) == float(ns['STILL_DOWN_SECONDS']),
          f"frontend {m_js.group(1) if m_js else 'missing'} vs backend {ns['STILL_DOWN_SECONDS']}")

    # The LINE wording is the thing a family actually reads, and it is the easiest place for
    # the old behaviour to creep back: an escalated alert is urgent because nobody answered,
    # not because the system knows a fall happened. Assert the urgent branch says so.
    line_src = open(os.path.join(ROOT, 'app', 'services', 'line_service.py'),
                    encoding='utf-8').read()
    fall_branch = line_src[line_src.index('if "fall" in detection_type'):]
    fall_branch = fall_branch[:fall_branch.index('elif "alone" in detection_type')]
    confirmed_text = [ln for ln in fall_branch.splitlines()
                      if 'event_text =' in ln and not ln.strip().startswith('#')]
    # Three wordings, and no more: still-down, unacknowledged, and the "please check" that a
    # fresh alert gets. Each urgent one names the FACT that raised it, because "they have not
    # got up for twelve seconds" and "nobody has looked at this" are different situations and a
    # single urgent message for both would be telling a family something the system does not
    # know. The count is asserted so a fourth wording cannot appear unnoticed.
    check('LINE has one wording per fact and no more',
          len(confirmed_text) == 3, f'found {len(confirmed_text)} event_text assignments')
    urgent = ' '.join(confirmed_text)
    check('no urgent LINE message asserts that a fall happened',
          'ตรวจพบการล้ม' not in urgent, urgent.strip()[:90])
    check('the unacknowledged wording says nobody has checked it',
          any('ยังไม่มีใครตรวจสอบ' in ln or 'ไม่มีใคร' in ln for ln in confirmed_text))
    check('the still-down wording says they have not got up',
          any('ยังไม่ลุก' in ln for ln in confirmed_text))

    # The defaults agreeing is not enough on its own: STILL_DOWN_SECONDS is settable, and a
    # deployment that changes it would leave the dashboard on its compiled-in number. The API
    # sends the live value with each notification for that reason, and the UI has to use it.
    logs_src = open(os.path.join(ROOT, 'app', 'routes', 'logs.py'), encoding='utf-8').read()
    check('the API sends the live threshold with each notification',
          "'still_down_threshold': STILL_DOWN_SECONDS" in logs_src)
    monitor = open(os.path.join(ROOT, 'frontend', 'src', 'views', 'MonitorView.vue'),
                   encoding='utf-8').read()
    check('the dashboard passes the threshold the server sent',
          'notification.still_down_threshold' in monitor)

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
    print('all checks passed -- backend, LINE wording and web UI agree on the tier')
    return 0


if __name__ == '__main__':
    sys.exit(main())
