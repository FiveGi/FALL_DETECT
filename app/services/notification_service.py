import os
from app.services.line_service import send_line_message_async

# The alert score does NOT decide how an alert is worded. Measured on the quantity the system
# actually sends -- the score at the instant the alert fires -- across all four labelled
# surfaces at the 15 fps the camera loop is pinned to (training/measure_alert_tier.py,
# 135 alerts):
#
#     bar    real-fall alerts above it    false alarms above it    precision above it
#     0.70          97/119 ( 82%)               14/16 ( 88%)              87%
#     0.80          71/119 ( 60%)                4/16 ( 25%)              95%
#     0.90          26/119 ( 22%)                1/16 (  6%)              96%
#
# Unlike the model this replaced, the score here does carry information: above 0.80 an alert is
# 95% real against 88% overall. That is a real change and it is recorded honestly -- an earlier
# version of this comment said the score separated nothing, which was true of the previous
# model and is not true of this one.
#
# It still does not justify telling a family a fall happened, for two reasons. The
# highest-scoring alert in the whole corpus, 0.96, is a man getting up from a bed
# (s4_ADL_08) -- the score's top end is exactly where the hardest false alarms sit. And the
# false-alarm column is 16 alerts, so the differences that look decisive are a few clips.
# Reintroducing a score-based bar would need it chosen on one half of URFD and confirmed on
# the other, the way SS51 chose the detection threshold; reading the table above and picking
# from it is the contamination this project has already paid for twice.
#
# So urgency comes from the one signal that is a fact rather than an estimate: nobody answered.
# A fresh fall alert asks a human to look; escalation_service promotes it once it goes
# unacknowledged (see notify_alert).


# Seconds on the floor after an alert before the system says so. Owned here rather than in
# Config because this is an alerting rule, and tools/check_alert_rules.py loads this file with
# no Flask present so the rule can be checked without Docker, a database or a GPU. The web UI
# has the same number in frontend/src/utils/detectionType.js and that checker compares them.
STILL_DOWN_SECONDS = float(os.environ.get('STILL_DOWN_SECONDS', 10))

# Fraction of the window a person must have been non-upright for "still down" to stand. Below
# 1.0 because a torso angle cannot always be measured -- an occluded or partly-visible person
# yields no reading, and those frames should not count as getting up.
STILL_DOWN_FRACTION = 0.8


# Fraction of the window in which the person must have been SEEN lying down. Deliberately
# lower than STILL_DOWN_FRACTION: a person on the floor is often partly hidden, and night-vision
# frames lose the person ~12% of the time, so demanding they be seen most of the window would
# close real cases as "unknown". What it rules out is a confirmation built on absence alone.
STILL_DOWN_SEEN_FRACTION = 0.3


def seen_down_since_alert(seen_now, seen_at_alert):
    """Sightings of the person lying down counted from the alert onward, not before it.

    `frames_seen_down` accumulates from the last upright sighting, which is BEFORE the alert --
    the fall itself is usually seen. Handing the raw counter to still_down_confirmed let those
    pre-alert sightings confirm a follow-up window in which the person was never seen at all
    (Codex REVIEW-2 delta: 3 sightings at the fall, then 8 absent frames -> confirmed). If the
    counter was reset by an upright sighting after the alert, what it holds is already all
    post-alert. Both callers -- the camera loop and tier_accuracy.py -- go through this.
    """
    return seen_now - seen_at_alert if seen_now >= seen_at_alert else seen_now


def still_down_confirmed(frames_since_upright, frames_seen_down, frames_elapsed):
    """Were they down for essentially the whole window? -> bool. Counted in FRAMES.

    Two conditions, both counted on the camera loop's own clock:
      1. not seen upright for STILL_DOWN_FRACTION of the window -- `frames_since_upright`
         counts every frame since the last upright sighting, seen or not;
      2. actually SEEN lying down for STILL_DOWN_SEEN_FRACTION of it -- counted within the
         window, via seen_down_since_alert().

    The second exists because the first alone confirmed "still on the floor" for somebody who
    had simply left the picture: an unseen frame is not evidence of getting up, but it is not
    evidence of lying there either (Codex REVIEW-2 reproduced it). And the first counts unseen
    frames because counting only seen-down frames closed real, partly-hidden cases as "got up"
    (Gemini's R5/R6 review).

    **Frames, never seconds**: dividing by a target frame rate halved the answer whenever the
    loop ran below target -- the CPU server's normal state. See tools/check_still_down_rule.py.
    """
    if frames_elapsed <= 0:
        return False
    return (frames_since_upright >= STILL_DOWN_FRACTION * frames_elapsed
            and frames_seen_down >= STILL_DOWN_SEEN_FRACTION * frames_elapsed)


def alert_tier(detection_type, escalation_level=0, still_down_seconds=None):
    """-> 'confirmed' | 'check'. Falls ask a human to look; an alert nobody acknowledged is
    escalated to the urgent wording. Non-fall events (bed exit, alone) are advisory by
    nature and always land in the lower tier.

    still_down_seconds raises the tier for the same reason escalation does: it is a FACT about
    what the camera saw, not an estimate of how sure the model was. Somebody who has been on
    the floor for ten seconds is a different situation from somebody the model thought fell a
    moment ago, and it is the one distinction a human makes that the classifier cannot.
    Measured at 94% of real falls against half the false alarms -- so it is useful evidence and
    poor proof, which is exactly what raising urgency (rather than asserting a fall) is for.

    It never lowers the tier. The inverse rule -- they got up, so cancel -- was measured on the
    same data and would suppress 6% of real falls, silently."""
    if "fall" not in detection_type:
        return "check"
    if escalation_level > 0:
        return "confirmed"
    if still_down_seconds is not None and still_down_seconds >= STILL_DOWN_SECONDS:
        return "confirmed"
    return "check"


def notify_alert(camera_id, camera_name, room_name, detection_type, timestamp, image_path,
                 confidence=None, escalation_level=0, notification_id=None,
                 still_down_seconds=None, wait=False):
    """Single entry point for every outbound alert channel (currently LINE). Detection loops call this
    instead of each channel's sender directly, so adding/removing a channel or changing
    the tier rule is a one-line change here rather than an edit repeated at every alert
    call site.

    escalation_level > 0 marks a re-send of an alert nobody acknowledged (see
    escalation_service), and that is the only thing that raises the tier: by then the point
    is that it went unanswered, which is a fact, rather than how sure the model was, which
    the measurement above shows is not usable."""
    tier = alert_tier(detection_type, escalation_level, still_down_seconds)
    # notification_id only reaches LINE: it is what the "รับทราบ" button posts back, so the
    # webhook can mark that exact alert acknowledged and reply with its clip.
    # wait=True returns how many targets the alert reached (see line_service); detection loops
    # do not wait, so a slow LINE API can never stall frame processing.
    return send_line_message_async(camera_id, camera_name, room_name, detection_type, timestamp,
                                   image_path, tier=tier, confidence=confidence,
                                   escalation_level=escalation_level,
                                   notification_id=notification_id,
                                   still_down_seconds=still_down_seconds, wait=wait)
