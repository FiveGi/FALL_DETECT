from app.services.line_service import send_line_message_async

# The alert score does NOT decide how an alert is worded, and the measurement that once
# said it could has been redone properly. `camera_manager` passes the score at the instant
# the alert fires, so the bar has to be judged on alert scores -- the earlier 0.85 was
# derived from clip peaks on GMDCSA24 val alone. Swept across all four labelled surfaces
# (GMDCSA24 val + train50, URFD falls, URFD ADL; 154 alerts):
#
#     bar    real-fall alerts above it    false alarms above it    precision above it
#     0.50         131/131 (100%)               23/23 (100%)              85%
#     0.70          60/131 ( 46%)                6/23 ( 26%)              91%
#     0.85          13/131 ( 10%)                2/23 (  9%)              87%
#
# At 0.85 the bar lets through 10% of genuine-fall alerts and 9% of false alarms: it is not
# separating the two. Precision above it (87%) is within noise of precision overall (85%) and
# rests on two clips. The justification written here previously -- "nothing above 0.85 was a
# false alarm" -- is false: s4_ADL_08 (a man getting up from a bed) alerts at 0.88.
# Reproduce with training/measure_alert_tier.py.
#
# So urgency comes from the one signal that does mean something: nobody answered.
# A fresh fall alert asks a human to look; escalation_service promotes it once it goes
# unacknowledged (see notify_alert). This is the honest version of the SS33/SS40 result
# that no measured signal separates a fall from a deep bend -- routing the ambiguity to a
# person is the alternative to guessing, and asserting a guess was the bug.


def alert_tier(detection_type, escalation_level=0):
    """-> 'confirmed' | 'check'. Falls ask a human to look; an alert nobody acknowledged is
    escalated to the urgent wording. Non-fall events (bed exit, alone) are advisory by
    nature and always land in the lower tier."""
    if "fall" not in detection_type:
        return "check"
    return "confirmed" if escalation_level > 0 else "check"


def notify_alert(camera_id, camera_name, room_name, detection_type, timestamp, image_path,
                 confidence=None, escalation_level=0, notification_id=None):
    """Single entry point for every outbound alert channel (currently LINE). Detection loops call this
    instead of each channel's sender directly, so adding/removing a channel or changing
    the tier rule is a one-line change here rather than an edit repeated at every alert
    call site.

    escalation_level > 0 marks a re-send of an alert nobody acknowledged (see
    escalation_service), and that is the only thing that raises the tier: by then the point
    is that it went unanswered, which is a fact, rather than how sure the model was, which
    the measurement above shows is not usable."""
    tier = alert_tier(detection_type, escalation_level)
    # notification_id only reaches LINE: it is what the "รับทราบ" button posts back, so the
    # webhook can mark that exact alert acknowledged and reply with its clip.
    send_line_message_async(camera_id, camera_name, room_name, detection_type, timestamp,
                            image_path, tier=tier, confidence=confidence,
                            escalation_level=escalation_level,
                            notification_id=notification_id)
