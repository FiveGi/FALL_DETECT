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
