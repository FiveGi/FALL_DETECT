from app.services.telegram_service import send_telegram_message_async
from app.services.line_service import send_line_message_async

# Confidence at or above which a fall is announced as confirmed rather than as something
# for staff to go check. Chosen from the measured GMDCSA24 held-out run (training/
# eval_v3_on_gmdcsa24_val.py): real falls there alert at 0.72-0.89 and false alarms at
# 0.60-0.81, so no cut cleanly separates them -- but nothing above 0.85 was a false alarm,
# which makes 0.85 a defensible bar for "say this is a fall" versus "ask someone to look".
# Both tiers still notify; this only changes how the alert is worded and prioritised, so a
# real fall landing in the lower tier is delayed by a human glance, never dropped. That
# trade is deliberate: SKILL.md SS33 tested four geometric/kinematic signals meant to
# auto-suppress the bed-lying false positives and all four overlapped genuine falls, so
# routing the ambiguous band to a person is the honest alternative to guessing.
CONFIRMED_CONFIDENCE = 0.85


def alert_tier(detection_type, confidence):
    """-> 'confirmed' | 'check' . Non-fall events (bed exit, alone) are advisory by nature
    and always land in the lower tier; a fall only counts as confirmed above the bar."""
    if "fall" not in detection_type:
        return "check"
    if confidence is None:
        return "confirmed"  # caller has no score to judge by -- don't silently downgrade
    return "confirmed" if confidence >= CONFIRMED_CONFIDENCE else "check"


def notify_alert(camera_id, camera_name, room_name, detection_type, timestamp, image_path,
                 confidence=None, escalation_level=0, notification_id=None):
    """Single entry point for every outbound alert channel. Detection loops call this
    instead of each channel's sender directly, so adding/removing a channel or changing
    the tier rule is a one-line change here rather than an edit repeated at every alert
    call site.

    escalation_level > 0 marks a re-send of an alert nobody acknowledged (see
    escalation_service); an escalated fall is always announced at the confirmed tier
    regardless of score, because by then the point is that it went unanswered, not how
    sure the model was."""
    tier = alert_tier(detection_type, confidence)
    if escalation_level > 0:
        tier = "confirmed"
    send_telegram_message_async(camera_id, camera_name, room_name, detection_type, timestamp,
                                image_path, tier=tier, confidence=confidence,
                                escalation_level=escalation_level)
    # notification_id only reaches LINE: it is what the "รับทราบ" button posts back, so the
    # webhook can mark that exact alert acknowledged and reply with its clip.
    send_line_message_async(camera_id, camera_name, room_name, detection_type, timestamp,
                            image_path, tier=tier, confidence=confidence,
                            escalation_level=escalation_level,
                            notification_id=notification_id)
