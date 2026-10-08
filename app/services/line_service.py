import os
import uuid
from threading import Thread
from datetime import datetime
import pytz
import requests
from app.models.camera import Camera
from app.config import Config
from app.models.line_settings import LineSettings

tz = pytz.timezone('Asia/Bangkok')


def acknowledge_ready():
    """True when the "รับทราบ" button is CONFIGURED: LINE posts a press to
    PUBLIC_BASE_URL/api/line/webhook, and the webhook rejects it without LINE_CHANNEL_SECRET.
    Configured is not proven reachable -- press it once on a real phone before relying on it.
    One definition for the settings page and the sender, so they can never disagree."""
    return bool((Config.PUBLIC_BASE_URL or '').strip()) and bool(Config.LINE_CHANNEL_SECRET)


def send_line_message_async(camera_id, camera_name, room_name, detection_type, timestamp,
                            image_path, tier="confirmed", confidence=None, escalation_level=0,
                            notification_id=None, still_down_seconds=None, wait=False):
    """Push one alert to every LINE target. Runs on a thread unless `wait` is set; with `wait`
    it runs inline and returns how many targets accepted it (0 = nothing delivered: LINE off,
    no token/target, or every push failed). Escalation waits, so it only counts a re-send that
    actually reached someone -- otherwise a disabled or failing channel used up every
    escalation without a single message delivered."""
    def _send():
        delivered = 0
        from app import get_worker_app
        app = get_worker_app()

        with app.app_context():
            camera = Camera.query.get(camera_id)
            if not camera:
                return 0

            now = datetime.now(tz)

            if "fall" in detection_type:
                risk_level = "red"
                # Two tiers, same channel: a fresh fall alert asks a human to look, because
                # the score cannot tell a fall from a deep bend; the urgent wording is
                # reserved for an alert nobody answered (see notification_service.alert_tier).
                if tier == "confirmed":
                    # Two different facts can reach this tier, and they are not the same
                    # message. "Still on the floor" is something the camera saw; "nobody
                    # answered" is something the system knows. Saying which one it is keeps
                    # the wording honest -- neither of them asserts that a fall happened.
                    if still_down_seconds is not None and escalation_level == 0:
                        event_text = ("🚨 ตรวจพบคนล้ม และยังไม่ลุกขึ้นเลยเป็นเวลา %d วินาที"
                                      " — กรุณาไปดูด่วน!" % int(still_down_seconds))
                    else:
                        event_text = "🚨 ยังไม่มีใครตรวจสอบการแจ้งเตือนล้ม — กรุณาไปดูด่วน!"
                else:
                    risk_level = "yellow"
                    event_text = "❓ อาจมีการล้ม — รบกวนตรวจสอบกล้องด้วยครับ"
            elif "alone" in detection_type:
                risk_level = "yellow"
                event_text = "⚠️ พบว่ามีคนอยู่ตามลำพัง"
            elif detection_type == "bed_exit":
                risk_level = "yellow"
                event_text = "⚠️ ตรวจพบการลุกออกจากเตียง"
            else:
                risk_level = "normal"
                event_text = "สถานะปกติ"

            conf_line = f"\nความมั่นใจ: {confidence:.0%}" if confidence is not None else ""
            escalation_prefix = (
                f"🔁 แจ้งซ้ำครั้งที่ {escalation_level} — ยังไม่มีใครกดรับทราบ\n\n"
                if escalation_level > 0 else ""
            )
            text = (
                f"{escalation_prefix}"
                f"{event_text}\n"
                f"กล้อง: {camera_name}\n"
                f"ห้อง: {room_name}\n"
                f"ความเสี่ยง: {risk_level.upper()}{conf_line}\n"
                f"เวลา: {now.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # Per camera owner, not global: each user configures their own LINE target in
            # the web UI (LineSettings seeds itself from the env vars the first time, so an
            # existing single-user .env deployment behaves exactly as before).
            settings = LineSettings.get_settings(camera.user_id)
            if not settings.enabled:
                # Checked inside the sender rather than at each call site so every caller --
                # detection loops, escalation, the settings test button -- is covered by the
                # one switch, and nothing is pushed to a real phone until it is turned on.
                print(f"[Camera {camera_id}] LINE disabled for this user -- alert not sent")
                return 0

            token = settings.channel_access_token
            targets = settings.targets()
            if not token or not targets:
                print("[LINE] Missing channel access token or a target -- not sent")
                return 0

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            if notification_id is not None and not acknowledge_ready():
                # Without a reachable webhook the button below would do nothing when pressed,
                # and escalation would keep re-sending to someone who believes they answered.
                # Say where acknowledging works instead.
                text += "\n\nกดรับทราบได้ที่หน้าเว็บ เมนู 'มอนิเตอร์'"
                messages = [{"type": "text", "text": text}]
            elif notification_id is not None:
                # A button in the message instead of "go open the web app": the person who
                # gets this alert is usually holding a phone, and the whole point of
                # acknowledging is that it happens within seconds. The postback carries the
                # notification id so the webhook knows exactly which alert was answered;
                # LINE caps altText at 400 chars and shows it on the lock screen.
                messages = [{
                    "type": "template",
                    "altText": text[:400],
                    "template": {
                        "type": "buttons",
                        "text": text[:160],
                        "actions": [{
                            "type": "postback",
                            "label": "รับทราบ",
                            "data": f"ack={notification_id}",
                            "displayText": "รับทราบแล้ว",
                        }],
                    },
                }]
            else:
                messages = [{"type": "text", "text": text}]

            # LINE needs a real public HTTPS URL for images (unlike Telegram's multipart upload) --
            # served from the same /api/alert-images/<filename> route the frontend already uses.
            if image_path and os.path.exists(image_path):
                base_url = (Config.PUBLIC_BASE_URL or "").rstrip("/")
                if base_url:
                    filename = os.path.basename(image_path)
                    image_url = f"{base_url}/api/alert-images/{filename}"
                    messages.append({
                        "type": "image",
                        "originalContentUrl": image_url,
                        "previewImageUrl": image_url,
                    })
                else:
                    print("[LINE] PUBLIC_BASE_URL not set -- sending text only, no image")

            # One push per target, and a failure on one does not stop the others: the whole
            # reason a group is worth having is that the individual phone may be asleep, out
            # of signal or have blocked the bot, and an alert that reached the family group is
            # still an alert that reached someone.
            for target in targets:
                # X-Line-Retry-Key makes a re-send of the SAME alert idempotent at LINE's end: if
                # an earlier push was accepted but its response was lost (timeout), the retry is
                # answered 409 instead of reaching the phone twice -- so a sweep that could not
                # see the first answer cannot turn one alert into a stream (Codex P2). The key
                # is fixed per alert, per message kind and per target.
                h = dict(headers)
                if notification_id is not None:
                    h["X-Line-Retry-Key"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "fall-alert/%s/%s/%s/%s" % (
                        notification_id, escalation_level,
                        "still" if still_down_seconds is not None else "first", target)))
                try:
                    resp = requests.post(
                        "https://api.line.me/v2/bot/message/push",
                        headers=h,
                        json={"to": target, "messages": messages},
                        timeout=10,
                    )
                    if resp.status_code == 409 and "X-Line-Retry-Key" in h:
                        delivered += 1   # accepted earlier under this key; not sent twice
                        print(f"[Camera {camera_id}] LINE alert already accepted for ...{target[-8:]}")
                    elif resp.status_code == 200:
                        delivered += 1
                        print(f"[Camera {camera_id}] LINE alert sent to ...{target[-8:]}")
                    else:
                        print(f"[Camera {camera_id}] LINE error for ...{target[-8:]} "
                              f"{resp.status_code}: {resp.text}")
                except Exception as e:
                    print(f"[Camera {camera_id}] LINE error for ...{target[-8:]}: {e}")
            return delivered

    if wait:
        return _send()
    Thread(target=_send).start()
    return None
