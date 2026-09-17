import os
from threading import Thread
from datetime import datetime
import pytz
import requests
from app.models.camera import Camera
from app.config import Config
from app.models.line_settings import LineSettings

tz = pytz.timezone('Asia/Bangkok')


def send_line_message_async(camera_id, camera_name, room_name, detection_type, timestamp,
                            image_path, tier="confirmed", confidence=None, escalation_level=0,
                            notification_id=None):
    def _send():
        from app import get_worker_app
        app = get_worker_app()

        with app.app_context():
            camera = Camera.query.get(camera_id)
            if not camera:
                return

            now = datetime.now(tz)

            if "fall" in detection_type:
                risk_level = "red"
                # Two tiers, same channel: a fresh fall alert asks a human to look, because
                # the score cannot tell a fall from a deep bend; the urgent wording is
                # reserved for an alert nobody answered (see notification_service.alert_tier).
                if tier == "confirmed":
                    # Urgent, but still doesn't assert a fall: what is certain at this point
                    # is that an alert has gone unanswered, not what the camera saw.
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
                return

            token = settings.channel_access_token
            user_id = settings.line_user_id
            if not token or not user_id:
                print("[LINE] Missing channel access token or LINE user id -- not sent")
                return

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            if notification_id is not None:
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

            try:
                resp = requests.post(
                    "https://api.line.me/v2/bot/message/push",
                    headers=headers,
                    json={"to": user_id, "messages": messages},
                    timeout=10,
                )
                if resp.status_code == 200:
                    print(f"[Camera {camera_id}] LINE alert sent")
                else:
                    print(f"[Camera {camera_id}] LINE error {resp.status_code}: {resp.text}")
            except Exception as e:
                print(f"[Camera {camera_id}] LINE error: {e}")

    Thread(target=_send).start()
