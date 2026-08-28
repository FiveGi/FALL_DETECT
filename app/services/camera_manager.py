from app import celery
from app.detection.bed_exit import detect_bed_exit, BedExitDetectionState
from app.detection.fall_detection import (
    detect_fall_legacy, FallDetectionState, FallONNXDetector, AlonePersonDetector
)
from app.services.logging_service import save_detection_log, save_system_log
from app.services.telegram_service import send_telegram_message_async
from app.services.alert_service import save_alert_log
from app.models.camera import Camera
from app import db
import cv2
import time
from datetime import datetime
import os
import tempfile
import pytz
from datetime import time as dtime
import threading
import queue
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Tuple, Dict
tz = pytz.timezone('Asia/Bangkok')
from app.config import Config

# How often alone-detection actually runs the person-detector model, in wall-clock seconds,
# rather than on every single frame (there was no stride/skip at all before this). Derived from
# the feature's own existing alert_cooldown of 300s (5 minutes, see detect_alone_with_state) --
# that cooldown is the strongest signal of how time-sensitive this feature actually needs to be,
# and 20s gives ~15 check opportunities within each cooldown window without burning CPU on
# checks far more frequent than the feature can even alert on. Not derived from model speed.
ALONE_DETECTION_CHECK_INTERVAL_S = 20

@dataclass
class TrackState:
    history: Deque[Tuple[int, float, float]]

class AloneDetectionState:
    def __init__(self, history_len=90):
        self.history_len = history_len
        self.states = defaultdict(lambda: TrackState(history=deque(maxlen=history_len)))
        self.frame_id = 0
        self.last_person_count = 0
        self.last_alone_status = "normal"
        self.last_alone_alert_time = 0
        self.last_check_time = 0
        self.last_result = ("normal", 0, "no_person", None, False)  # cached (risk_level, person_count, detection_result, frame, should_alert_alone)

def detect_alone_with_state(frame, state: AloneDetectionState, person_detector: AlonePersonDetector):
    # Was previously called on every single frame with no stride at all. Alone-detection only
    # needs to know "is someone alone" on the order of tens of seconds (see
    # ALONE_DETECTION_CHECK_INTERVAL_S's derivation from the feature's own 5-minute alert
    # cooldown below) -- not every frame. Between checks, return the cached last result.
    now = time.time()
    if now - state.last_check_time < ALONE_DETECTION_CHECK_INTERVAL_S:
        return state.last_result
    state.last_check_time = now

    state.frame_id += 1

    # Was previously two separate person_detector.detect_persons(frame) calls -- one inside
    # detect_alone_only(), one right here -- silently doubling YOLO inference cost per check.
    # Single call now, reused for both the alone/count logic and the (currently unread, kept
    # for future use) per-track position history below.
    person_count, detections = person_detector.detect_persons(frame)
    if person_count == 1:
        detection_result, risk_level = "alone", "yellow"
    elif person_count == 0:
        detection_result, risk_level = "no_person", "normal"
    else:
        detection_result, risk_level = "normal", "normal"
    processed_frame = frame

    for detection in detections:
        tid = detection['track_id']
        x1, y1, x2, y2 = detection['bbox']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        if tid != -1:
            track_state = state.states[tid]
            track_state.history.append((state.frame_id, cx, cy))
    
    current_time = time.time()
    alone_cooldown = 300  # 5 minutes cooldown for alone detection
    should_alert_alone = False
    
    if person_count == 1:
        # Check if enough time has passed since last alone alert
        if current_time - state.last_alone_alert_time >= alone_cooldown:
            should_alert_alone = True
            state.last_alone_alert_time = current_time
    
    state.last_person_count = person_count
    state.last_alone_status = risk_level

    state.last_result = (risk_level, person_count, detection_result, processed_frame, should_alert_alone)
    return state.last_result

@celery.task
def process_fall_detection(camera_id, config):
    from app import create_app
    app = create_app()
    with app.app_context():
        try:
            camera = Camera.query.get(camera_id)
            if not camera or not camera.is_active:
                save_system_log('WARNING', f'Fall detection: Camera {camera_id} not found or inactive', 'DETECTION')
                return
            
            save_system_log('INFO', f'Fall detection started for camera {camera.name}', 'DETECTION', camera.user_id)
            
            # Setup camera stream
            original_url = camera.url
            if camera.url.startswith('http://localhost:3000/videos/'):
                filename = camera.url.split('/')[-1]
                camera.url = f"/app/videos/{filename}"
            elif '\\' in camera.url and 'videos' in camera.url:
                filename = camera.url.split('\\')[-1]
                camera.url = f"/app/videos/{filename}"
            elif 'videos/' in camera.url and not camera.url.startswith('/app/videos/'):
                filename = camera.url.split('/')[-1]
                camera.url = f"/app/videos/{filename}"
            
            is_video_file = not camera.url.startswith(('rtsp', 'rtmp')) and ('.' in camera.url or 'localhost' in original_url)
            
            cap = cv2.VideoCapture(camera.url)
            if not cap.isOpened():
                save_system_log('ERROR', f'Fall detection: Failed to open camera URL: {camera.url}', 'DETECTION', camera.user_id)
                return
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # avoid processing a growing backlog of stale frames on live RTSP sources -- see stream_service.py, which already does this

            if is_video_file:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30
            
            # Initialize fall detection
            state = FallDetectionState()
            model_path = config["FALL_DETECTION_MODEL_PATH"]
            detector = FallONNXDetector(model_path)
            
            last_log_time = 0
            last_detection_result = None
            log_interval = config["LOGGING_INTERVAL"]
            loop_count = 0
            frame_skip_count = 0
            max_retries = 5
            
            while camera.is_active:
                ret, frame = cap.read()
                if not ret:
                    if is_video_file:
                        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        if current_frame >= total_frames - 1 or current_frame == 0:
                            loop_count += 1
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if not ret:
                                frame_skip_count += 1
                                if frame_skip_count >= max_retries:
                                    break
                                time.sleep(1)
                                continue
                            else:
                                frame_skip_count = 0
                        else:
                            frame_skip_count += 1
                            if frame_skip_count >= max_retries:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                frame_skip_count = 0
                                continue
                            time.sleep(0.1)
                            continue
                    else:
                        time.sleep(1)
                        continue
                
                frame_skip_count = 0
                threshold = camera.ai_confidence_threshold
                
                # Fall detection
                detected, confidence, label, img = detect_fall_legacy(
                    frame, state, detector, config, camera=camera, threshold=threshold
                )
                
                # Logging
                now = time.time()
                if now - last_log_time >= log_interval:
                    if last_detection_result != label or label not in ['no_fall', 'normal']:
                        save_detection_log(camera_id, label, confidence)
                        print(f"[Camera {camera_id}] Fall Detection: {label}, Confidence: {confidence:.2f}")
                        last_detection_result = label
                    last_log_time = now
                
                # Alert handling
                if not hasattr(camera, '_last_fall_alert_time'):
                    camera._last_fall_alert_time = 0
                
                notification_cooldown = camera.notification_cooldown
                
                def parse_time(timestr):
                    h, m = map(int, timestr.split(':'))
                    return dtime(h, m)
                    
                now_time = datetime.now(tz).time()
                start_time = parse_time(camera.alert_start_time)
                end_time = parse_time(camera.alert_end_time)
                in_time_window = False
                if start_time < end_time:
                    in_time_window = start_time <= now_time < end_time
                else:
                    in_time_window = now_time >= start_time or now_time < end_time
                
                if detected and in_time_window and (now - camera._last_fall_alert_time > notification_cooldown):
                    alert_dir = getattr(Config, 'ALERT_IMAGE_DIR', None) or '/app/tmp'
                    os.makedirs(alert_dir, exist_ok=True)
                    img_path = os.path.join(alert_dir, f"fall_detect_{camera_id}_{int(time.time())}.jpg")
                    cv2.imwrite(img_path, img)
                    
                    save_alert_log(camera_id, "fall_red", img_path, f"Confidence: {confidence:.2f}")
                    
                    send_telegram_message_async(
                        camera_id, camera.name, camera.room_name, "fall_red", datetime.now(tz).isoformat(), img_path
                    )
                    
                    print(f"[Camera {camera_id}] FALL ALERT: {label}, Confidence: {confidence:.2f}")
                    camera._last_fall_alert_time = now
                
                db.session.refresh(camera)
                if not camera.is_active:
                    break
                
                if is_video_file and fps > 0:
                    time.sleep(1.0 / fps)
            
            cap.release()
            save_system_log('INFO', f'Fall detection session ended for camera {camera.name}', 'DETECTION', camera.user_id)
            
        except Exception as e:
            save_system_log('ERROR', f'Fall detection error for {camera_id}: {str(e)}', 'DETECTION')
            if 'cap' in locals():
                cap.release()
            raise

@celery.task
def process_alone_detection(camera_id, config):
    from app import create_app
    app = create_app()
    with app.app_context():
        try:
            camera = Camera.query.get(camera_id)
            if not camera or not camera.is_active:
                save_system_log('WARNING', f'Alone detection: Camera {camera_id} not found or inactive', 'DETECTION')
                return
            
            save_system_log('INFO', f'Alone detection started for camera {camera.name}', 'DETECTION', camera.user_id)
            
            # Setup camera stream
            original_url = camera.url
            if camera.url.startswith('http://localhost:3000/videos/'):
                filename = camera.url.split('/')[-1]
                camera.url = f"/app/videos/{filename}"
            elif '\\' in camera.url and 'videos' in camera.url:
                filename = camera.url.split('\\')[-1]
                camera.url = f"/app/videos/{filename}"
            elif 'videos/' in camera.url and not camera.url.startswith('/app/videos/'):
                filename = camera.url.split('/')[-1]
                camera.url = f"/app/videos/{filename}"
            
            is_video_file = not camera.url.startswith(('rtsp', 'rtmp')) and ('.' in camera.url or 'localhost' in original_url)
            
            cap = cv2.VideoCapture(camera.url)
            if not cap.isOpened():
                save_system_log('ERROR', f'Alone detection: Failed to open camera URL: {camera.url}', 'DETECTION', camera.user_id)
                return
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # avoid processing a growing backlog of stale frames on live RTSP sources -- see stream_service.py, which already does this

            if is_video_file:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30
            
            # Initialize alone detection
            state = AloneDetectionState()
            person_detector = AlonePersonDetector()
            
            last_log_time = 0
            last_detection_result = None
            last_risk_level = None
            log_interval = config["LOGGING_INTERVAL"]
            loop_count = 0
            frame_skip_count = 0
            max_retries = 5
            
            while camera.is_active:
                ret, frame = cap.read()
                if not ret:
                    if is_video_file:
                        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        if current_frame >= total_frames - 1 or current_frame == 0:
                            loop_count += 1
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if not ret:
                                frame_skip_count += 1
                                if frame_skip_count >= max_retries:
                                    break
                                time.sleep(1)
                                continue
                            else:
                                frame_skip_count = 0
                        else:
                            frame_skip_count += 1
                            if frame_skip_count >= max_retries:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                frame_skip_count = 0
                                continue
                            time.sleep(0.1)
                            continue
                    else:
                        time.sleep(1)
                        continue
                
                frame_skip_count = 0
                
                # Alone detection
                risk_level, person_count, detection_result, img, should_alert_alone = detect_alone_with_state(
                    frame, state, person_detector
                )
                
                # Logging
                now = time.time()
                if now - last_log_time >= log_interval:
                    if (last_detection_result != detection_result or last_risk_level != risk_level or 
                        detection_result not in ['normal', 'no_person'] or risk_level != 'normal'):
                        save_detection_log(camera_id, detection_result, 0.8, risk_level, person_count)
                        print(f"[Camera {camera_id}] Alone Detection: {detection_result}, Risk: {risk_level}, Persons: {person_count}")
                        last_detection_result = detection_result
                        last_risk_level = risk_level
                    last_log_time = now
                
                # Alert handling - Skip telegram notification for yellow alerts (alone detection)
                if not hasattr(camera, '_last_alone_alert_time'):
                    camera._last_alone_alert_time = 0
                
                notification_cooldown = camera.notification_cooldown
                
                def parse_time(timestr):
                    h, m = map(int, timestr.split(':'))
                    return dtime(h, m)
                    
                now_time = datetime.now(tz).time()
                start_time = parse_time(camera.alert_start_time)
                end_time = parse_time(camera.alert_end_time)
                in_time_window = False
                if start_time < end_time:
                    in_time_window = start_time <= now_time < end_time
                else:
                    in_time_window = now_time >= start_time or now_time < end_time
                
                if should_alert_alone and in_time_window and (now - camera._last_alone_alert_time > notification_cooldown):
                    print(f"[Camera {camera_id}] ALONE DETECTED (No notification sent): {detection_result}, Risk: {risk_level}")
                    camera._last_alone_alert_time = now
                
                db.session.refresh(camera)
                if not camera.is_active:
                    break
                
                if is_video_file and fps > 0:
                    time.sleep(1.0 / fps)
            
            cap.release()
            save_system_log('INFO', f'Alone detection session ended for camera {camera.name}', 'DETECTION', camera.user_id)
            
        except Exception as e:
            save_system_log('ERROR', f'Alone detection error for {camera_id}: {str(e)}', 'DETECTION')
            if 'cap' in locals():
                cap.release()
            raise
@celery.task
def process_bed_exit_detection(camera_id, config):
    from app import create_app
    app = create_app()
    with app.app_context():
        try:
            camera = Camera.query.get(camera_id)
            if not camera or not camera.is_active:
                save_system_log('WARNING', f'Bed exit detection: Camera {camera_id} not found or inactive', 'DETECTION')
                return
            
            save_system_log('INFO', f'Bed exit detection started for camera {camera.name}', 'DETECTION', camera.user_id)
            
            # Setup camera stream
            original_url = camera.url
            if camera.url.startswith('http://localhost:3000/videos/'):
                filename = camera.url.split('/')[-1]
                camera.url = f"/app/videos/{filename}"
            elif '\\' in camera.url and 'videos' in camera.url:
                filename = camera.url.split('\\')[-1]
                camera.url = f"/app/videos/{filename}"
            elif 'videos/' in camera.url and not camera.url.startswith('/app/videos/'):
                filename = camera.url.split('/')[-1]
                camera.url = f"/app/videos/{filename}"
            
            is_video_file = not camera.url.startswith(('rtsp', 'rtmp')) and ('.' in camera.url or 'localhost' in original_url)
            
            cap = cv2.VideoCapture(camera.url)
            if not cap.isOpened():
                save_system_log('ERROR', f'Bed exit detection: Failed to open camera URL: {camera.url}', 'DETECTION', camera.user_id)
                return
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # avoid processing a growing backlog of stale frames on live RTSP sources -- see stream_service.py, which already does this

            if is_video_file:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30
            
            # Initialize bed exit detection
            state = BedExitDetectionState()
            
            last_log_time = 0
            last_detection_result = None
            log_interval = config["LOGGING_INTERVAL"]
            loop_count = 0
            frame_skip_count = 0
            max_retries = 5
            
            while camera.is_active:
                ret, frame = cap.read()
                if not ret:
                    if is_video_file:
                        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        if current_frame >= total_frames - 1 or current_frame == 0:
                            loop_count += 1
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if not ret:
                                frame_skip_count += 1
                                if frame_skip_count >= max_retries:
                                    break
                                time.sleep(1)
                                continue
                            else:
                                frame_skip_count = 0
                        else:
                            frame_skip_count += 1
                            if frame_skip_count >= max_retries:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                frame_skip_count = 0
                                continue
                            time.sleep(0.1)
                            continue
                    else:
                        time.sleep(1)
                        continue
                
                frame_skip_count = 0
                threshold = camera.ai_confidence_threshold
                
                # Bed exit detection
                result = detect_bed_exit(
                    frame, state, config, camera=camera, threshold=threshold
                )
                
                detected = False
                confidence = 0.0
                label = "no_detection"
                img = frame
                
                if result is not None:
                    detected, confidence, label, img = result
                
                # Logging
                now = time.time()
                if now - last_log_time >= log_interval:
                    if last_detection_result != label or label not in ['no_detection', 'normal']:
                        save_detection_log(camera_id, label, confidence)
                        print(f"[Camera {camera_id}] Bed Exit Detection: {label}, Confidence: {confidence:.2f}")
                        last_detection_result = label
                    last_log_time = now
                
                # Alert handling
                if not hasattr(camera, '_last_bed_alert_time'):
                    camera._last_bed_alert_time = 0
                
                notification_cooldown = camera.notification_cooldown
                
                def parse_time(timestr):
                    h, m = map(int, timestr.split(':'))
                    return dtime(h, m)
                    
                now_time = datetime.now(tz).time()
                start_time = parse_time(camera.alert_start_time)
                end_time = parse_time(camera.alert_end_time)
                in_time_window = False
                if start_time < end_time:
                    in_time_window = start_time <= now_time < end_time
                else:
                    in_time_window = now_time >= start_time or now_time < end_time
                
                if detected and in_time_window and (now - camera._last_bed_alert_time > notification_cooldown):
                    img_path = f"/app/tmp/bed_exit_detect_{camera_id}_{int(time.time())}.jpg"
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    cv2.imwrite(img_path, img)
                    
                    save_alert_log(camera_id, "bed_exit", img_path, f"Confidence: {confidence:.2f}")
                    
                    send_telegram_message_async(
                        camera_id, camera.name, camera.room_name, "bed_exit", datetime.now(tz).isoformat(), img_path
                    )
                    
                    print(f"[Camera {camera_id}] BED EXIT ALERT: {label}, Confidence: {confidence:.2f}")
                    camera._last_bed_alert_time = now
                
                db.session.refresh(camera)
                if not camera.is_active:
                    break
                
                if is_video_file and fps > 0:
                    time.sleep(1.0 / fps)
            
            cap.release()
            save_system_log('INFO', f'Bed exit detection session ended for camera {camera.name}', 'DETECTION', camera.user_id)
            
        except Exception as e:
            save_system_log('ERROR', f'Bed exit detection error for {camera_id}: {str(e)}', 'DETECTION')
            if 'cap' in locals():
                cap.release()
            raise

# V2 Fall Detection Tasks
@celery.task
def process_v2_fall_detection(camera_id, config):
    """Process V2 fall detection for a camera"""
    from app import create_app
    app = create_app()
    with app.app_context():
        # Active pipeline is v3 (pose-based, CPU-friendly, already trained) not v4 (RF-DETR).
        # v4 needs a GPU to run at usable speed (confirmed too slow even on a good desktop CPU,
        # see SKILL.md SS9) and its checkpoint is 369MB -- too large for a normal git push without
        # Git LFS. v3's model files are a few MB total and already committed. v4's code/weights
        # are left in place for anyone who does have GPU access -- see get_v4_fall_detector().
        # Multi-person: detect_v3_fall_multi tracks up to NUM_POSES people independently per
        # camera (each with their own rolling window/alert state), instead of only ever seeing
        # whichever one person single-pose extraction happened to pick.
        from app.detection.v3_fall_detection import (
            V3MultiPersonFallState,
            detect_v3_fall_multi
        )
        from app.services.model_manager import model_manager

        try:
            camera = Camera.query.get(camera_id)
            if not camera:
                save_system_log('ERROR', f'Camera {camera_id} not found for V2 fall detection', 'DETECTION')
                return

            print(f"[V3 Pose] Starting fall detection session for camera {camera.name}")
            save_system_log('INFO', f'V2 Fall detection session started for camera {camera.name}', 'DETECTION', camera.user_id)

            fall_detector = model_manager.get_v3_fall_detector()
            if fall_detector is None:
                save_system_log('ERROR', f'V3 Fall detector not available for camera {camera.name}', 'DETECTION', camera.user_id)
                print(f"[V3 Pose] Fall detector not available for camera {camera.name}")
                return

            print(f"[V3 Pose] Using pre-loaded fall detector for camera {camera.name}")

            # Initialize detection state
            fall_state = V3MultiPersonFallState()

            # Initialize camera capture
            cap = cv2.VideoCapture(camera.url)
            if not cap.isOpened():
                save_system_log('ERROR', f'Failed to open camera {camera.name} for V2 fall detection', 'DETECTION', camera.user_id)
                return
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # avoid processing a growing backlog of stale frames on live RTSP sources -- see stream_service.py, which already does this

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30  # Default FPS
            
            is_video_file = not camera.url.startswith(('rtsp:', 'rtmp:', 'http:'))
            
            frame_count = 0
            last_log_time = 0
            log_interval = config.get("LOGGING_INTERVAL", 60)  # Default 60 seconds
            
            while camera.is_active:
                ret, frame = cap.read()
                if not ret:
                    if is_video_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                        continue
                    else:
                        break
                
                frame_count += 1
                # Print every 30 frames (about 1 second at 30 fps)
                if frame_count % 30 == 0:
                    print(f"[Camera {camera_id}] V2 Fall Detection - Processing frame {frame_count}")
                
                # Perform V3 pose-based fall detection -- one result per tracked person
                results = detect_v3_fall_multi(frame, fall_state, fall_detector, config, camera)
                any_detected = any(r[1] for r in results)
                # For logging/alerting a single confidence number, use whichever tracked
                # person is most fall-like this frame (the detected one if any, else the max).
                top = max(results, key=lambda r: r[2]) if results else (None, False, 0.0, "no_person", None)
                _, detected, probability, label, _ = top

                # Save detection log periodically
                now = time.time()
                if now - last_log_time >= log_interval:
                    detection_result = 'fall_v2' if any_detected else 'no_fall'
                    risk_level = 'red' if any_detected else 'normal'
                    save_detection_log(
                        camera_id=camera.id,
                        detection_result=detection_result,
                        confidence_score=probability,
                        risk_level=risk_level,
                        person_count=len(results)
                    )
                    print(f"[Camera {camera_id}] V2 Fall Log: {detection_result}, Confidence: {probability:.2f}, People tracked: {len(results)}")
                    last_log_time = now

                if any_detected:

                    # Send alerts
                    current_time = datetime.now(tz)
                    now = time.time()

                    alert_cooldown = getattr(camera, 'alert_cooldown', None) or Config.NOTIFICATION_COOLDOWN
                    last_alert_time = getattr(camera, '_last_fall_v2_alert_time', 0)

                    if now - last_alert_time >= alert_cooldown:
                        # Save image for alert
                        alert_dir = getattr(Config, 'ALERT_IMAGE_DIR', None) or '/app/tmp'
                        os.makedirs(alert_dir, exist_ok=True)
                        img_path = os.path.join(alert_dir, f"v2_fall_detect_{camera_id}_{int(time.time())}.jpg")
                        cv2.imwrite(img_path, frame)

                        save_alert_log(camera.id, 'fall_red', img_path, additional_info={'model': 'v2', 'confidence': probability})
                        send_telegram_message_async(
                            camera.id, camera.name, camera.room_name,
                            'fall_red', current_time, img_path
                        )

                        print(f"[Camera {camera_id}] V2 FALL ALERT: {label}, Confidence: {probability:.2f}, People tracked: {len(results)}")
                        camera._last_fall_v2_alert_time = now
                
                db.session.refresh(camera)
                if not camera.is_active:
                    break
                
                if is_video_file and fps > 0:
                    time.sleep(1.0 / fps)
            
            cap.release()
            save_system_log('INFO', f'V2 Fall detection session ended for camera {camera.name}', 'DETECTION', camera.user_id)
            
        except Exception as e:
            save_system_log('ERROR', f'V2 Fall detection error for {camera_id}: {str(e)}', 'DETECTION')
            if 'cap' in locals():
                cap.release()
            raise

@celery.task
def process_v2_alone_detection(camera_id, config):
    """Process V2 alone detection for a camera"""
    from app import create_app
    app = create_app()
    with app.app_context():
        from app.detection.v2_fall_detection_onnx import (
            detect_v2_alone_only_onnx
        )
        from app.services.model_manager import model_manager
        
        try:
            camera = Camera.query.get(camera_id)
            if not camera:
                save_system_log('ERROR', f'Camera {camera_id} not found for V2 alone detection', 'DETECTION')
                return
            
            # Check if alone detection is enabled for this camera
            if not getattr(camera, 'enable_alone_detection', True):
                print(f"[V2 ONNX] Alone detection is disabled for camera {camera.name}")
                save_system_log('INFO', f'V2 Alone detection skipped (disabled) for camera {camera.name}', 'DETECTION', camera.user_id)
                return
            
            print(f"[V2 ONNX] Starting alone detection session for camera {camera.name}")
            save_system_log('INFO', f'V2 Alone detection session started for camera {camera.name}', 'DETECTION', camera.user_id)
            
            person_detector = model_manager.get_v2_person_detector()
            if person_detector is None:
                save_system_log('ERROR', f'V2 Person detector not available for camera {camera.name}', 'DETECTION', camera.user_id)
                print(f"[V2 ONNX] Person detector not available for camera {camera.name}")
                return
            
            print(f"[V2 ONNX] Using pre-loaded person detector for camera {camera.name}")
            
            cap = cv2.VideoCapture(camera.url)
            if not cap.isOpened():
                save_system_log('ERROR', f'Failed to open camera {camera.name} for V2 alone detection', 'DETECTION', camera.user_id)
                return
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # avoid processing a growing backlog of stale frames on live RTSP sources -- see stream_service.py, which already does this

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30
            
            is_video_file = not camera.url.startswith(('rtsp:', 'rtmp:', 'http:'))
            
            frame_count = 0
            last_log_time = 0
            log_interval = 300
            last_check_time = 0
            last_result = ("normal", 0, "no_person", None)  # cached (risk_level, person_count, detection_result, processed_frame)

            while camera.is_active:
                ret, frame = cap.read()
                if not ret:
                    if is_video_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                        continue
                    else:
                        break

                frame_count += 1
                # Print every 30 frames (about 1 second at 30 fps)
                if frame_count % 30 == 0:
                    print(f"[Camera {camera_id}] V2 Alone Detection - Processing frame {frame_count}")

                # Perform V2 ONNX alone detection -- only actually run the model every
                # ALONE_DETECTION_CHECK_INTERVAL_S seconds (was every single frame with no
                # stride at all); see that constant's definition for why.
                now_check = time.time()
                if now_check - last_check_time >= ALONE_DETECTION_CHECK_INTERVAL_S:
                    last_check_time = now_check
                    last_result = detect_v2_alone_only_onnx(frame, person_detector)
                risk_level, person_count, detection_result, processed_frame = last_result
                
                # Save detection log periodically
                now = time.time()
                if now - last_log_time >= log_interval:
                    save_detection_log(
                        camera_id=camera.id,
                        detection_result=detection_result,
                        confidence_score=1.0,
                        risk_level=risk_level,
                        person_count=person_count
                    )
                    print(f"[Camera {camera_id}] V2 Alone Log: {detection_result}, Risk: {risk_level}, Count: {person_count}")
                    last_log_time = now
                
                if risk_level == "yellow":  # Alone detected
                    
                    # Send alerts with cooldown
                    current_time = datetime.now(tz)
                    now = time.time()
                    
                    alert_cooldown = getattr(camera, 'alert_cooldown', None) or Config.NOTIFICATION_COOLDOWN
                    last_alert_time = getattr(camera, '_last_alone_v2_alert_time', 0)
                    
                    if now - last_alert_time >= alert_cooldown:
                        save_alert_log(camera.id, 'alone_yellow', image_path=None, additional_info={'model': 'v2', 'person_count': person_count})
                        send_telegram_message_async(
                            camera.id, camera.name, camera.room_name,
                            'alone_yellow', current_time, None
                        )
                        
                        print(f"[Camera {camera_id}] V2 ALONE ALERT: {detection_result}, Count: {person_count}")
                        camera._last_alone_v2_alert_time = now
                
                db.session.refresh(camera)
                if not camera.is_active:
                    break
                
                if is_video_file and fps > 0:
                    time.sleep(1.0 / fps)
            
            cap.release()
            save_system_log('INFO', f'V2 Alone detection session ended for camera {camera.name}', 'DETECTION', camera.user_id)
            
        except Exception as e:
            save_system_log('ERROR', f'V2 Alone detection error for {camera_id}: {str(e)}', 'DETECTION')
            if 'cap' in locals():
                cap.release()
            raise

# Legacy task for backward compatibility - now delegates to specific detection tasks
@celery.task
def process_camera(camera_id, detection_type, config):
    """Legacy task that delegates to specific detection tasks"""
    if detection_type == 'bed_exit':
        return process_bed_exit_detection.delay(camera_id, config)
    elif detection_type == 'fall':
        # Start both fall and alone detection tasks
        fall_task = process_fall_detection.delay(camera_id, config)
        alone_task = process_alone_detection.delay(camera_id, config)
        return {'fall_task': fall_task.id, 'alone_task': alone_task.id}
    elif detection_type == 'fall_v2':
        # Start both V2 fall and alone detection tasks
        fall_v2_task = process_v2_fall_detection.delay(camera_id, config)
        alone_v2_task = process_v2_alone_detection.delay(camera_id, config)
        return {'fall_v2_task': fall_v2_task.id, 'alone_v2_task': alone_v2_task.id}
    else:
        # Default to fall detection only
        return process_fall_detection.delay(camera_id, config) 