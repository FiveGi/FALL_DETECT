import cv2
import threading
import time
from collections import defaultdict
from io import BytesIO
from typing import Dict, Optional
import numpy as np

class RTSPStreamManager:
    """
    Manager class for handling RTSP streams and converting them to MJPEG for web viewing
    Flow: RTSP Camera → OpenCV VideoCapture → Background Thread → Frame Processing → 
          Memory Storage → MJPEG Generator → HTTP Response → Frontend
    """
    
    def __init__(self):
        self.streams: Dict[int, 'RTSPStream'] = {}
        self.lock = threading.Lock()
    
    def get_stream(self, camera_id: int, camera_url: str, camera_name: str) -> Optional['RTSPStream']:
        """Get or create an RTSP stream for the given camera"""
        with self.lock:
            if camera_id not in self.streams:
                # Create new stream
                stream = RTSPStream(camera_id, camera_url, camera_name)
                if stream.start():
                    self.streams[camera_id] = stream
                    print(f'RTSP stream started for camera {camera_name}')
                else:
                    print(f'Failed to start RTSP stream for camera {camera_name}')
                    return None
            
            return self.streams.get(camera_id)
    
    def stop_stream(self, camera_id: int) -> bool:
        """Stop an RTSP stream"""
        with self.lock:
            if camera_id in self.streams:
                stream = self.streams[camera_id]
                stream.stop()
                del self.streams[camera_id]
                print(f'RTSP stream stopped for camera ID {camera_id}')
                return True
            return False
    
    def get_active_streams(self) -> list:
        """Get list of active stream IDs"""
        with self.lock:
            return list(self.streams.keys())
    
    def cleanup_inactive_streams(self):
        """Remove streams that are no longer active"""
        with self.lock:
            inactive_streams = []
            for camera_id, stream in self.streams.items():
                if not stream.is_active():
                    inactive_streams.append(camera_id)
            
            for camera_id in inactive_streams:
                self.streams[camera_id].stop()
                del self.streams[camera_id]


class RTSPStream:
    """
    Individual RTSP stream handler
    Handles: OpenCV VideoCapture → Background Thread → Frame Processing → Memory Storage
    """
    
    def __init__(self, camera_id: int, url: str, name: str):
        self.camera_id = camera_id
        self.url = url
        self.name = name
        self.cap = None
        self.thread = None
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.last_frame_time = time.time()
        self.fps = 15  # Target FPS for web streaming
        self.frame_interval = 1.0 / self.fps

        # Pose inference is comparatively expensive and backend now runs CPU-throttled
        # (see docker-compose.yml -- celery_worker, the actual detector, needs the CPU
        # more than this live preview does). Re-running it on every single served frame
        # made the preview visibly stutter. Cache the last result and only re-infer every
        # Nth frame; skeleton lag by a couple frames is imperceptible for a live preview.
        self._pose_cache = []
        self._pose_cache_lock = threading.Lock()
        self._pose_frame_counter = 0
        self._pose_infer_every = 3

        # Stream statistics
        self.total_frames = 0
        self.dropped_frames = 0
        self.last_error_time = 0
        self.error_count = 0
        
        # Auto-reconnect settings
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 2.0
    
    def start(self) -> bool:
        """Start the RTSP stream capture"""
        try:
            # Handle different URL formats (same as in camera_manager.py)
            processed_url = self._process_url()
            
            self.cap = cv2.VideoCapture(processed_url)
            if not self.cap.isOpened():
                print(f"Failed to open RTSP stream: {processed_url}")
                return False
            
            # Configure capture properties for better performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Test initial frame capture
            ret, frame = self.cap.read()
            if not ret:
                print(f"Failed to read initial frame from: {processed_url}")
                self.cap.release()
                return False
            
            self.current_frame = frame
            self.running = True
            
            # Start background capture thread
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            print(f"RTSP stream started successfully for camera {self.name}")
            return True
            
        except Exception as e:
            print(f"Error starting RTSP stream for camera {self.name}: {str(e)}")
            if self.cap:
                self.cap.release()
            return False
    
    def stop(self):
        """Stop the RTSP stream capture"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        print(f"RTSP stream stopped for camera {self.name}")
    
    def is_active(self) -> bool:
        """Check if the stream is active"""
        return self.running and self.thread and self.thread.is_alive()
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame (thread-safe)"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_mjpeg_frame(self, draw_bbox: bool = True) -> Optional[bytes]:
        """Get current frame encoded as MJPEG with optional person bounding boxes"""
        frame = self.get_frame()
        if frame is None:
            return None
        
        try:
            # Draw pose skeletons if requested
            if draw_bbox:
                frame = self._draw_pose_skeleton(frame)
            
            # Resize frame for web streaming (optional, for performance)
            height, width = frame.shape[:2]
            if width > 640:  # Resize if too large
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                return buffer.tobytes()
            
        except Exception as e:
            print(f"Error encoding frame for camera {self.name}: {str(e)}")
        
        return None
    
    # COCO-17 skeleton edges (index-identical to the pose model's output --
    # see app/detection/v3_fall_detection.py's module docstring).
    _SKELETON_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),          # head
        (5, 6),                                   # shoulders
        (5, 7), (7, 9),                           # left arm
        (6, 8), (8, 10),                          # right arm
        (5, 11), (6, 12), (11, 12),               # torso
        (11, 13), (13, 15),                       # left leg
        (12, 14), (14, 16),                       # right leg
    ]
    _KEYPOINT_CONF_THRESHOLD = 0.3

    def _draw_pose_skeleton(self, frame: np.ndarray) -> np.ndarray:
        """Draw the COCO-17 pose skeleton using the same YOLO-pose model that
        drives fall detection (V3PoseFallDetector), instead of a separate
        bounding-box-only person detector -- one model doing double duty for
        both detection and the live preview, rather than two."""
        try:
            from app.services.model_manager import model_manager

            h, w = frame.shape[:2]

            with self._pose_cache_lock:
                self._pose_frame_counter += 1
                should_infer = (self._pose_frame_counter % self._pose_infer_every) == 1

            if should_infer:
                fall_detector = model_manager.get_v3_fall_detector()
                if fall_detector is None:
                    return frame
                people = fall_detector.extract_all_keypoints(frame)
                with self._pose_cache_lock:
                    self._pose_cache = people
            else:
                with self._pose_cache_lock:
                    people = self._pose_cache

            for kpts17, _hip_center in people:
                pts = [
                    (int(x * w), int(y * h)) if conf >= self._KEYPOINT_CONF_THRESHOLD else None
                    for x, y, conf in kpts17
                ]
                visible = [p for p in pts if p is not None]

                if visible:
                    xs = [p[0] for p in visible]
                    ys = [p[1] for p in visible]
                    pad_x = int((max(xs) - min(xs)) * 0.15) + 10
                    pad_y = int((max(ys) - min(ys)) * 0.1) + 10
                    x1, y1 = max(0, min(xs) - pad_x), max(0, min(ys) - pad_y)
                    x2, y2 = min(w, max(xs) + pad_x), min(h, max(ys) + pad_y)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for a, b in self._SKELETON_EDGES:
                    if pts[a] is not None and pts[b] is not None:
                        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

                for pt in pts:
                    if pt is not None:
                        cv2.circle(frame, pt, 3, (0, 200, 255), -1)

            count_text = f"Persons: {len(people)}"
            cv2.putText(frame, count_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error drawing pose skeleton: {str(e)}")

        return frame
    
    def get_stats(self) -> dict:
        """Get stream statistics"""
        return {
            'camera_id': self.camera_id,
            'name': self.name,
            'is_active': self.is_active(),
            'total_frames': self.total_frames,
            'dropped_frames': self.dropped_frames,
            'error_count': self.error_count,
            'fps': self.fps,
            'last_frame_time': self.last_frame_time
        }
    
    def _process_url(self) -> str:
        """Process camera URL (same logic as camera_manager.py)"""
        processed_url = self.url
        
        # Handle local video files
        if self.url.startswith('http://localhost:3000/videos/'):
            filename = self.url.split('/')[-1]
            processed_url = f"/app/Test/{filename}"
        elif '\\' in self.url and 'videos' in self.url:
            filename = self.url.split('\\')[-1]
            processed_url = f"/app/Test/{filename}"
        elif 'Test/' in self.url and not self.url.startswith('/app/Test/'):
            filename = self.url.split('/')[-1]
            processed_url = f"/app/Test/{filename}"
        
        return processed_url
    
    def _capture_loop(self):
        """Background thread for continuous frame capture"""
        reconnect_attempts = 0
        
        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    if not self._reconnect():
                        reconnect_attempts += 1
                        if reconnect_attempts >= self.max_reconnect_attempts:
                            print(f"Max reconnection attempts reached for camera {self.name}")
                            break
                        time.sleep(self.reconnect_delay)
                        continue
                    reconnect_attempts = 0
                
                ret, frame = self.cap.read()
                
                if not ret:
                    self.dropped_frames += 1
                    
                    # Check if this is a video file that ended
                    if not self.url.startswith(('rtsp', 'rtmp')):
                        # For video files, loop back to beginning
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        # For RTSP streams, try to reconnect
                        print(f"Failed to read frame from RTSP stream: {self.name}")
                        self._reconnect()
                        continue
                
                # Update frame with thread safety
                with self.frame_lock:
                    self.current_frame = frame
                
                self.total_frames += 1
                self.last_frame_time = time.time()
                
                # Control frame rate
                time.sleep(self.frame_interval)
                
            except Exception as e:
                self.error_count += 1
                current_time = time.time()
                
                # Rate limit error logging
                if current_time - self.last_error_time > 10.0:
                    print(f"Error in capture loop for camera {self.name}: {str(e)}")
                    self.last_error_time = current_time
                
                time.sleep(1.0)
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to the stream"""
        try:
            if self.cap:
                self.cap.release()
            
            processed_url = self._process_url()
            self.cap = cv2.VideoCapture(processed_url)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                print(f"Successfully reconnected to camera {self.name}")
                return True
            
        except Exception as e:
            print(f"Reconnection failed for camera {self.name}: {str(e)}")
        
        return False


# Global stream manager instance
stream_manager = RTSPStreamManager()


def generate_mjpeg_stream(camera_id: int, camera_url: str, camera_name: str):
    """
    Generator function for MJPEG streaming
    Returns: MJPEG stream for HTTP response
    """
    stream = stream_manager.get_stream(camera_id, camera_url, camera_name)
    if not stream:
        # Return empty response if stream cannot be created
        yield b''
        return
    
    try:
        while stream.is_active():
            frame_data = stream.get_mjpeg_frame()
            
            if frame_data:
                # MJPEG format for streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            else:
                # If no frame available, send empty frame or wait
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Error in MJPEG stream generation for camera {camera_id}: {str(e)}")
    finally:
        # Don't automatically stop the stream here, let the route handler decide
        pass


def get_stream_stats() -> dict:
    """Get statistics for all active streams"""
    streams_stats = []
    for camera_id in stream_manager.get_active_streams():
        stream = stream_manager.streams.get(camera_id)
        if stream:
            streams_stats.append(stream.get_stats())
    
    return {
        'active_streams': len(streams_stats),
        'streams': streams_stats
    }


def cleanup_streams():
    """Cleanup inactive streams (can be called periodically)"""
    stream_manager.cleanup_inactive_streams()
