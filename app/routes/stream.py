from flask import Blueprint, Response, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt, get_jwt_identity
from app.models.camera import Camera
from app.services.stream_service import (
    stream_manager, 
    generate_mjpeg_stream, 
    get_stream_stats,
    cleanup_streams
)
from app.services.logging_service import save_system_log
from app.services.authz import admin_required
from app.services import media_token

# Query-string media tokens must never reach the request log.
media_token.install_log_redaction()

bp = Blueprint('stream', __name__, url_prefix='/api/stream')


def current_user_id_int():
    """-> the verified caller's id as an int, or None.

    JWT identities are strings here (`create_access_token(identity=str(user.id))`), and
    `Camera.user_id` is an integer column, so a raw identity must never be handed to a
    filter -- `filter_by(user_id='4')` is a string-to-integer comparison whose behaviour is the
    database's business, not something this code should depend on.
    """
    try:
        return int(get_jwt_identity())
    except (TypeError, ValueError):
        return None


def get_owned_camera(camera_id):
    """-> the camera, only if the caller's verified token owns it (or they are an admin).

    Requires a route that has already demanded a token. It used to take an optional `user_id`
    and, when that was None, fall back to `Camera.query.get(camera_id)` -- "for testing without
    auth". Since none of these routes required a token and the identity helper turned every
    verification failure into None, that fallback was the normal path: an unauthenticated
    request reached any camera by id. Verified live before the fix -- `GET
    /api/stream/camera/11/status` and `/test` answered HTTP 200 with no token at all, returning
    the camera's name, its source path and a freshly grabbed frame.

    There is no anonymous branch any more. A caller without a token cannot get here.
    """
    user_id = current_user_id_int()
    if user_id is None:
        return None

    from app.models.user import User
    user = User.query.get(user_id)
    if not user:
        return None
    if user.is_admin():
        return Camera.query.get(camera_id)
    return Camera.query.filter_by(id=camera_id, user_id=user_id).first()


@bp.route('/camera/<int:camera_id>/media-token', methods=['POST'])
@jwt_required()
def issue_media_token(camera_id):
    """A short-lived credential for the live view of ONE camera -- see app/services/media_token.

    Requires the normal login and ownership (admins excepted), exactly as the other stream
    routes do; the token it returns can then be put in an <img src>, which cannot carry a header.
    """
    camera = get_owned_camera(camera_id)
    if not camera:
        return jsonify({'error': f'No camera found with ID {camera_id}.'}), 404
    token = media_token.issue(current_user_id_int(), camera_id, get_jwt().get('jti'))
    resp = jsonify({'token': token, 'expires_in': media_token.START_SECONDS,
                    'max_view_seconds': media_token.STREAM_MAX_SECONDS})
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@bp.route('/camera/<int:camera_id>', methods=['GET'])
def stream_camera(camera_id):
    """
    Stream video from a specific camera as MJPEG, to the holder of a media token (?t=).

    This was the last anonymous route: any camera, to anyone who could reach the API. It now
    requires a token from POST .../media-token, checked in full at connect time and before any
    camera work, and the view ends after media_token.STREAM_MAX_SECONDS so a leaked link cannot
    watch indefinitely. The client fetches a fresh token and reconnects.
    """
    camera, status = media_token.verify(request.args.get('t'), camera_id)
    if camera is None:
        return jsonify({'error': 'Not authorised to view this camera.'}), status

    try:
        # Deliberately not logged. Opening a video element is a page view, not a system event,
        # and the dashboard reconnects streams constantly: this single line produced 800+ of
        # the ~820 rows in the system log, burying the AUTH, DETECTION and ERROR entries the
        # page exists to show, and it accumulated for the full 90-day retention. Stream
        # failures are still logged in the except below, which is the part worth keeping.

        # Pose-skeleton overlay is opt-in (?overlay=1) -- it re-runs the AI model on
        # served frames, which competes with celery_worker's actual detection for CPU
        # on this host. Covered by the same token check as the raw view.
        draw_overlay = request.args.get('overlay', '').lower() in ('1', 'true', 'yes')

        # No CORS headers: an <img> does not need them, and a wildcard only widened who could
        # read the response from script.
        deadline = media_token.view_deadline()
        return Response(
            media_token.bounded(
                generate_mjpeg_stream(camera_id, camera.url, camera.name,
                                      draw_overlay=draw_overlay, stop_at=deadline),
                deadline),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
            }
        )

    except Exception as e:
        save_system_log('ERROR', f'Stream error for camera {camera.name}: {str(e)}', 'STREAM', camera.user_id)
        return jsonify({'error': f'Failed to start stream for camera {camera_id}'}), 500


@bp.route('/camera/<int:camera_id>/start', methods=['POST'])
@jwt_required()
def start_camera_stream(camera_id):
    """
    Start a camera stream
    """
    current_user_id = current_user_id_int()
    camera = get_owned_camera(camera_id)
    
    if not camera:
        return jsonify({'error': f'No camera found with ID {camera_id}.'}), 404
    
    try:
        stream = stream_manager.streams.get(camera_id)
        if stream and stream.is_active():
            return jsonify({
                'message': f'Stream for camera "{camera.name}" is already active.',
                'camera_id': camera_id,
                'status': 'active'
            })
        
        # This will create and start the stream
        stream = stream_manager.get_stream(camera_id, camera.url, camera.name)
        if stream:
            save_system_log('INFO', f'Stream started for camera {camera.name}', 'STREAM', current_user_id)
            return jsonify({
                'message': f'Stream started for camera "{camera.name}".',
                'camera_id': camera_id,
                'status': 'started'
            })
        else:
            return jsonify({'error': f'Failed to start stream for camera {camera_id}'}), 500
            
    except Exception as e:
        save_system_log('ERROR', f'Error starting stream for camera {camera.name}: {str(e)}', 'STREAM', current_user_id)
        return jsonify({'error': f'Failed to start stream for camera {camera_id}'}), 500


@bp.route('/camera/<int:camera_id>/stop', methods=['POST'])
@jwt_required()
def stop_camera_stream(camera_id):
    """
    Stop a camera stream
    """
    current_user_id = current_user_id_int()
    camera = get_owned_camera(camera_id)
    
    if not camera:
        return jsonify({'error': f'No camera found with ID {camera_id}.'}), 404
    
    try:
        success = stream_manager.stop_stream(camera_id)
        if success:
            save_system_log('INFO', f'Stream stopped for camera {camera.name}', 'STREAM', current_user_id)
            return jsonify({
                'message': f'Stream stopped for camera "{camera.name}".',
                'camera_id': camera_id,
                'status': 'stopped'
            })
        else:
            return jsonify({
                'message': f'No active stream found for camera "{camera.name}".',
                'camera_id': camera_id,
                'status': 'not_found'
            })
            
    except Exception as e:
        save_system_log('ERROR', f'Error stopping stream for camera {camera.name}: {str(e)}', 'STREAM', current_user_id)
        return jsonify({'error': f'Failed to stop stream for camera {camera_id}'}), 500


@bp.route('/camera/<int:camera_id>/status', methods=['GET'])
@jwt_required()
def get_camera_stream_status(camera_id):
    """
    Get the status of a camera stream
    """
    current_user_id = current_user_id_int()
    camera = get_owned_camera(camera_id)
    
    if not camera:
        return jsonify({'error': f'No camera found with ID {camera_id}.'}), 404
    
    try:
        stream = stream_manager.streams.get(camera_id)
        if stream:
            stats = stream.get_stats()
            return jsonify({
                'camera_id': camera_id,
                'camera_name': camera.name,
                'is_streaming': stream.is_active(),
                'stats': stats
            })
        else:
            return jsonify({
                'camera_id': camera_id,
                'camera_name': camera.name,
                'is_streaming': False,
                'stats': None
            })
            
    except Exception as e:
        save_system_log('ERROR', f'Error getting stream status for camera {camera.name}: {str(e)}', 'STREAM', current_user_id)
        return jsonify({'error': f'Failed to get stream status for camera {camera_id}'}), 500


@bp.route('/stats', methods=['GET'])
@jwt_required()
def get_all_stream_stats():
    """
    Get statistics for all active streams
    """
    current_user_id = current_user_id_int()
    if current_user_id is None:
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        # Only this caller's cameras. The branch that used to sit here returned EVERY camera
        # when the identity was falsy -- "for testing without auth" -- and an admin who owns
        # no cameras still sees only their own here, which is the conservative reading.
        user_cameras = Camera.query.filter_by(user_id=current_user_id).all()
        user_camera_ids = [camera.id for camera in user_cameras]

        all_stats = get_stream_stats()
        
        # Filter stats to only include user's cameras
        user_streams = []
        for stream_stat in all_stats['streams']:
            if stream_stat['camera_id'] in user_camera_ids:
                user_streams.append(stream_stat)
        
        # 'total_system_streams' used to be here: a count of every stream on the box,
        # including other households'. It told the caller nothing about their own cameras and
        # told them something about everyone else's, so it is gone rather than filtered.
        return jsonify({
            'active_streams': len(user_streams),
            'streams': user_streams,
        })
        
    except Exception as e:
        save_system_log('ERROR', f'Error getting stream stats: {str(e)}', 'STREAM', current_user_id)
        return jsonify({'error': 'Failed to get stream statistics'}), 500


@bp.route('/cleanup', methods=['POST'])
@jwt_required()
@admin_required
def cleanup_inactive_streams():
    """
    Cleanup inactive streams (maintenance endpoint)

    Admin-only, because `cleanup_streams()` is global: it walks every stream on the host and
    stops the inactive ones, regardless of who owns them. Requiring a token was not enough --
    any authenticated user could run maintenance across other households' cameras. The
    alternative, scoping the sweep to the caller's own streams, would change what the endpoint
    is for; this is a maintenance action, so it is restricted to the people who do maintenance.
    """
    current_user_id = current_user_id_int()
    
    try:
        # Get count before cleanup
        before_count = len(stream_manager.get_active_streams())
        
        # Perform cleanup
        cleanup_streams()
        
        # Get count after cleanup
        after_count = len(stream_manager.get_active_streams())
        cleaned_count = before_count - after_count
        
        save_system_log('INFO', f'Stream cleanup completed: {cleaned_count} streams cleaned', 'STREAM', current_user_id)
        
        return jsonify({
            'message': 'Stream cleanup completed',
            'streams_cleaned': cleaned_count,
            'active_streams': after_count
        })
        
    except Exception as e:
        save_system_log('ERROR', f'Error during stream cleanup: {str(e)}', 'STREAM', current_user_id)
        return jsonify({'error': 'Failed to cleanup streams'}), 500


# Utility endpoint for testing stream connectivity
@bp.route('/camera/<int:camera_id>/test', methods=['GET'])
@jwt_required()
def test_camera_stream(camera_id):
    """
    Test camera stream connectivity without starting a full stream
    """
    current_user_id = current_user_id_int()
    camera = get_owned_camera(camera_id)
    
    if not camera:
        return jsonify({'error': f'No camera found with ID {camera_id}.'}), 404
    
    try:
        import cv2
        
        # Process URL same way as stream service
        processed_url = camera.url
        if camera.url.startswith('http://localhost:3000/videos/'):
            filename = camera.url.split('/')[-1]
            processed_url = f"/app/Test/{filename}"
        elif '\\' in camera.url and 'videos' in camera.url:
            filename = camera.url.split('\\')[-1]
            processed_url = f"/app/Test/{filename}"
        elif 'Test/' in camera.url and not camera.url.startswith('/app/Test/'):
            filename = camera.url.split('/')[-1]
            processed_url = f"/app/Test/{filename}"
        
        # Test connection
        cap = cv2.VideoCapture(processed_url)
        if not cap.isOpened():
            return jsonify({
                'camera_id': camera_id,
                'camera_name': camera.name,
                'url': camera.url,
                'processed_url': processed_url,
                'status': 'failed',
                'message': 'Unable to open video source'
            }), 400
        
        # Try to read a frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({
                'camera_id': camera_id,
                'camera_name': camera.name,
                'url': camera.url,
                'processed_url': processed_url,
                'status': 'failed',
                'message': 'Unable to read frame from video source'
            }), 400
        
        # Get frame dimensions
        height, width = frame.shape[:2] if frame is not None else (0, 0)
        
        return jsonify({
            'camera_id': camera_id,
            'camera_name': camera.name,
            'url': camera.url,
            'processed_url': processed_url,
            'status': 'success',
            'message': 'Camera stream is accessible',
            'frame_info': {
                'width': width,
                'height': height,
                'channels': frame.shape[2] if len(frame.shape) > 2 else 1
            }
        })
        
    except Exception as e:
        return jsonify({
            'camera_id': camera_id,
            'camera_name': camera.name,
            'url': camera.url,
            'status': 'error',
            'message': f'Error testing camera stream: {str(e)}'
        }), 500
