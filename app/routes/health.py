from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.models.camera import Camera

bp = Blueprint('health', __name__, url_prefix='/api/health')

@bp.route('', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'The server is running and operational.'}), 200

@bp.route('/camera-detection-service', methods=['GET'])
@jwt_required()
def check_camera_detection_service():
    """
    Check the status of camera detection services.
    Returns overall service status and individual camera statuses.
    """
    current_user_id = int(get_jwt_identity())
    
    cameras = Camera.query.filter_by(user_id=current_user_id).all()
    
    if not cameras:
        return jsonify({
            'service_status': 'no_cameras',
            'message': 'No cameras configured for this user.',
            'total_cameras': 0,
            'active_cameras': 0,
            'inactive_cameras': 0,
            'cameras': []
        }), 200
    
    # `is_active` is a SETTING -- "this camera should be monitored" -- and this endpoint used to
    # report it as a FACT: `'running' if camera.is_active`. Those come apart exactly when it
    # matters. A worker restart, a crashed task or a camera whose stream ended leaves the row
    # active with nothing watching, and a health check that reads the row says everything is
    # fine. That is the failure this endpoint exists to catch, and it was the one thing it
    # could not see.
    #
    # `running_camera_ids()` asks the workers what they are actually running. It returns None
    # rather than an empty set when they cannot be reached, so "nothing is running" stays
    # distinguishable from "I could not find out" -- and this reports the difference instead of
    # collapsing it into a reassuring answer.
    from app.services.detection_dispatch import running_camera_ids

    running_ids = running_camera_ids()
    celery_status = 'unknown' if running_ids is None else 'running'

    by_status = {'running': [], 'stopped': [], 'not_running': [], 'unknown': []}
    for camera in cameras:
        if not camera.is_active:
            status = 'stopped'
        elif running_ids is None:
            status = 'unknown'
        elif camera.id in running_ids:
            status = 'running'
        else:
            # Configured to be monitored, and nothing is monitoring it.
            status = 'not_running'
        by_status[status].append({
            'id': camera.id,
            'name': camera.name,
            'room_name': camera.room_name,
            'detection_type': camera.detection_type,
            'is_active': camera.is_active,
            'status': status,
        })

    total_cameras = len(cameras)
    running_count = len(by_status['running'])
    stalled_count = len(by_status['not_running'])
    unknown_count = len(by_status['unknown'])

    if unknown_count:
        service_status = 'unknown'
        message = ('Could not reach the detection workers, so the state of '
                   f'{unknown_count} camera(s) marked active is unknown.')
    elif stalled_count:
        # Named first and loudly: this is the state that looks healthy and is not.
        service_status = 'stalled'
        message = (f'{stalled_count} camera(s) are set to monitor but no detection task is '
                   'running for them.')
    elif running_count == 0:
        service_status = 'all_stopped'
        message = 'All camera detection services are stopped.'
    elif running_count == total_cameras:
        service_status = 'all_running'
        message = 'All camera detection services are running.'
    else:
        service_status = 'partially_running'
        message = f'{running_count} out of {total_cameras} camera detection services are running.'

    return jsonify({
        'service_status': service_status,
        'message': message,
        'celery_worker_status': celery_status,
        'total_cameras': total_cameras,
        # Kept for anything reading the old shape, but now counted from what is running rather
        # than from the setting.
        'active_cameras': running_count,
        'inactive_cameras': len(by_status['stopped']),
        'stalled_cameras': stalled_count,
        'unknown_cameras': unknown_count,
        'cameras': (by_status['not_running'] + by_status['unknown']
                    + by_status['running'] + by_status['stopped']),
    }), 200 