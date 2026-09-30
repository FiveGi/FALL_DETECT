"""What detector is this system actually running, and what did that configuration score?

Every setting here is read from the environment when the worker imports the detector, which
means the answer to "what is running" is not in the database and not in any config page -- it
is in whichever container happens to be up. That has been a real source of confusion: an
accuracy figure quoted from the README describes one configuration, and there is no way from
inside the product to tell whether that is the configuration running.

So this reports the live values and, for exactly that combination, the measurement it came
from. The measurements live in `tools/check_config_coherence.MEASURED`, which is the same table
the pre-deploy check uses; there is no second copy to drift.

Settings can be changed, but only by choosing a whole **profile** that has been measured end to
end -- see app/services/detector_profiles. A form of independent knobs would let somebody build
a detector nobody has tested, and it would look exactly as trustworthy as one that had been:
input size and frame rate are one decision on CPU, and the threshold is not independent of
either. Applying a profile writes .env and needs the detection workers restarted, which the
response says plainly rather than pretending the change is already live.
"""
import importlib.util
import os
import sys

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from app.services.authz import admin_required

bp = Blueprint('detector_info', __name__, url_prefix='/api/detector')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_COHERENCE = None


def _coherence_module():
    """tools/check_config_coherence.py, loaded not copied -- for its MEASURED table AND its
    measured_key(), so this page and the check can never build the key two different ways.

    Cached after the first call: the file is a constant on disk, and re-executing it on every
    request would make a status page do work proportional to how often anyone looks at it.
    """
    global _COHERENCE
    if _COHERENCE is not None:
        return _COHERENCE
    path = os.path.join(ROOT, 'tools', 'check_config_coherence.py')
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location('_coherence', path)
    module = importlib.util.module_from_spec(spec)
    # The tool runs its checks under __main__ only, so importing it is side-effect free.
    sys.modules['_coherence'] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    _COHERENCE = module
    return _COHERENCE


def _measured_table():
    module = _coherence_module()
    return getattr(module, 'MEASURED', {}) if module else {}


@bp.route('', methods=['GET'])
@jwt_required()
def detector_info():
    from app.services.detection_dispatch import read_detector_config

    config = read_detector_config()
    if config is None:
        # Not an error and not something to paper over with defaults: nothing is running the
        # detector, or the worker that was has not started since this was added.
        return jsonify({
            'success': True,
            'data': None,
            'message': 'No detection worker has reported its configuration yet. '
                       'Start the worker, or restart it if it has been running since before '
                       'this was added.'
        }), 200

    coherence = _coherence_module()
    key = coherence.measured_key(
        config.get('input_size'), config.get('window_frames'), config.get('target_fps'),
        config.get('partial_window_from'), config.get('preprocess'), config.get('threshold'),
        config.get('roi_input_size'), config.get('roi_full_every')) if coherence else None
    measured = _measured_table().get(key) if (key and config.get('target_fps')) else None
    fps = config.get('target_fps')
    config = dict(config)
    config['window_seconds'] = round(config['window_frames'] / fps, 2) if fps else None
    # The honest part. A missing entry is not a formatting problem: it means this exact
    # combination has never been run end to end, so no accuracy figure describes it.
    config['measured'] = measured
    config['measured_known'] = measured is not None
    return jsonify({'success': True, 'data': config}), 200


@bp.route('/profiles', methods=['GET'])
@jwt_required()
def list_profiles():
    """Every configuration somebody may choose, with what it actually scored."""
    from app.services.detection_dispatch import read_detector_config
    from app.services import detector_profiles as dp

    running = read_detector_config() or {}
    # Matched against what the WORKER reported, not this process's environment: the backend
    # container does not share the worker's settings, and matching against its own would name
    # a profile nobody is running.
    as_env = {
        'V3_IMGSZ': str(running.get('input_size', '')),
        'V3_TARGET_FPS': ('%g' % running['target_fps']) if running.get('target_fps') else '',
        'V3_PARTIAL_MIN': str(running.get('partial_window_from', '')),
        'V3_THRESHOLD': ('%g' % running['threshold']) if running.get('threshold') else '',
        'V3_PREPROCESS': ','.join(running.get('preprocess') or []),
        'V3_ROI_IMGSZ': str(running.get('roi_input_size') or 0),
        'V3_ROI_FULL_EVERY': str(running.get('roi_full_every') or 0),
        'V3_PREPROCESS_DARK_BELOW': ('%g' % running['preprocess_dark_below'])
                                    if running.get('preprocess_dark_below') else '',
    }
    return jsonify({'success': True, 'data': {
        'running': dp.current_profile(as_env) if running else None,
        'profiles': [dict(key=k, label=v['label'], hardware=v['hardware'],
                          label_th=v['label_th'], measured_th=v['measured_th'], note_th=v['note_th'],
                          measured=v['measured'], note=v['note'], settings=v['env'])
                     for k, v in dp.PROFILES.items()],
    }}), 200


@bp.route('/profiles', methods=['POST'])
@jwt_required()
@admin_required
def apply_profile():
    """Write a profile to .env. Admins only, and it does not take effect until a restart.

    Authority comes from the user record via `admin_required`, not from a JWT claim. This route
    used to check `if role and 'admin' not in role` against `get_jwt()`, and since no token this
    application issues carries a `role` claim at all, that check was never once able to deny
    anybody: every logged-in user could change the detector settings for every camera on the
    machine. See app/services/authz.py.
    """
    from app.services import detector_profiles as dp

    key = (request.get_json(silent=True) or {}).get('profile')
    try:
        settings = dp.apply_profile(key)
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except OSError as exc:
        return jsonify({'success': False, 'error': 'Could not write .env: %s' % exc}), 500

    return jsonify({'success': True, 'data': {
        'profile': key, 'settings': settings,
        # Said plainly, because the alternative is a page that claims a change is live when the
        # running detector has not heard of it.
        'restart_required': True,
        'message': 'Saved. The detection worker is still running the previous settings until '
                   'it is restarted: docker compose restart celery_worker backend',
    }}), 200
