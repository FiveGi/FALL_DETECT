"""What detector is this system actually running, and what did that configuration score?

Every setting here is read from the environment when the worker imports the detector, which
means the answer to "what is running" is not in the database and not in any config page -- it
is in whichever container happens to be up. That has been a real source of confusion: an
accuracy figure quoted from the README describes one configuration, and there is no way from
inside the product to tell whether that is the configuration running.

So this reports the live values and, for exactly that combination, the measurement it came
from. The measurements live in `tools/check_config_coherence.MEASURED`, which is the same table
the pre-deploy check uses; there is no second copy to drift.

**Read-only on purpose.** A form that set these would be worse than none: they are read at
import, so a saved change would appear to take effect and not have, and several of them are not
independent -- the input size and the frame rate are one decision on CPU, and the threshold is
not independent of the window or the rate. A combination nobody has measured is a detector
nobody has tested, which is why check_config_coherence refuses to pass one.
"""
import importlib.util
import os
import sys

from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required

bp = Blueprint('detector_info', __name__, url_prefix='/api/detector')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _measured_table():
    """The same table tools/check_config_coherence.py checks against, loaded not copied."""
    path = os.path.join(ROOT, 'tools', 'check_config_coherence.py')
    if not os.path.exists(path):
        return {}
    spec = importlib.util.spec_from_file_location('_coherence', path)
    module = importlib.util.module_from_spec(spec)
    # The tool runs its checks under __main__ only, so importing it is side-effect free.
    sys.modules['_coherence'] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        return {}
    return getattr(module, 'MEASURED', {})


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

    key = (config.get('input_size'), config.get('window_frames'), config.get('target_fps'),
           config.get('partial_window_from'), tuple(config.get('preprocess') or ('off',)))
    measured = _measured_table().get(key) if config.get('target_fps') else None
    fps = config.get('target_fps')
    config = dict(config)
    config['window_seconds'] = round(config['window_frames'] / fps, 2) if fps else None
    # The honest part. A missing entry is not a formatting problem: it means this exact
    # combination has never been run end to end, so no accuracy figure describes it.
    config['measured'] = measured
    config['measured_known'] = measured is not None
    return jsonify({'success': True, 'data': config}), 200
