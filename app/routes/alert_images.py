import os
from flask import Blueprint, send_from_directory, abort
from app.config import Config

bp = Blueprint('alert_images', __name__, url_prefix='/api/alert-images')

# No @jwt_required() here on purpose: this is loaded via a plain <img src="...">
# in the frontend (MonitorView.vue), which can't attach an Authorization header --
# same reasoning as the unauthenticated MJPEG stream route in stream.py.
@bp.route('/<path:filename>', methods=['GET'])
def get_alert_image(filename):
    alert_dir = getattr(Config, 'ALERT_IMAGE_DIR', None) or '/app/tmp'
    # filename comes straight from the URL; reject any path traversal before
    # touching the filesystem instead of relying solely on send_from_directory.
    safe_name = os.path.basename(filename)
    if safe_name != filename or not os.path.isfile(os.path.join(alert_dir, safe_name)):
        abort(404)
    return send_from_directory(alert_dir, safe_name)
