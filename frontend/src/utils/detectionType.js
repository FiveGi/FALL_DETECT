/**
 * Single source of truth for detection_type <-> Thai display text.
 *
 * This used to be copy-pasted (with drifting text) into CameraManagementView.vue,
 * MonitorView.vue, DashboardView.vue and App.vue -- fixing "fall_v2" once could,
 * and did, leave the other three still showing stale text. Import from here
 * instead of writing another local switch statement.
 */

// Selectable in the "add/edit camera" forms.
export const DETECTION_TYPE_OPTIONS = [
    { value: 'bed_exit', label: 'ตรวจจับการลุกจากเตียง' },
    { value: 'fall', label: 'ตรวจจับการล้ม' },
    { value: 'fall_v2', label: 'ตรวจจับการล้ม (เวอร์ชั่น 3 - YOLO-pose)' },
]

// Every value getDetectionTypeText() may see, including ones that aren't a
// selectable camera.detection_type on their own (e.g. alone_v2 shows up in
// notification/detection-log records, not the camera setup form).
const DETECTION_TYPE_LABELS = {
    bed_exit: 'ตรวจจับการลุกจากเตียง',
    fall: 'ตรวจจับการล้ม',
    fall_detection: 'ตรวจจับการล้ม',
    fall_v2: 'ตรวจจับการล้ม (เวอร์ชั่น 3 - YOLO-pose)',
    alone_v2: 'ตรวจจับผู้สูงอายุอยู่คนเดียว',
}

export function getDetectionTypeText(detectionType) {
    return DETECTION_TYPE_LABELS[detectionType] || detectionType || 'ไม่ระบุ'
}

export const DETECTION_TYPE_FORM_HELP =
    'เลือกว่าต้องการให้ AI ตรวจจับพฤติกรรมแบบไหน - "ตรวจจับการล้ม" คือโมเดลเดิม (MediaPipe), "เวอร์ชั่น 3" คือโมเดลล่าสุด (YOLO-pose) แม่นยำกว่า แนะนำให้ใช้'

// Separate value space from DETECTION_TYPE_LABELS above: these come from alert/
// notification records (app/services/alert_service.py's save_alert_log), tagged
// with a risk level baked into the string (fall_red, alone_yellow, ...), not
// camera.detection_type (bed_exit/fall/fall_v2, the *setting* that produced the
// alert). Mixing them up is exactly what made the popup notification show the
// raw string "fall_red" instead of Thai text -- see App.vue's
// showGlobalNotificationAlert, which used to call getDetectionTypeText() here.
const ALERT_TYPE_LABELS = {
    bed_exit: 'ตรวจจับการลุกจากเตียง',
    alone_yellow: 'ตรวจจับคนอยู่คนเดียว',
    fall_red: 'ตรวจจับการล้ม (อันตราย)',
}

export function getAlertTypeText(alertDetectionType) {
    return ALERT_TYPE_LABELS[alertDetectionType] || 'ตรวจจับการล้ม'
}
