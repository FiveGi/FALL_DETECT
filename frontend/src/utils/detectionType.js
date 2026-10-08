/**
 * Single source of truth for detection_type <-> Thai display text.
 *
 * This used to be copy-pasted (with drifting text) into CameraManagementView.vue,
 * MonitorView.vue, DashboardView.vue and App.vue -- fixing "fall_v2" once could,
 * and did, leave the other three still showing stale text. Import from here
 * instead of writing another local switch statement.
 */

// Selectable in the "add/edit camera" forms.
//
// The labels name which MODEL each option runs, because they are not variants of one detector:
// 'fall' is the superseded MediaPipe model and 'fall_v2' is the current YOLO-pose one. On URFD
// -- the public set neither was tuned against -- the old model catches 45% of falls and the
// current one 93%. The old label for 'fall' was simply "detect falls", the plainest entry in
// the list, so the weakest detector was also the most obvious thing to pick while the good one
// looked like an experiment. Anyone choosing here is choosing accuracy, so it says so.
export const DETECTION_TYPE_OPTIONS = [
    { value: 'bed_exit', label: 'ตรวจจับการลุกจากเตียง' },
    { value: 'fall_v2', label: 'ตรวจจับการล้ม — โมเดลปัจจุบัน YOLO-pose (แนะนำ)' },
    // 'fall' (MediaPipe) is no longer offered: its task cannot start in the deployed image, and the
    // backend now runs the current detector for it (detection_dispatch). Its label stays below for
    // cameras and records that still carry the old value.
]

// Every value getDetectionTypeText() may see, including ones that aren't a
// selectable camera.detection_type on their own (e.g. alone_v2 shows up in
// notification/detection-log records, not the camera setup form).
const DETECTION_TYPE_LABELS = {
    bed_exit: 'ตรวจจับการลุกจากเตียง',
    fall: 'ตรวจจับการล้ม (โมเดลเก่า MediaPipe)',
    fall_detection: 'ตรวจจับการล้ม (โมเดลเก่า MediaPipe)',
    fall_v2: 'ตรวจจับการล้ม (YOLO-pose)',
    alone_v2: 'ตรวจจับผู้สูงอายุอยู่คนเดียว',
}

export function getDetectionTypeText(detectionType) {
    return DETECTION_TYPE_LABELS[detectionType] || detectionType || 'ไม่ระบุ'
}

export const DETECTION_TYPE_FORM_HELP =
    'เลือกโมเดลที่จะใช้ตรวจจับ — วัดกับชุดข้อมูล URFD ที่ไม่เคยใช้ปรับจูน: '
    + 'YOLO-pose จับการล้มได้ 93% (บนเครื่องที่มีการ์ดจอ) หรือ 75% (บนเซิร์ฟเวอร์ CPU), '
    + 'ส่วนโมเดลเก่า MediaPipe จับได้ 45%'

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
    // Matches the LINE wording. The system does not assert a fall: measured, its
    // highest-scoring alert is a man getting up from a bed, so "(อันตราย)" was a claim the
    // evidence does not support -- and it sat next to a "รอตรวจสอบ" badge saying the opposite.
    fall_red: 'อาจมีการล้ม — ต้องตรวจสอบ',
}

export function getAlertTypeText(alertDetectionType) {
    return ALERT_TYPE_LABELS[alertDetectionType] || 'ตรวจจับการล้ม'
}

// Must match alert_tier() in app/services/notification_service.py -- the backend decides the
// wording of the LINE message with that rule, and this decides the badge shown for the same
// alert in the web UI. If one moves and the other doesn't, the same event reads as urgent in
// chat and "please check" on screen.
//
// The confidence score deliberately plays no part. It is not uninformative with the current
// model -- above 0.80 an alert is 95% real against 88% overall -- but the single
// highest-scoring alert measured, 0.96, is a man getting up from a bed, so the top of the
// range is where the hardest false alarms live (see the backend comment for the table).
// Urgency comes from nobody having acknowledged the alert instead, which is a fact.

/**
 * 'check'     -> a fall alert: staff are asked to look rather than told what happened.
 * 'confirmed' -> nobody acknowledged it and escalation_service re-sent it.
 * Non-fall alerts (bed exit, alone) are advisory by nature and always 'check'.
 */
// Must stay identical to notification_service.alert_tier on the backend: this decides what a
// family is told, and two copies that disagree means the page says one thing and the LINE
// message another. tools/check_alert_rules.py compares them.
//
// stillDownSeconds is a fact about what the camera saw -- the person who triggered the alert
// has not been upright since. It raises the tier for the same reason an unacknowledged alert
// does, and like escalation it can only ever raise it. The inverse, cancelling when somebody
// gets back up, was measured and would suppress 6% of real falls.
export const STILL_DOWN_SECONDS = 10

export function getAlertTier(alertDetectionType, escalationCount = 0, stillDownSeconds = null,
                             threshold = STILL_DOWN_SECONDS) {
    if (!alertDetectionType || !alertDetectionType.includes('fall')) return 'check'
    if (escalationCount > 0) return 'confirmed'
    if (stillDownSeconds != null && stillDownSeconds >= threshold) return 'confirmed'
    return 'check'
}

const ALERT_TIER_LABELS = {
    confirmed: 'ยังไม่มีใครรับทราบ',
    check: 'รอตรวจสอบ',
}

export function getAlertTierText(tier) {
    return ALERT_TIER_LABELS[tier] || ''
}
