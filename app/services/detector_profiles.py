"""The detector settings a person may choose from, and what each one scored.

**Named profiles, not a form of knobs, and the difference matters.** Input size and frame rate
are one decision on CPU, not two: the classifier's window is a fixed number of frames, so the
rate decides how much real time it covers, and the input size decides whether the pose model
can see the person at all. The alerting threshold is not independent of either. A page that let
somebody pick any combination would let them build a detector nobody has ever tested, and it
would look exactly as trustworthy as one that had been.

So a profile is a whole configuration that has been measured end to end, it ships with the
numbers it scored, and `tools/check_config_coherence.py` refuses to pass anything that is not
in its MEASURED table -- which is where these numbers come from, rather than a second copy.

**Applying one is a restart, not a save.** Every value here is read when the worker imports the
detector, so writing them to the database would change nothing until something restarted. They
are written to `.env` and the caller is told, plainly, that the detection workers have to come
back before it takes effect.

The server this is built for has no GPU, so the CPU profiles come first and are the ones with
advice attached.
"""
import os
import re

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), '.env')

# key -> (label, settings, what it scored, when to choose it)
PROFILES = {
    'cpu_balanced': {
        'label': 'CPU server — balanced, full frame',
        'label_th': 'เครื่อง CPU — สมดุล ภาพเต็ม',
        'measured_th': 'ชุดทดสอบ URFD: ตรวจพบการล้ม 45/60; คลิปที่ไม่ล้มซึ่งแยกไว้ทดสอบ ไม่แจ้งเตือนผิด: 43/56',
        'note_th': 'ค่าเริ่มต้นของระบบ ต้องใช้ภาพความละเอียดต่ำจากช่องภาพรองของกล้อง',
        'hardware': 'cpu',
        'env': {'V3_IMGSZ': '320', 'V3_TARGET_FPS': '8', 'V3_PARTIAL_MIN': '4',
                'V3_THRESHOLD': '0.65', 'V3_PREPROCESS': 'auto',
                'V3_PREPROCESS_DARK_BELOW': '70', 'V3_ROI_IMGSZ': '0'},
        'measured': 'URFD 45/60 falls; held-out clean 43/56',
        'note': 'The previous default. Needs the camera pointed at its low-resolution substream.',
    },
    'cpu_fewer_false_alarms': {
        'label': 'CPU server — fewer false alarms',
        'label_th': 'เครื่อง CPU — ลดการแจ้งเตือนผิด',
        'measured_th': 'ชุดทดสอบ URFD: ตรวจพบการล้ม 44/60; คลิปที่ไม่ล้มซึ่งแยกไว้ทดสอบ ไม่แจ้งเตือนผิด: 44/56',
        'note_th': 'ตรวจพบการล้มน้อยลง 1 ครั้ง แต่แจ้งเตือนผิดน้อยลง 1 ครั้ง '
                   'เลือกเมื่อครอบครัวถูกปลุกบ่อยเกินไป ไม่ใช่เพื่อให้ตัวเลขดูดีขึ้น',
        'hardware': 'cpu',
        'env': {'V3_IMGSZ': '320', 'V3_TARGET_FPS': '8', 'V3_PARTIAL_MIN': '4',
                'V3_THRESHOLD': '0.70', 'V3_PREPROCESS': 'auto',
                'V3_PREPROCESS_DARK_BELOW': '70', 'V3_ROI_IMGSZ': '0'},
        'measured': 'URFD 44/60 falls; held-out clean 44/56',
        'note': 'One fewer fall, one fewer false alarm. Choose it if the family is being woken '
                'too often, never to make the numbers look better.',
    },
    'cpu_catch_more': {
        'label': 'CPU server — catch more, accept more false alarms',
        'label_th': 'เครื่อง CPU — ตรวจพบการล้มมากขึ้น แต่แจ้งเตือนผิดมากขึ้น',
        'measured_th': 'ชุดทดสอบ URFD: ตรวจพบการล้ม 49/60; คลิปที่ไม่ล้มซึ่งแยกไว้ทดสอบ ไม่แจ้งเตือนผิด: 40/56',
        'note_th': 'ตรวจพบการล้มเพิ่ม 4 ครั้ง แลกกับการแจ้งเตือนผิดเพิ่ม 5 ครั้ง '
                   'เหมาะกับผู้ที่อยู่บ้านคนเดียว',
        'hardware': 'cpu',
        'env': {'V3_IMGSZ': '320', 'V3_TARGET_FPS': '8', 'V3_PARTIAL_MIN': '4',
                'V3_THRESHOLD': '0.50', 'V3_PREPROCESS': 'auto',
                'V3_PREPROCESS_DARK_BELOW': '70', 'V3_ROI_IMGSZ': '0'},
        'measured': 'URFD 49/60 falls; held-out clean 40/56',
        'note': 'Four more falls for five more false alarms. For somebody living alone with '
                'nobody else in the house.',
    },
    'cpu_no_preprocessing': {
        'label': 'CPU server — main stream, no image cleanup',
        'label_th': 'เครื่อง CPU — ใช้ภาพหลัก ไม่ปรับภาพ',
        'measured_th': 'ชุดทดสอบ URFD: ตรวจพบการล้ม 45/60; คลิปที่ไม่ล้มซึ่งแยกไว้ทดสอบ ไม่แจ้งเตือนผิด: 41/56',
        'note_th': 'ใช้เมื่อกล้องไม่มีช่องภาพรองความละเอียดต่ำเท่านั้น การปรับภาพ 1080p '
                   'ใช้เวลา 38% ของเวลาที่มีต่อภาพ ทำให้เครื่องนี้ตรวจภาพได้ช้าลงและอาจพลาดการล้ม',
        'hardware': 'cpu',
        'env': {'V3_IMGSZ': '320', 'V3_TARGET_FPS': '8', 'V3_PARTIAL_MIN': '4',
                'V3_THRESHOLD': '0.65', 'V3_PREPROCESS': 'off', 'V3_ROI_IMGSZ': '0'},
        'measured': 'URFD 45/60 falls; held-out clean 41/56',
        'note': 'Only if the camera cannot give a low-resolution substream. Cleaning up a '
                '1080p frame costs 38% of the per-frame budget, and on this machine frame rate '
                'is recall.',
    },
    'cpu_far_people': {
        'label': 'CPU server — crop around each person (recommended, default)',
        'label_th': 'เครื่อง CPU — คนอยู่ไกลกล้อง (ตัดภาพเฉพาะรอบตัวคน) (แนะนำ ค่าเริ่มต้น)',
        'measured_th': 'ชุดทดสอบ URFD: ตรวจพบการล้ม 46/60; คลิปที่ไม่ล้มซึ่งแยกไว้ทดสอบ ไม่แจ้งเตือนผิด: 42/56; '
                       'คลิปของเจ้าของระบบ: 34/75 เทียบกับ 28/75',
        'note_th': 'ตัดภาพรอบตัวคนขนาด 256 พิกเซล ช่วยให้เห็นคนที่อยู่ไกลชัดขึ้น '
                   'ใช้เวลาต่อภาพน้อยลงประมาณ 11% ในคลิปรวมจากกล้องมุมกว้างของเจ้าของระบบ '
                   'ตรวจพบการล้ม 34 จาก 75 ครั้ง เทียบกับ 28 ครั้ง โดยผล URFD ใกล้เคียงเดิม '
                   'ข้อควรระวัง: มองหาคนที่เพิ่งเข้าภาพทุก 8 ภาพ (ประมาณ 1 วินาที)',
        'hardware': 'cpu',
        'env': {'V3_IMGSZ': '320', 'V3_TARGET_FPS': '8', 'V3_PARTIAL_MIN': '4',
                'V3_THRESHOLD': '0.65', 'V3_PREPROCESS': 'auto',
                'V3_PREPROCESS_DARK_BELOW': '70', 'V3_ROI_IMGSZ': '256',
                'V3_ROI_FULL_EVERY': '8'},
        'measured': 'URFD 46/60 falls; held-out clean 42/56; owner clips 34/75 vs 28/75',
        'note': 'Pose runs on a crop around the people at 256 px, so a small, distant person '
                'gets more pixels, and it is ~11% cheaper per frame. On the owner\'s compilation '
                'footage (wide doorbell-style cameras) it catches 34 of 75 falls against 28, '
                'with URFD near-unchanged. Risk: someone entering the frame is only looked for '
                'on the full-frame pass every 8 frames (about 1 s).',
    },
    'gpu': {
        'label': 'Machine with an NVIDIA GPU',
        'label_th': 'เครื่องที่มีการ์ดจอ NVIDIA',
        'measured_th': 'ชุดทดสอบ URFD: ตรวจพบการล้ม 56/60; คลิปที่ไม่ล้มซึ่งแยกไว้ทดสอบ ไม่แจ้งเตือนผิด: 41/56',
        'note_th': 'ต้องใช้ไฟล์ตั้งค่าเสริมสำหรับ GPU ของ Docker Compose ใช้กับเซิร์ฟเวอร์นี้ไม่ได้',
        'hardware': 'gpu',
        'env': {'V3_IMGSZ': '960', 'V3_TARGET_FPS': '20', 'V3_PARTIAL_MIN': '4',
                'V3_THRESHOLD': '0.65', 'V3_PREPROCESS': 'auto',
                'V3_PREPROCESS_DARK_BELOW': '70', 'V3_ROI_IMGSZ': '0'},
        'measured': 'URFD 56/60 falls; held-out clean 41/56',
        'note': 'Needs the GPU compose overlay. Not this server.',
    },
}

MANAGED_KEYS = sorted({k for p in PROFILES.values() for k in p['env']})


def _same_setting(a, b):
    """Is this the same setting value, written two ways?

    Compared as NUMBERS when both sides are numbers, because they arrive from different places
    in different spellings. `/api/detector` builds its view of the running configuration with
    `'%g' % threshold`, which renders 0.70 as `"0.7"` and 0.50 as `"0.5"`, while the table below
    stores `'0.70'` and `'0.50'`. A string comparison therefore made two of the five profiles --
    both threshold variants -- **impossible to recognise as the running one**, and the page then
    told the operator that nothing matched and their detector was unmeasured. That warning is
    worth having when it is true, which is exactly why it must not fire when it is false.
    """
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


# A setting that is absent means its default, not "something else": an environment with no
# V3_ROI_IMGSZ is running with the crop off, which is what '0' says.
_UNSET_MEANS = {'V3_ROI_IMGSZ': '0'}


def current_profile(env=None):
    """-> the key of the profile the environment matches, or None if it matches none.

    None is a real answer and the UI shows it as one: somebody may have set these by hand, and
    a page that silently picked the nearest profile would be describing a detector that is not
    running.
    """
    env = env if env is not None else os.environ
    for key, profile in PROFILES.items():
        if all(_same_setting(env.get(k, _UNSET_MEANS.get(k, '')), v)
               for k, v in profile['env'].items()):
            return key
    return None


def read_env_file(path=None):
    path = path or ENV_PATH
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            out[key.strip()] = value.strip()
    return out


def apply_profile(key, path=None):
    """Write a profile's settings into .env. -> the settings written.

    Rewrites the keys this module manages and leaves every other line of the file exactly as it
    was, comments included: .env holds credentials and a deployment's own choices, and a
    settings page has no business reformatting it.
    """
    if key not in PROFILES:
        raise ValueError('unknown profile: %s' % key)
    path = path or ENV_PATH
    wanted = dict(PROFILES[key]['env'])
    # A profile that does not set a key must CLEAR it, or leftovers from the previous profile
    # survive and the result is a combination no profile describes.
    for managed in MANAGED_KEYS:
        wanted.setdefault(managed, None)

    lines = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as fh:
            lines = fh.read().splitlines()

    seen = set()
    out = []
    for line in lines:
        match = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=', line)
        name = match.group(1) if match else None
        if name in wanted:
            seen.add(name)
            if wanted[name] is not None:
                out.append('%s=%s' % (name, wanted[name]))
            # else: drop the line, which is how a key gets cleared
            continue
        out.append(line)

    added = [(k, v) for k, v in wanted.items() if v is not None and k not in seen]
    if added:
        out.append('')
        out.append('# Detector profile: %s' % PROFILES[key]['label'])
        out.extend('%s=%s' % (k, v) for k, v in added)

    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(out).rstrip('\n') + '\n')
    return PROFILES[key]['env']
