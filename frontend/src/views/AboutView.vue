<template>
  <div class="about">
    <div class="about-content card">
      <h1 class="page-title">เกี่ยวกับ V89 Fall Management System</h1>

      <div class="about-section">
        <h2>ระบบตรวจจับความเคลื่อนไหวและแจ้งเตือนอัจฉริยะ</h2>
        <p>
          V89 Fall Management System เป็นระบบตรวจจับความเคลื่อนไหวผ่านกล้องวงจรปิด พร้อมระบบแจ้งเตือนอัตโนมัติ
          ออกแบบมาเพื่อเพิ่มความปลอดภัยให้กับบ้าน สำนักงาน หรืออาคารสถานที่ต่างๆ
        </p>
      </div>

      <div class="about-section">
        <h2>คุณสมบัติหลัก</h2>
        <ul class="feature-list">
          <li>รองรับการเชื่อมต่อกล้องวงจรปิดหลายรูปแบบ (RTSP, RTMP, HLS)</li>
          <li>ระบบตรวจจับความเคลื่อนไหวอัตโนมัติ</li>
          <li>การแจ้งเตือนแบบเรียลไทม์</li>
          <li>การตั้งค่าช่วงเวลาการแจ้งเตือน</li>
          <li>จัดเก็บบันทึกกิจกรรมและการแจ้งเตือน</li>
          <li>หน้า Dashboard สำหรับมอนิเตอร์</li>
        </ul>
      </div>

      <div class="about-section">
        <h2>ตัวตรวจจับที่กำลังทำงานอยู่</h2>
        <p class="section-note">
          ค่าเหล่านี้อ่านมาจาก worker ที่รันตัวตรวจจับจริง ไม่ใช่ค่าที่ตั้งไว้ในหน้าเว็บ —
          เพราะมันถูกอ่านตอนโปรเซสเริ่มทำงาน ที่นี่จึงเป็นที่เดียวที่ตอบได้ว่า
          <strong>ตอนนี้ระบบรันอะไรอยู่จริงๆ</strong>
        </p>

        <p v-if="detectorError" class="detector-warn">{{ detectorError }}</p>
        <p v-else-if="!detector" class="detector-warn">{{ detectorMessage || 'กำลังโหลด...' }}</p>

        <template v-else>
          <div class="detector-grid">
            <div class="detector-item"><span>อุปกรณ์</span><b>{{ detector.device }}</b></div>
            <div class="detector-item"><span>โมเดลท่าทาง</span><b>{{ detector.pose_model }}</b></div>
            <div class="detector-item"><span>ขนาดภาพเข้า</span><b>{{ detector.input_size }}</b></div>
            <div class="detector-item">
              <span>อัตราเฟรม</span>
              <b>{{ detector.target_fps ? detector.target_fps + ' fps' : 'ไม่ได้ตรึงไว้' }}</b>
            </div>
            <div class="detector-item">
              <span>หน้าต่าง</span>
              <b>{{ detector.window_frames }} เฟรม<template v-if="detector.window_seconds"> ({{ detector.window_seconds }} วิ)</template></b>
            </div>
            <div class="detector-item"><span>เกณฑ์แจ้งเตือน</span><b>{{ detector.threshold }}</b></div>
            <div class="detector-item"><span>การกรอง</span><b>{{ detector.smoothing }}</b></div>
            <div class="detector-item"><span>เริ่มคิดเมื่อมีกี่เฟรม</span><b>{{ detector.partial_window_from }}</b></div>
            <div class="detector-item">
              <span>ปรับแสงอัตโนมัติ</span>
              <b>{{ detector.preprocess.join(', ') }}<template v-if="detector.preprocess[0] !== 'off'"> (เมื่อมืดกว่า {{ detector.preprocess_dark_below }})</template></b>
            </div>
            <div class="detector-item"><span>ยกระดับเมื่อยังไม่ลุก</span><b>{{ detector.still_down_seconds }} วินาที</b></div>
          </div>

          <div v-if="detector.measured_known" class="detector-measured ok">
            <strong>ค่าชุดนี้วัดผลไว้แล้ว:</strong> {{ detector.measured }}
          </div>
          <div v-else class="detector-measured warn">
            <strong>ค่าชุดนี้ยังไม่เคยวัดผลแบบครบวงจร</strong> —
            ตัวเลขความแม่นยำที่อ้างอิงกันอยู่ไม่ได้อธิบายการตั้งค่าแบบนี้
            ให้รัน <code>tools/check_config_coherence.py</code> ก่อนนำไปใช้จริง
          </div>

          <p class="section-note">
            รายงานโดย <code>{{ detector.reported_by }}</code> เมื่อ {{ detector.reported_at }}
          </p>
        </template>
      </div>

      <div class="about-section" v-if="isAdmin">
        <h2>เปลี่ยนค่าการตรวจจับ</h2>
        <p class="section-note">
          เลือกได้เฉพาะ<strong>ชุดค่าที่วัดผลไว้แล้วทั้งชุด</strong> ไม่ใช่ปรับทีละค่า —
          เพราะขนาดภาพกับอัตราเฟรมบน CPU เป็นการตัดสินใจเดียวกัน และเกณฑ์แจ้งเตือนก็ไม่อิสระจากทั้งคู่
          ถ้าเลือกผสมกันเองได้ จะได้ระบบที่<strong>ไม่มีใครเคยวัด</strong> แต่หน้าตาน่าเชื่อถือเท่ากัน
        </p>

        <p v-if="profileError" class="detector-warn">{{ profileError }}</p>
        <p v-if="profileSaved" class="profile-saved">{{ profileSaved }}</p>

        <div class="profile-list">
          <label
            v-for="p in profiles"
            :key="p.key"
            class="profile-row"
            :class="{ 'is-running': p.key === runningProfile, 'is-chosen': p.key === chosen }"
          >
            <input type="radio" :value="p.key" v-model="chosen" :disabled="savingProfile" />
            <span class="profile-body">
              <span class="profile-title">
                {{ p.label_th || p.label }}
                <span v-if="p.key === runningProfile" class="profile-now">กำลังใช้อยู่</span>
              </span>
              <span class="profile-measured">{{ p.measured_th || p.measured }}</span>
              <span class="profile-note">{{ p.note_th || p.note }}</span>
            </span>
          </label>
        </div>

        <p class="section-note" v-if="runningProfile === null && detector">
          ตอนนี้ค่าที่รันอยู่<strong>ไม่ตรงกับชุดไหนเลย</strong> — น่าจะมีคนตั้งเองไว้
          ซึ่งแปลว่าไม่มีตัวเลขความแม่นยำชุดไหนอธิบายระบบที่รันอยู่
        </p>

        <button
          class="btn-apply"
          type="button"
          :disabled="!chosen || chosen === runningProfile || savingProfile"
          @click="applyProfile"
        >
          {{ savingProfile ? 'กำลังบันทึก...' : 'บันทึกชุดค่านี้' }}
        </button>
        <p class="section-note">
          บันทึกแล้ว<strong>ยังไม่มีผลทันที</strong> ต้องรีสตาร์ท worker ก่อน —
          ค่าพวกนี้อ่านตอนโปรเซสเริ่มทำงาน ปุ่มนี้จึงบอกตามตรงแทนที่จะแกล้งทำเป็นว่าเปลี่ยนแล้ว
        </p>
      </div>

      <div class="about-section">
        <h2>เวอร์ชั่น</h2>
        <p>V89 Fall Management System เวอร์ชั่น 1.0.0</p>
        <p>พัฒนาโดยทีม V89</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { getApiBaseUrl } from '@/config/api'

// Read-only on purpose. These settings are read when the worker imports the detector, so a
// form that set them would appear to take effect and not have -- and several of them are not
// independent of each other, so a free choice of combinations produces a detector nobody has
// measured. The panel shows what is running and whether that exact combination has a
// measurement behind it; changing it is a deployment action, not a web action.
const detector = ref(null)
const detectorMessage = ref('')
const detectorError = ref('')

// Whole profiles rather than individual settings, and the page says why. A form of knobs would
// let somebody assemble a configuration nobody has measured, and nothing on screen could tell
// them apart from one that had been.
const profiles = ref([])
const runningProfile = ref(null)
const chosen = ref(null)
const savingProfile = ref(false)
const profileError = ref('')
const profileSaved = ref('')
const isAdmin = ref(false)

function authHeaders() {
  return { Authorization: `Bearer ${localStorage.getItem('authToken')}` }
}

async function loadProfiles() {
  try {
    const res = await fetch(`${getApiBaseUrl()}/detector/profiles`, { headers: authHeaders() })
    const body = await res.json()
    if (!res.ok || body.success === false) throw new Error(body.error || 'โหลดไม่สำเร็จ')
    profiles.value = body.data.profiles
    runningProfile.value = body.data.running
    chosen.value = body.data.running
  } catch (e) {
    profileError.value = e.message
  }
}

async function applyProfile() {
  savingProfile.value = true
  profileError.value = ''
  profileSaved.value = ''
  try {
    const res = await fetch(`${getApiBaseUrl()}/detector/profiles`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ profile: chosen.value })
    })
    const body = await res.json()
    if (!res.ok || body.success === false) throw new Error(body.error || 'บันทึกไม่สำเร็จ')
    profileSaved.value = body.data.message
  } catch (e) {
    profileError.value = e.message
  } finally {
    savingProfile.value = false
  }
}

onMounted(async () => {
  try {
    const res = await fetch(`${getApiBaseUrl()}/detector`, {
      headers: { Authorization: `Bearer ${localStorage.getItem('authToken')}` }
    })
    const body = await res.json()
    if (!res.ok || body.success === false) throw new Error(body.error || 'โหลดไม่สำเร็จ')
    detector.value = body.data
    detectorMessage.value = body.message || ''
  } catch (e) {
    detectorError.value = e.message
  }
  try {
    const user = JSON.parse(localStorage.getItem('user') || '{}')
    isAdmin.value = String(user.role || '').toLowerCase().includes('admin')
  } catch {
    isAdmin.value = false
  }
  if (isAdmin.value) await loadProfiles()
})
</script>

<style scoped>
.section-note { font-size: 0.85rem; color: #666; margin: 0.35rem 0 0.75rem; }
.detector-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 0.4rem 1.25rem;
}
.detector-item {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.35rem 0;
  border-bottom: 1px solid #eee;
}
.detector-item span { color: #555; }
.detector-item b { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85rem; }
.detector-measured {
  margin-top: 0.9rem;
  padding: 0.6rem 0.75rem;
  border-radius: 6px;
  font-size: 0.9rem;
  line-height: 1.55;
}
.detector-measured.ok { background: #eef7ee; border-left: 4px solid #2e7d32; }
/* Deliberately loud: an unmeasured combination means no published accuracy figure describes
   what is running, and that is worth interrupting someone over. */
.detector-measured.warn { background: #fdecea; border-left: 4px solid #c62828; }
.detector-warn { color: #c62828; }
.profile-saved {
  background: #eef7ee;
  border-left: 4px solid #2e7d32;
  padding: 0.6rem 0.75rem;
  border-radius: 6px;
  font-size: 0.9rem;
  line-height: 1.55;
}
.profile-list { display: flex; flex-direction: column; gap: 0.5rem; margin: 0.75rem 0; }
.profile-row {
  display: flex;
  gap: 0.7rem;
  align-items: flex-start;
  padding: 0.7rem 0.8rem;
  border: 1px solid #ddd;
  border-radius: 8px;
  cursor: pointer;
}
.profile-row:hover { border-color: #999; }
.profile-row.is-chosen { border-color: #1565c0; background: #f5f9ff; }
/* What is running is stated separately from what is selected: they are different facts and a
   page that merged them would hide an unsaved change. */
.profile-row.is-running { box-shadow: inset 3px 0 0 #2e7d32; }
.profile-body { display: flex; flex-direction: column; gap: 0.15rem; }
.profile-title { font-weight: 600; }
.profile-now {
  margin-left: 0.4rem;
  font-size: 0.72rem;
  font-weight: 700;
  color: #2e7d32;
  border: 1px solid #2e7d32;
  border-radius: 999px;
  padding: 0.05rem 0.4rem;
}
.profile-measured { font-size: 0.85rem; color: #1565c0; font-weight: 600; }
.profile-note { font-size: 0.82rem; color: #555; line-height: 1.5; }
.btn-apply {
  padding: 0.55rem 1.1rem;
  border: none;
  border-radius: 6px;
  background: #1565c0;
  color: #fff;
  font-weight: 600;
  cursor: pointer;
}
.btn-apply:disabled { background: #b0bec5; cursor: default; }

.about {
  padding: 1rem;
}

.about-content {
  max-width: 800px;
  margin: 0 auto;
}

.about-section {
  margin-bottom: 2rem;
}

.about-section h2 {
  color: #1e40af;
  font-size: 1.5rem;
  margin-bottom: 1rem;
  font-weight: 600;
}

.about-section p {
  line-height: 1.6;
  color: #374151;
}

.feature-list {
  list-style-type: disc;
  padding-left: 2rem;
  color: #374151;
}

.feature-list li {
  margin-bottom: 0.5rem;
}

@media (min-width: 1024px) {
  .about {
    min-height: calc(100vh - 200px);
    display: flex;
    align-items: flex-start;
    padding: 2rem;
  }
}
</style>
