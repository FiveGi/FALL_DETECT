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
