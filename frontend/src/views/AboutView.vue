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
        <h2>เวอร์ชั่น</h2>
        <p>V89 Fall Management System เวอร์ชั่น 1.0.0</p>
        <p>พัฒนาโดยทีม V89</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
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
