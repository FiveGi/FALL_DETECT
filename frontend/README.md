# V89 Elderly Surveillance

ระบบตรวจจับความเคลื่อนไหวและแจ้งเตือนอัจฉริยะ

## การติดตั้ง

### การติดตั้งแบบปกติ

```bash
# ติดตั้ง dependencies
npm install

# รัน development server
npm run dev

# build สำหรับ production
npm run build
```

### การติดตั้งด้วย Docker

```bash
# build Docker image
docker build -t frontend .

# รัน container (แบบเชื่อมต่อกับ backend network)
docker run -d -p 3000:3000 --network backend-elderly-surveillance_default --name frontend-app frontend

# หรือรันแบบ standalone
docker run -d -p 3000:3000 --name frontend-app frontend
```

## การตั้งค่า Firebase

1. สร้างโปรเจค Firebase ที่ [Firebase Console](https://console.firebase.google.com/)
2. เปิดใช้งาน Authentication และเพิ่ม Google เป็น Sign-in method
3. คัดลอกค่า configuration จาก Project settings และนำมาใส่ในไฟล์ `.env`

## เทคโนโลยีที่ใช้
- Vue 3
- Vite
- Firebase Authentication
- Pinia
- Vue Router

VITE_FIREBASE_STORAGE_BUCKET=your_project_id.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=your_sender_id
VITE_FIREBASE_APP_ID=your_app_id
```

### 3. รันโปรเจ็ค:
```bash
npm run dev
```

## การแก้ไขปัญหา

### ถ้าเจอปัญหา PowerShell execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### ถ้าเจอปัญหา dependency:
```bash
# ลบทุกอย่างและติดตั้งใหม่
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

## คุณสมบัติ

- ✅ การเข้าสู่ระบบแบบปกติ (username/password)
- ✅ การเข้าสู่ระบบด้วย Google (Firebase)
- ✅ จัดการกล้องวงจรปิด
- ✅ ระบบมอนิเตอร์แบบเรียลไทม์
- ✅ การแจ้งเตือนอัตโนมัติ
- ✅ ระบบการตั้งค่าการแจ้งเตือน
- ✅ รองรับ AI Detection หลายรูปแบบ:
  - `bed_exit`: ตรวจจับการลุกจากเตียง
  - `fall`: ตรวจจับการล้ม (Enhanced Detection)
  - `fall_v2`: ตรวจจับการล้ม (เวอร์ชั่น 2)

## การใช้งาน

### ข้อมูลสำหรับทดสอบ
- **Username**: admin
- **Password**: admin123

หรือใช้การเข้าสู่ระบบด้วย Google

## เทคโนโลยีที่ใช้

- Vue 3 + Composition API
- Pinia (State Management)
- Vue Router
- Firebase Authentication
- Vite

## โครงสร้างโปรเจ็ค

```
src/
├── components/       # Vue components
├── views/           # หน้าต่างๆ ของแอป
├── stores/          # Pinia stores
├── services/        # API services
├── config/          # ไฟล์ config
└── router/          # Vue Router config
```

## การตั้งค่าสำหรับการพัฒนา

โปรเจคนี้ใช้ Vite, Vue 3 และ Pinia

### คำสั่งที่ใช้บ่อย

```bash
# เริ่มเซิร์ฟเวอร์สำหรับพัฒนา
npm run dev

# สร้างไฟล์สำหรับ production
npm run build

# ตรวจสอบโค้ดด้วย ESLint
npm run lint

# จัดรูปแบบโค้ดด้วย Prettier
npm run format
```

### คำสั่ง Docker

```bash
# build image ใหม่
docker build -t frontend .

# รัน container
docker run -d -p 3000:3000 --network backend-elderly-surveillance_default --name frontend-app frontend

# ดู logs
docker logs frontend-app

# หยุด container
docker stop frontend-app

# ลบ container
docker rm frontend-app

# เข้าใน container เพื่อ debug
docker exec -it frontend-app sh
```

## สถาปัตยกรรม

Frontend: Vue 3 + Pinia + Vue Router
Backend: (จะเพิ่มในอนาคต)

## การเชื่อมต่อกับ Backend

ระบบถูกออกแบบให้เชื่อมต่อกับ Backend API ที่กำหนด endpoint ตามที่ระบุใน `src/config/api.js`
ในขณะนี้ระบบใช้ localStorage เป็น fallback สำหรับการทดสอบเมื่อไม่มี Backend

## ลิขสิทธิ์

© 2023 V89 Elderly Surveillance - สงวนลิขสิทธิ์
