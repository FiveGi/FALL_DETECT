
<script setup>
import { RouterView, useRoute, RouterLink, useRouter } from 'vue-router'
import NetworkStatus from '@/components/common/NetworkStatus.vue'
import { computed, ref, watch, onMounted, onBeforeUnmount } from 'vue'
import { useAuthStore } from './stores/auth'
import { useCameraStore } from './stores/camera'
import { useNotificationStore } from './stores/notification'
import { useLeftNotificationStore } from './stores/leftNotification'
import LeftSideNotification from '@/components/common/LeftSideNotification.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import logService from '@/services/logService'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()
const cameraStore = useCameraStore()
const notificationStore = useNotificationStore()
const leftNotificationStore = useLeftNotificationStore()

const currentRoutePath = computed(() => route.path)

// Global notification polling
let lastNotificationTimestamp = null
let globalNotificationPollingInterval = null

// ฟังก์ชันสำหรับออกจากระบบและนำทางกลับไปที่หน้า login
async function handleLogout() {
  await authStore.logout()
  router.push('/')
}

// Global notification polling functions
async function fetchGlobalNotifications() {
  try {
    // Only fetch if user is logged in
    if (!authStore.isLoggedIn) return

    const res = await logService.fetchAllNotification()

    // Handle both direct array and wrapped object formats
    let notificationsArray = []
    if (Array.isArray(res)) {
      notificationsArray = res
    } else if (res && res.notifications && Array.isArray(res.notifications)) {
      notificationsArray = res.notifications
    } else if (res && Array.isArray(res.data)) {
      notificationsArray = res.data
    }

    // Sort notifications by timestamp (newest first)
    notificationsArray.sort((a, b) => new Date(b.sent_at) - new Date(a.sent_at))

    // Find new notifications since last fetch
    const newNotifications = []

    if (lastNotificationTimestamp === null) {
      // First fetch - don't show alerts for existing notifications
      if (notificationsArray.length > 0) {
        lastNotificationTimestamp = new Date(notificationsArray[0].sent_at)
      }
    } else {
      for (const notification of notificationsArray) {
        const notificationTime = new Date(notification.sent_at)
        if (notificationTime > lastNotificationTimestamp) {
          newNotifications.push(notification)
        } else {
          break
        }
      }

      if (newNotifications.length > 0) {
        lastNotificationTimestamp = new Date(newNotifications[0].sent_at)

        newNotifications.forEach(notification => {
          showGlobalNotificationAlert(notification)
        })
      }
    }
  } catch (error) {
    console.error('Error fetching global notifications:', error)
  }
}

function showGlobalNotificationAlert(apiNotification) {
  const detectionTypeText = getDetectionTypeText(apiNotification.detection_type)
  // ใช้ cameraStore instance เดียวกัน
  const camera = cameraStore?.cameras?.find(c => c.id === apiNotification.camera_id)
  const cameraName = camera ? camera.name : `กล้อง #${apiNotification.camera_id}`
  const notification = {
    id: `alert_${apiNotification.id}_${Date.now()}`,
    title: '🚨 การแจ้งเตือนความปลอดภัย',
    message: `${cameraName}: ${detectionTypeText}`,
    type: 'warning',
    duration: 10000,
    autoClose: true,
    showTime: true,
  }
  notificationStore.sendNotification(notification)
  leftNotificationStore.addNotification(notification)
}

function getDetectionTypeText(detectionType) {
  switch (detectionType) {
    case 'bed_exit':
      return 'ตรวจจับการลุกจากเตียง'
    case 'fall':
    case 'fall_detection':
      return 'ตรวจจับการล้ม'
    case 'fall_v2':
      return 'ตรวจจับการล้ม (เวอร์ชั่น 2)'
    case 'alone_v2':
      return 'ตรวจจับผู้สูงอายุอยู่คนเดียว'
    default:
      return detectionType || 'การเคลื่อนไหว'
  }
}

// Start global notification polling
function startGlobalNotificationPolling() {
  if (globalNotificationPollingInterval) clearInterval(globalNotificationPollingInterval)

  // Only start if user is logged in
  if (!authStore.isLoggedIn) return

  // Fetch immediately
  fetchGlobalNotifications()

  // Poll every 5 seconds for faster notification detection
  globalNotificationPollingInterval = setInterval(fetchGlobalNotifications, 5000)
}

// Stop global notification polling
function stopGlobalNotificationPolling() {
  if (globalNotificationPollingInterval) {
    clearInterval(globalNotificationPollingInterval)
    globalNotificationPollingInterval = null
  }
}

// เมื่อมีการล็อกอินสำเร็จให้เริ่มระบบแจ้งเตือน
onMounted(() => {
  // เริ่มระบบแจ้งเตือนหลังจาก auth ถูกเริ่มต้นแล้ว
  if (authStore.isInitialized && authStore.isLoggedIn) {
    // Start global notification polling
    startGlobalNotificationPolling()
  }
})

// Clean up polling on unmount
onBeforeUnmount(() => {
  stopGlobalNotificationPolling()
})

// Watch for authentication state changes
watch(
  () => authStore.isLoggedIn,
  (isLoggedIn) => {
    if (isLoggedIn) {
      // Start polling when user logs in
      startGlobalNotificationPolling()
    } else {
      // Stop polling when user logs out
      stopGlobalNotificationPolling()
      // Reset timestamp tracking
      lastNotificationTimestamp = null
    }
  }
)
</script>

<template>
  <div class="app-container">
    <!-- Loading state while auth is initializing -->
    <div v-if="!authStore.isInitialized" class="auth-loading">
      <LoadingSpinner />
      <p>กำลังตรวจสอบสถานะการเข้าสู่ระบบ...</p>
    </div>

    <!-- Main app content after auth is initialized -->

    <template v-else>
      <div>
        <header v-if="authStore.isLoggedIn" class="app-header">
          <div class="app-logo">
            <RouterLink to="/dashboard" class="logo-link">V89 Fall Management System</RouterLink>
          </div>
          <nav class="app-nav">
            <!-- หน้าหลัก/Dashboard มาก่อน -->
            <RouterLink :to="'/dashboard'" :class="{ active: currentRoutePath === '/dashboard' }">
              แดชบอร์ด
            </RouterLink>

            <!-- ฟังก์ชันหลักของระบบ - Monitor -->
            <RouterLink :to="'/monitor'" :class="{ active: currentRoutePath === '/monitor' }">
              มอนิเตอร์
            </RouterLink>

            <!-- ข้อมูลและรายงาน -->
            <RouterLink :to="'/thai-frat-list'" :class="{ active: currentRoutePath === '/thai-frat-list' }">
              ข้อมูล Thai-FRAT
            </RouterLink>

            <!-- เมนูการจัดการสำหรับ Admin (เรียงตามความสำคัญ) -->
            <template v-if="authStore.isAdmin">
              <RouterLink :to="'/camera'" :class="{ active: currentRoutePath === '/camera' }">
                จัดการกล้อง
              </RouterLink>
              <RouterLink :to="'/users'" :class="{ active: currentRoutePath === '/users' }">
                จัดการผู้ใช้
              </RouterLink>
              <RouterLink :to="'/notification-settings'" :class="{ active: currentRoutePath === '/notification-settings' }">
                ตั้งค่าการแจ้งเตือน
              </RouterLink>
            </template>

            <!-- ข้อมูลทั่วไป -->
            <RouterLink :to="'/about'" :class="{ active: currentRoutePath === '/about' }">
              เกี่ยวกับ
            </RouterLink>

            <!-- ข้อมูลผู้ใช้และการออกจากระบบ (ด้านขวาสุด) -->
            <div class="user-section">
              <!-- แสดง role เฉพาะสำหรับ Admin -->
              <span v-if="authStore.isAdmin" class="user-role-badge admin">
                ผู้ดูแลระบบ
              </span>

              <button type="button" @click="handleLogout" class="nav-logout-btn">ออกจากระบบ</button>
            </div>
          </nav>
        </header>

        <!-- การแจ้งเตือนทางด้านซ้าย -->
        <LeftSideNotification v-if="authStore.isLoggedIn" />
        <NetworkStatus /> <!-- เพิ่ม Network Status Monitor -->

        <main class="app-content">
          <RouterView />
        </main>
      </div>
    </template>

    <!-- แสดงการแจ้งเตือน -->
    <div v-if="typeof currentNotification !== 'undefined' && showNotification && currentNotification" class="notification-toast">
      <div class="notification-header">
        <span class="notification-title">{{ currentNotification.title }}</span>
        <button @click="showNotification = false" class="close-btn">&times;</button>
      </div>
      <div class="notification-body">
        {{ currentNotification.message }}
      </div>
      <div class="notification-time">
        {{ currentNotification.timestamp ? new Date(currentNotification.timestamp).toLocaleTimeString() : '' }}
      </div>
    </div>

    <footer class="app-footer">
      <div>
        © {{ new Date().getFullYear() }} V89 Fall Management System - ระบบตรวจจับความเคลื่อนไหวและแจ้งเตือน
      </div>
    </footer>
  </div>
</template>

<style>
/* BackendSwitcher ขนาดเล็กลง */
/* BackendSwitcher ขนาดเล็กสุด */
/* BackendSwitcher ดีไซน์ใหม่ สวยงามและกลมกลืนกับ header */
/* ปรับ BackendSwitcher ให้อยู่ในระนาบเดียวกับโลโก้และเมนู */
.header-flex-row {
  display: flex;
  align-items: center;
  width: 100%;
}
@media (max-width: 768px) {
  .header-flex-row {
    flex-direction: column;
    align-items: stretch;
    gap: 0.5rem;
  }
}
.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  width: 100%;
}

.app-header {
  background-color: #1e40af;
  color: white;
  padding: 1rem 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  width: 100%;
}

.app-logo {
  font-weight: 700;
  font-size: 1.5rem;
}

.logo-link {
  color: white;
  text-decoration: none;
  transition: opacity 0.2s ease;
}

.logo-link:hover {
  opacity: 0.9;
  background-color: transparent; /* ป้องกันไม่ให้ได้รับสไตล์พื้นหลังจาก global styles */
}

.app-nav {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  flex: 1;
  gap: 1.5rem;
  margin-left: auto;
}

.user-section {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding-left: 1.5rem;
  border-left: 1px solid rgba(255, 255, 255, 0.2);
}

.app-nav a {
  color: rgba(255, 255, 255, 0.8);
  text-decoration: none;
  font-weight: 500;
  padding: 0.5rem 0;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
}

.app-nav a:hover,
.app-nav a.active {
  color: white;
  border-bottom-color: white;
}

.nav-logout-btn {
  background-color: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  position: relative;
  z-index: 1;
}

.nav-logout-btn:hover {
  background-color: rgba(255, 255, 255, 0.2);
}

.user-role-badge {
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  border-radius: 12px;
  font-weight: 500;
  white-space: nowrap;
}

.user-role-badge.admin {
  background-color: rgba(251, 191, 36, 0.2);
  color: #fbbf24;
  border: 1px solid rgba(251, 191, 36, 0.3);
}

.user-role-badge.user {
  background-color: rgba(34, 197, 94, 0.2);
  color: #22c55e;
  border: 1px solid rgba(34, 197, 94, 0.3);
}

.app-content {
  flex: 1;
  width: 100%;
  max-width: 1400px; /* เพิ่มความกว้างสูงสุด */
  margin: 0 auto;
  padding: 2rem 1rem;
}

.app-footer {
  background-color: #f3f4f6;
  padding: 1.5rem;
  text-align: center;
  color: #6b7280;
  border-top: 1px solid #e5e7eb;
  font-size: 0.875rem;
}

.notification-toast {
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 350px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  overflow: hidden;
  border-left: 4px solid #2563eb;
  animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
  }
  to {
    transform: translateX(0);
  }
}

.notification-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background-color: #f8fafc;
  border-bottom: 1px solid #e5e7eb;
}

.notification-title {
  font-weight: 600;
  color: #1e40af;
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
  color: #6b7280;
}

.notification-body {
  padding: 16px;
  color: #374151;
}

.notification-time {
  padding: 8px 16px;
  font-size: 0.75rem;
  color: #6b7280;
  text-align: right;
  background-color: #f8fafc;
  border-top: 1px solid #e5e7eb;
}

.auth-loading {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  background-color: #f8fafc;
}

.auth-loading p {
  margin-top: 1rem;
  color: #6b7280;
  font-size: 0.875rem;
}

@media (max-width: 768px) {
  .app-header {
    flex-direction: column;
    gap: 1rem;
    padding: 1rem;
  }

  .app-nav {
    width: 100%;
    justify-content: center;
    flex-wrap: wrap;
    gap: 1rem;
  }

  .app-content {
    padding: 1rem 0.5rem;
  }

  .notification-toast {
    width: calc(100% - 40px);
    bottom: 10px;
    right: 10px;
    left: 10px;
  }
}
</style>
