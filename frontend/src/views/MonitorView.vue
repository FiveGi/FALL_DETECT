<script setup>
import { ref, onMounted, onBeforeUnmount, computed, nextTick, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useCameraStore } from '@/stores/camera'
import { useNotificationStore } from '@/stores/notification'
import { useAuthStore } from '@/stores/auth'
import IconAlert from '@/components/icons/IconAlert.vue'
import IconCamera from '@/components/icons/IconCamera.vue'
import MediaViewer from '@/components/common/MediaViewer.vue'
import logService from '@/services/logService'
import cameraService from '@/services/cameraService'
import streamService from '@/services/streamService'
import adminService from '@/services/adminService'
import { getDetectionTypeText, DETECTION_TYPE_OPTIONS, DETECTION_TYPE_FORM_HELP } from '@/utils/detectionType'

const cameraStore = useCameraStore()
const notificationStore = useNotificationStore()
const authStore = useAuthStore()
const router = useRouter()

// ตรวจสอบสิทธิ์การเข้าถึง
const isAdmin = computed(() => authStore.isAdmin)
const currentUser = computed(() => authStore.user)

// กรองกล้องตาม role และ owner filter
const availableCameras = computed(() => {
  let cameras = []

  if (isAdmin.value) {
    // Admin เห็นกล้องทั้งหมด
    cameras = cameraStore.cameras

    // กรองตามเจ้าของถ้ามีการเลือก
    if (selectedOwnerFilter.value !== null) {
      cameras = cameras.filter(camera =>
        camera.owner?.id === selectedOwnerFilter.value ||
        camera.owner_id === selectedOwnerFilter.value
      )
    // ← เพิ่ม: กรอง camera ที่ข้อมูลไม่ครบออก
    return cameras.filter(camera =>
      camera &&
      camera.id !== undefined &&
      camera.id !== null &&
      camera.url !== undefined &&
      camera.url !== null &&
      camera.url !== ''
    )
  }
  } else {
    // User เห็นเฉพาะกล้องของตัวเอง
    cameras = cameraStore.cameras.filter(camera =>
      camera.owner?.id === currentUser.value?.id ||
      camera.owner_id === currentUser.value?.id
    )
  }

  return cameras
})

// ตรวจสอบว่ามีกล้องจริงๆ ในกรณีกรองตามเจ้าของ
const hasFilteredCameras = computed(() => availableCameras.value.length > 0)
const showNoFilteredCamerasMessage = computed(() => {
  return isAdmin.value && selectedOwnerFilter.value !== null && !hasFilteredCameras.value
})

// สำหรับเก็บกล้องทั้งหมดไม่กรอง (สำหรับ admin)
const allCameras = computed(() => cameraStore.cameras)

// รายการผู้ใช้ที่มีกล้อง (สำหรับแสดงใน filter)
const usersForFilter = computed(() => {
  if (!isAdmin.value) return []

  const userMap = new Map()

  // สร้างรายการผู้ใช้จากกล้องที่มี
  allCameras.value.forEach(camera => {
    if (camera.owner) {
      userMap.set(camera.owner.id, camera.owner)
    }
  })

  // เพิ่มผู้ใช้จาก allUsers ที่อาจไม่มีกล้อง
  allUsers.value.forEach(user => {
    userMap.set(user.id, user)
  })

  return Array.from(userMap.values()).sort((a, b) => a.username.localeCompare(b.username))
})

// State
const activeMonitors = ref({})
const motionDetected = ref({})
const riskLevel = ref({})
const cameraStatuses = ref({}) // เก็บสถานะรายละเอียดของกล้องจาก backend
const operationInProgress = ref({}) // เก็บสถานะการดำเนินการ start/stop
const isBlurAllActive = ref(false) // สำหรับการเบลอทั้งหมด
const logs = ref([])
const notifications = ref([])
const combinedActivities = ref([]) // รวม logs + notifications สำหรับแสดงในกิจกรรม
const fullscreenCamera = ref(null) // เพิ่มตัวแปรสำหรับเก็บกล้องที่กำลังแสดงแบบเต็มจอ
const previousFrames = ref({}) // เก็บ previous frame ของแต่ละกล้อง
const isSavingSettings = ref(false) // สถานะการบันทึกการตั้งค่า
const selectedCameraFilter = ref(null) // เก็บ ID ของกล้องที่เลือกสำหรับกรองการแจ้งเตือน
const selectedOwnerFilter = ref(null) // เก็บ ID ของผู้ใช้ที่เลือกสำหรับกรองกล้อง
const allUsers = ref([]) // เก็บรายการผู้ใช้ทั้งหมดสำหรับ admin

// เพิ่ม state สำหรับการแก้ไขกล้อง
const isEditing = ref(false)
const editingCamera = ref({
  id: '',
  name: '',
  room_name: '',
  url: '',
  detection_type: 'bed_exit',
  alert_start_time: '21:00',
  alert_end_time: '05:00',
  notification_cooldown_sec: 600,
  ai_confidence_threshold: 0.5,
})
const showEditModal = ref(false)
const message = ref('')
const messageType = ref('')
const testVideos = ref([]) // รายชื่อไฟล์วิดีโอทดสอบในโฟลเดอร์ Test/ ของ backend
const editingCameraSourceType = ref('url')

// Time filter states
const timeFilterEnabled = ref(false)
const timeFilterStart = ref('')
const timeFilterEnd = ref('')

// Notification settings
const notificationPeriod = ref({
  start: '00:00',
  end: '23:59',
})

// Global detection type setting
const globalDetectionType = ref('bed_exit')

// Timer references
const monitoringTimers = ref({})
const motionSimulationTimers = ref({})
const riskResetTimers = ref({})

// Helper function to check if URL is RTSP or streaming URL
// แก้ไขให้ทุก camera ใช้ Backend streaming เสมอ
function isRtspOrStreamUrl(url) {
  // ให้ทุก camera ใช้ Backend streaming API
  return true
}

// Helper function to check if URL is a video file
function isVideoFileUrl(url) {
  if (!url) return false
  const videoExtensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v']
  const urlLower = url.toLowerCase()
  return videoExtensions.some(ext => urlLower.includes(ext))
}

function toggleBlurAll() {
  isBlurAllActive.value = !isBlurAllActive.value

  localStorage.setItem('blurAllState', JSON.stringify(isBlurAllActive.value))
}

function saveMonitorState() {
  const state = {
    selectedOwnerFilter: selectedOwnerFilter.value,
    selectedCameraFilter: selectedCameraFilter.value,
    timeFilterEnabled: timeFilterEnabled.value,
    timeFilterStart: timeFilterStart.value,
    timeFilterEnd: timeFilterEnd.value,
    fullscreenCamera: fullscreenCamera.value
  }
  localStorage.setItem('monitorViewState', JSON.stringify(state))
}

// Helper function to load monitor view state
function loadMonitorState() {
  const savedState = localStorage.getItem('monitorViewState')
  if (savedState) {
    try {
      const state = JSON.parse(savedState)
      selectedOwnerFilter.value = state.selectedOwnerFilter || null
      selectedCameraFilter.value = state.selectedCameraFilter || null
      timeFilterEnabled.value = state.timeFilterEnabled || false
      timeFilterStart.value = state.timeFilterStart || ''
      timeFilterEnd.value = state.timeFilterEnd || ''
      // ไม่เก็บ fullscreen state เพราะไม่เหมาะสมเมื่อกลับมาหน้าใหม่
    } catch (e) {
      console.error('Error loading monitor state:', e)
    }
  }
}// Computed properties
const cameras = computed(() => availableCameras.value)
const hasCameras = computed(() => cameras.value.length > 0)
const eventCount = computed(() => filteredActivities.value.length)
const alertCount = computed(() => notifications.value.length)

// กรองกิจกรรมตามกล้องที่เลือก เวลา และไม่แสดงการตั้งค่า
const filteredActivities = computed(() => {
  let filtered = combinedActivities.value

  // กรองออกกิจกรรมที่เกี่ยวกับการตั้งค่า (settings)
  filtered = filtered.filter(activity => {
    // ถ้าเป็น log และมี category = 'settings' ให้ filter ออก
    if (activity.activityType === 'log' && activity.category === 'settings') {
      return false
    }
    return true
  })

  // กรองตามกล้องที่เลือก
  if (selectedCameraFilter.value) {
    filtered = filtered.filter(activity => {
      // กรองทั้ง logs และ notifications ที่มา camera_id ตรงกับที่เลือก
      if (activity.camera_id) {
        return activity.camera_id === selectedCameraFilter.value
      }
      // สำหรับ local logs ที่ไม่มี camera_id ให้แสดงทั้งหมด
      return activity.isLocal
    })
  }

  // กรองตามเวลา
  if (timeFilterEnabled.value && timeFilterStart.value && timeFilterEnd.value) {
    filtered = filtered.filter(activity => {
      try {
        let activityTime

        // Handle different timestamp formats
        if (activity.isLocal) {
          // Local logs have time format like "14:30:25"
          const today = new Date().toDateString()
          activityTime = new Date(`${today} ${activity.timestamp}`)
        } else {
          // Database logs/notifications have formatted timestamps
          if (activity.timestamp.includes('/')) {
            // Format: "23/07/2025 14:55:58"
            const [datePart, timePart] = activity.timestamp.split(' ')
            const [day, month, year] = datePart.split('/')
            activityTime = new Date(`${year}-${month}-${day}T${timePart}`)
          } else {
            // ISO format or other formats
            activityTime = new Date(activity.timestamp)
          }
        }

        if (isNaN(activityTime.getTime())) {
          return true // If can't parse time, include the activity
        }

        const startTime = new Date(`${new Date().toDateString()} ${timeFilterStart.value}`)
        const endTime = new Date(`${new Date().toDateString()} ${timeFilterEnd.value}`)

        // Handle case where end time is on the next day (e.g., start: 22:00, end: 06:00)
        if (endTime < startTime) {
          endTime.setDate(endTime.getDate() + 1)
        }

        const activityTimeOfDay = new Date(`${new Date().toDateString()} ${activityTime.toTimeString().split(' ')[0]}`)

        if (endTime.getDate() > startTime.getDate()) {
          // Spans midnight
          return activityTimeOfDay >= startTime || activityTimeOfDay <= new Date(`${new Date().toDateString()} ${timeFilterEnd.value}`)
        } else {
          // Same day
          return activityTimeOfDay >= startTime && activityTimeOfDay <= endTime
        }
      } catch (error) {
        console.error('Error filtering by time:', error, activity)
        return true // If error, include the activity
      }
    })
  }

  // Sort all activities by timestamp (most recent first)
  return filtered.sort((a, b) => {
    try {
      let timeA, timeB

      // Parse timestamp for activity A
      if (a.isLocal) {
        const today = new Date().toDateString()
        timeA = new Date(`${today} ${a.timestamp}`)
      } else {
        if (a.timestamp.includes('/')) {
          // Format: "23/07/2025 14:55:58"
          const [datePart, timePart] = a.timestamp.split(' ')
          const [day, month, year] = datePart.split('/')
          timeA = new Date(`${year}-${month}-${day}T${timePart}`)
        } else {
          timeA = new Date(a.timestamp)
        }
      }

      // Parse timestamp for activity B
      if (b.isLocal) {
        const today = new Date().toDateString()
        timeB = new Date(`${today} ${b.timestamp}`)
      } else {
        if (b.timestamp.includes('/')) {
          // Format: "23/07/2025 14:55:58"
          const [datePart, timePart] = b.timestamp.split(' ')
          const [day, month, year] = datePart.split('/')
          timeB = new Date(`${year}-${month}-${day}T${timePart}`)
        } else {
          timeB = new Date(b.timestamp)
        }
      }

      // If parsing failed, fallback to original timestamp comparison
      if (isNaN(timeA.getTime()) || isNaN(timeB.getTime())) {
        return b.timestamp.localeCompare(a.timestamp)
      }

      // Sort by time (most recent first)
      return timeB.getTime() - timeA.getTime()
    } catch (error) {
      console.error('Error sorting activities:', error, a, b)
      return 0
    }
  })
})

// Watch for state changes and save to localStorage
watch([selectedOwnerFilter, selectedCameraFilter, timeFilterEnabled, timeFilterStart, timeFilterEnd],
  () => {
    saveMonitorState()
  },
  { deep: true }
)

onMounted(async () => {
  // Load cameras เฉพาะเมื่อ user ล็อกอินแล้ว
  if (authStore.isLoggedIn) {
    cameraStore.loadCameras()
  }

  // โหลดรายการผู้ใช้สำหรับ admin
  if (isAdmin.value) {
    try {
      const response = await adminService.getUsers()
      allUsers.value = response.users || []
    } catch (error) {
      console.error('Failed to load users:', error)
    }
  }

  // Load saved notification settings if any
  const savedSettings = localStorage.getItem('notificationSettings')
  if (savedSettings) {
    try {
      const parsed = JSON.parse(savedSettings)

      // Handle both old and new format
      if (parsed.start && parsed.end) {
        // Old format: { start: "21:00", end: "05:00" }
        notificationPeriod.value = parsed
      } else if (parsed.timeRange) {
        // New format from NotificationSettingsView: { timeRange: { start: "21:00", end: "05:00" }, ... }
        notificationPeriod.value = {
          start: parsed.timeRange.start,
          end: parsed.timeRange.end
        }
      }
    } catch (e) {
      console.error('Error parsing notification settings:', e)
    }
  }

  // โหลดสถานะ blur all จาก localStorage
  const savedBlurState = localStorage.getItem('blurAllState')
  if (savedBlurState) {
    try {
      isBlurAllActive.value = JSON.parse(savedBlurState)
    } catch (e) {
      console.error('Error parsing blur state:', e)
    }
  }

  // โหลดสถานะการใช้งานจาก localStorage
  loadMonitorState()

  // Initialize state for each camera
  cameras.value.forEach(async (camera) => {
    motionDetected.value[camera.id] = false
    riskLevel.value[camera.id] = 'normal'
    activeMonitors.value[camera.id] = false
    operationInProgress.value[camera.id] = false

    // เริ่ม stream เฉพาะสำหรับ RTSP/HTTP streams เท่านั้น (ไม่ใช่ไฟล์วิดีโอ)
    // ใช้ Backend streaming สำหรับทุกประเภท camera
    try {
      await streamService.startCameraStream(camera.id)
    } catch (error) {
      console.error(`Failed to start stream for camera ${camera.name}:`, error)
    }
  })

  await checkAllCameraStatuses()

  if (cameras.value.length > 0) {
    const detectionTypes = cameras.value.map(camera => camera.detection_type || 'bed_exit')
    const bedExitCount = detectionTypes.filter(type => type === 'bed_exit').length
    const fallDetectionCount = detectionTypes.filter(type => type === 'fall').length
    const fallV2Count = detectionTypes.filter(type => type === 'fall_v2').length

    // Set to the most common type, priority: fall_v2 > fall > bed_exit
    if (fallV2Count > 0 && fallV2Count >= fallDetectionCount && fallV2Count >= bedExitCount) {
      globalDetectionType.value = 'fall_v2'
    } else if (fallDetectionCount > bedExitCount) {
      globalDetectionType.value = 'fall'
    } else {
      globalDetectionType.value = 'bed_exit'
    }
  }

  // Start polling all logs and notifications
  startAllDataPolling()

  // Add keyboard event listener for ESC key to exit fullscreen
  const handleKeydown = (event) => {
    if (event.key === 'Escape' && fullscreenCamera.value) {
      fullscreenCamera.value = null
    }
  }
  document.addEventListener('keydown', handleKeydown)

  // Store the handler reference for cleanup
  window.fullscreenKeyHandler = handleKeydown
})

onBeforeUnmount(() => {
  // หยุด Backend streams สำหรับทุก camera
  cameras.value.forEach(async (camera) => {
    try {
      await streamService.stopCameraStream(camera.id)
    } catch (error) {
      console.error(`Failed to stop stream for camera ${camera.name}:`, error)
    }
  })

  Object.keys(monitoringTimers.value).forEach((cameraId) => {
    clearTimeout(monitoringTimers.value[cameraId])
  })

  Object.keys(motionSimulationTimers.value).forEach((cameraId) => {
    clearTimeout(motionSimulationTimers.value[cameraId])
  })

  // Clean up keyboard event listener
  if (window.fullscreenKeyHandler) {
    document.removeEventListener('keydown', window.fullscreenKeyHandler)
    delete window.fullscreenKeyHandler
  }

  stopLogPolling()
})

// Check camera status from backend
async function checkCameraStatus(camera) {
  try {
    const status = await cameraService.getCameraStatus(camera.id)

    // Update the activeMonitors state based on backend status
    activeMonitors.value[camera.id] = status.is_active && status.status === 'running'

    // Store detailed status information
    cameraStatuses.value[camera.id] = status

    return status
  } catch (error) {
    console.error(`Error checking camera ${camera.id} status:`, error)
    // If we can't get status, assume it's not active
    activeMonitors.value[camera.id] = false
    cameraStatuses.value[camera.id] = null
    return null
  }
}

// Check all camera statuses
async function checkAllCameraStatuses() {
  if (!cameras.value.length) return

  try {
    const statusPromises = cameras.value.map(camera => checkCameraStatus(camera))
    await Promise.all(statusPromises)
  } catch (error) {
    console.error('Error checking camera statuses:', error)
  }
}

async function startMonitoring(camera) {
  if (operationInProgress.value[camera.id]) return

  operationInProgress.value[camera.id] = true

  try {
    // เรียก backend API เพื่อเริ่ม monitoring
    const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/cameras/${camera.id}/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('authToken')}`
      },
    })

    if (response.ok) {
      // รอสักครู่เพื่อให้ backend เริ่มทำงาน แล้วตรวจสอบสถานะจริง
      setTimeout(async () => {
        await checkCameraStatus(camera)
      }, 1000)

      notificationStore.sendNotification({
        title: 'เริ่มการตรวจจับ',
        message: `เริ่มการตรวจจับบนกล้อง ${camera.name} เรียบร้อยแล้ว`,
        type: 'success'
      })
    } else {
      throw new Error('Failed to start monitoring')
    }
  } catch (error) {
    console.error('Error starting monitoring:', error)
    notificationStore.sendNotification({
      title: 'เกิดข้อผิดพลาด',
      message: `ไม่สามารถเริ่มการตรวจจับบนกล้อง ${camera.name} ได้`,
      type: 'error'
    })
  } finally {
    operationInProgress.value[camera.id] = false
  }
}

// Stop monitoring for a specific camera
async function stopMonitoring(camera) {
  if (operationInProgress.value[camera.id]) return

  operationInProgress.value[camera.id] = true

  try {
    // เรียก backend API เพื่อหยุด monitoring
    const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/cameras/${camera.id}/stop`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('authToken')}`
      },
    })

    if (response.ok) {
      // รอสักครู่เพื่อให้ backend หยุดทำงาน แล้วตรวจสอบสถานะจริง
      setTimeout(async () => {
        await checkCameraStatus(camera)
      }, 1000)

      // Clear frontend state
      motionDetected.value[camera.id] = false
      riskLevel.value[camera.id] = 'normal'

      // Clear timers
      if (monitoringTimers.value[camera.id]) {
        clearTimeout(monitoringTimers.value[camera.id])
        delete monitoringTimers.value[camera.id]
      }

      if (motionSimulationTimers.value[camera.id]) {
        clearTimeout(motionSimulationTimers.value[camera.id])
        delete motionSimulationTimers.value[camera.id]
      }

      if (riskResetTimers.value[camera.id]) {
        clearTimeout(riskResetTimers.value[camera.id])
        delete riskResetTimers.value[camera.id]
      }

      notificationStore.sendNotification({
        title: 'หยุดการตรวจจับ',
        message: `หยุดการตรวจจับบนกล้อง ${camera.name} เรียบร้อยแล้ว`,
        type: 'success'
      })
    } else {
      throw new Error('Failed to stop monitoring')
    }
  } catch (error) {
    console.error('Error stopping monitoring:', error)
    notificationStore.sendNotification({
      title: 'เกิดข้อผิดพลาด',
      message: `ไม่สามารถหยุดการตรวจจับบนกล้อง ${camera.name} ได้`,
      type: 'error'
    })
  } finally {
    operationInProgress.value[camera.id] = false
  }
}

// Trigger an alert notification
function triggerAlert(notification) {
  const riskLevel = getRiskLevelFromDetectionType(notification.detection_type)
  setCameraRiskLevel(notification.camera_id, riskLevel, 15)
  notificationStore.sendNotification({
    title: 'การแจ้งเตือนความปลอดภัย',
    message: `ตรวจพบการเคลื่อนไหวที่น่าสงสัย - ${notification.detection_type}`,
    type: 'warning',
    timestamp: formatTimestamp(notification.sent_at),
  })
}

// Get camera name by ID
function getCameraName(cameraId) {
  const camera = cameras.value.find(c => c.id === cameraId)
  return camera ? camera.name : `กล้อง #${cameraId}`
}

// Handle media loading events from MediaViewer
function handleMediaLoad(camera, event) {
  camera.status = 'online'
}

function handleMediaError(camera, event) {
  console.error(`Error loading ${event.mediaType} for camera ${camera.name}:`, event.url)
  camera.status = 'offline'
}

// Translate detection result to user-friendly Thai text
function getDetectionResultText(detectionResult) {
  switch (detectionResult?.toLowerCase()) {
    case 'bed':
      return 'เตียงว่าง (ไม่มีคนบนเตียง)'
    case 'sleep':
      return 'กำลังนอน'
    case 'sit':
      return 'กำลังนั่ง'
    case 'alone':
      return 'อยู่คนเดียว'
    case 'alone_v2':
      return 'อยู่คนเดียว (เวอร์ชั่น 2)'
    case 'fall':
      return 'ล้ม'
    case 'fall_v2':
      return 'ล้ม (เวอร์ชั่น 3 - YOLO-pose)'
    case 'no_fall':
      return 'ไม่ตรวจพบการล้ม'
    case 'no_detection':
      return 'ไม่ตรวจพบการล้ม'
    default:
      return 'ไม่ตรวจพบการล้ม'
  }
}

// Map backend detection types to risk level
function getRiskLevelFromDetectionType(detectionType) {
  const dt = (detectionType || '').toString().trim().toLowerCase()
  const redTypes = ['fall', 'fall_red', 'fall_v2']
  const yellowTypes = ['bed_exit', 'alone', 'alone_yellow']
  if (redTypes.includes(dt)) return 'red'
  if (yellowTypes.includes(dt)) return 'yellow'
  return 'normal'
}

function setCameraRiskLevel(cameraId, level = 'normal', durationSeconds = 15) {
  if (!cameraId) return
  const normalized = (level || 'normal').toString().toLowerCase()
  riskLevel.value[cameraId] = normalized
  if (riskResetTimers.value[cameraId]) {
    clearTimeout(riskResetTimers.value[cameraId])
    delete riskResetTimers.value[cameraId]
  }
  if (normalized !== 'normal' && durationSeconds > 0) {
    riskResetTimers.value[cameraId] = setTimeout(() => {
      riskLevel.value[cameraId] = 'normal'
      delete riskResetTimers.value[cameraId]
    }, durationSeconds * 1000)
  }
}

// Get risk level text and color class
function getRiskLevelText(riskLevel) {
  switch (riskLevel?.toLowerCase()) {
    case 'yellow':
      return 'ระวัง'
    case 'red':
      return 'อันตราย'
    default:
      return 'ปกติ'
  }
}

function getRiskLevelClass(riskLevel) {
  switch (riskLevel?.toLowerCase()) {
    case 'yellow':
      return 'risk-yellow'
    case 'red':
      return 'risk-red'
    default:
      return 'risk-green'
  }
}

// Format person count text
function getPersonCountText(personCount) {
  if (personCount === undefined || personCount === null) return ''
  if (personCount === 0) return '(ไม่มีคน)'
  if (personCount === 1) return '(1 คน)'
  return `(${personCount} คน)`
}

// Get status text for camera status
function getStatusText(status) {
  switch (status?.toLowerCase()) {
    case 'running':
      return 'กำลังทำงาน'
    case 'stopped':
      return 'หยุดการทำงาน'
    case 'error':
      return 'เกิดข้อผิดพลาด'
    default:
      return 'ไม่ทราบสถานะ'
  }
}

// Get status CSS class for camera status
function getStatusClass(status) {
  switch (status?.toLowerCase()) {
    case 'running':
      return 'status-running'
    case 'stopped':
      return 'status-stopped'
    case 'error':
      return 'status-error'
    default:
      return 'status-unknown'
  }
}

// Add a local log entry (for UI events) - can be filtered by category
function addLocalLog(type, message, cameraId = null, category = 'system') {
  const newLog = {
    id: `local_${Date.now()}`,
    type,
    message,
    timestamp: new Date().toLocaleTimeString(),
    isLocal: true,
    activityType: 'log',
    camera_id: cameraId,
    category: category // 'system', 'settings', 'detection', 'camera'
  }

  // Add to combined activities (sorting will be handled by filteredActivities computed property)
  combinedActivities.value.push(newLog)

  // Keep only the last 100 activities to allow for better chronological mixing
  if (combinedActivities.value.length > 100) {
    combinedActivities.value.splice(0, combinedActivities.value.length - 100)
  }
}

// Save notification settings to localStorage and update all cameras
async function saveNotificationSettings() {
  if (isSavingSettings.value) return // Prevent multiple simultaneous saves

  isSavingSettings.value = true

  try {
    // Save to localStorage - try to preserve extended format if it exists
    const existingSettings = localStorage.getItem('notificationSettings')
    let settingsToSave

    try {
      const parsed = JSON.parse(existingSettings)
      if (parsed.timeRange) {
        // Extended format exists, update it
        parsed.timeRange.start = notificationPeriod.value.start
        parsed.timeRange.end = notificationPeriod.value.end
        settingsToSave = parsed
      } else {
        // Simple format, keep it simple
        settingsToSave = {
          start: notificationPeriod.value.start,
          end: notificationPeriod.value.end
        }
      }
    } catch (e) {
      // If parsing fails or no existing settings, use simple format
      settingsToSave = {
        start: notificationPeriod.value.start,
        end: notificationPeriod.value.end
      }
    }

    localStorage.setItem('notificationSettings', JSON.stringify(settingsToSave))

    // Update all cameras with new alert time settings
    const updatePromises = cameras.value.map(async (camera) => {
      try {
        const updatedData = {
          name: camera.name,
          url: camera.url,
          room_name: camera.room_name,
          detection_type: camera.detection_type,
          alert_start_time: notificationPeriod.value.start,
          alert_end_time: notificationPeriod.value.end,
          notification_cooldown: camera.notification_cooldown,
          ai_confidence_threshold: camera.ai_confidence_threshold
        }

        const result = await cameraService.updateCamera(camera.id, updatedData)
        return { success: true, camera: camera.name }
      } catch (error) {
        console.error(`Failed to update camera ${camera.name}:`, error)
        return { success: false, camera: camera.name, error }
      }
    })

    // Wait for all updates to complete
    const results = await Promise.all(updatePromises)
    const successCount = results.filter(r => r.success).length
    const failedCount = results.filter(r => !r.success).length

    if (failedCount === 0) {
      // Show success notification
      notificationStore.sendNotification({
        title: 'บันทึกการตั้งค่าสำเร็จ',
        message: `อัปเดตเวลาการแจ้งเตือนสำหรับกล้องทั้งหมด ${successCount} ตัวแล้ว`,
        type: 'success'
      })
    } else {

      // Show warning notification
      notificationStore.sendNotification({
        title: 'บันทึกการตั้งค่าบางส่วน',
        message: `อัปเดตสำเร็จ ${successCount} ตัว, ล้มเหลว ${failedCount} ตัว`,
        type: 'warning'
      })
    }

    // Reload cameras to get updated data
    await cameraStore.loadCameras()

  } catch (error) {
    console.error('Error saving notification settings:', error)

    notificationStore.sendNotification({
      title: 'เกิดข้อผิดพลาด',
      message: 'ไม่สามารถบันทึกการตั้งค่าการแจ้งเตือนได้',
      type: 'error'
    })
  } finally {
    isSavingSettings.value = false
  }
}

// Save global detection type to all cameras
async function saveGlobalDetectionType() {
  if (isSavingSettings.value) return // Prevent multiple simultaneous saves

  isSavingSettings.value = true

  try {
    // Update all cameras with new detection type
    const updatePromises = cameras.value.map(async (camera) => {
      try {
        const updatedData = {
          name: camera.name,
          url: camera.url,
          room_name: camera.room_name,
          detection_type: globalDetectionType.value,
          alert_start_time: camera.alert_start_time,
          alert_end_time: camera.alert_end_time,
          notification_cooldown: camera.notification_cooldown,
          ai_confidence_threshold: camera.ai_confidence_threshold
        }

        const result = await cameraService.updateCamera(camera.id, updatedData)
        return { success: true, camera: camera.name }
      } catch (error) {
        console.error(`Failed to update camera ${camera.name}:`, error)
        return { success: false, camera: camera.name, error }
      }
    })

    // Wait for all updates to complete
    const results = await Promise.all(updatePromises)
    const successCount = results.filter(r => r.success).length
    const failedCount = results.filter(r => !r.success).length

    const detectionTypeText = getDetectionTypeText(globalDetectionType.value)

    if (failedCount === 0) {
      // Show success notification
      notificationStore.sendNotification({
        title: 'อัปเดตการตรวจจับสำเร็จ',
        message: `อัปเดตประเภทการตรวจจับเป็น ${detectionTypeText} สำหรับกล้องทั้งหมด ${successCount} ตัวแล้ว`,
        type: 'success'
      })
    } else {
      // Show warning notification
      notificationStore.sendNotification({
        title: 'อัปเดตการตรวจจับบางส่วน',
        message: `อัปเดตสำเร็จ ${successCount} ตัว, ล้มเหลว ${failedCount} ตัว`,
        type: 'warning'
      })
    }

    // Reload cameras to get updated data
    await cameraStore.loadCameras()

  } catch (error) {
    console.error('Error saving global detection type:', error)

    notificationStore.sendNotification({
      title: 'เกิดข้อผิดพลาด',
      message: 'ไม่สามารถอัปเดตประเภทการตรวจจับได้',
      type: 'error'
    })
  } finally {
    isSavingSettings.value = false
  }
}

// Go to camera management
function goToAddCamera() {
  router.push('/camera')
}

// // Clear all logs
// function clearLogs() {
//   if (confirm('ต้องการล้างบันทึกกิจกรรมทั้งหมดใช่หรือไม่?')) {
//     // Clear all activities
//     combinedActivities.value = []

//     // Add a log entry about clearing
//     addLocalLog('info', 'ล้างบันทึกกิจกรรมทั้งหมดแล้ว')

//     // Show success notification
//     notificationStore.sendNotification({
//       title: 'ล้างบันทึกสำเร็จ',
//       message: 'ล้างบันทึกกิจกรรมทั้งหมดเรียบร้อยแล้ว',
//       type: 'success'
//     })
//   }
// }

// Get risk status text
function getRiskStatusText(level) {
  if (level < 3) return 'ต่ำ'
  if (level < 7) return 'ปานกลาง'
  return 'สูง'
}

// Get risk status class
function getRiskStatusClass(level) {
  if (level < 3) return 'risk-low'
  if (level < 7) return 'risk-medium'
  return 'risk-high'
}

// ฟังก์ชั่นสำหรับเปิด/ปิดโหมดเต็มจอของกล้อง
function toggleFullscreen(camera) {
  if (fullscreenCamera.value === camera.id) {
    fullscreenCamera.value = null
  } else {
    fullscreenCamera.value = camera.id
  }
}

// กรองการแจ้งเตือนตามกล้องที่เลือก
function selectCameraFilter(camera) {
  if (selectedCameraFilter.value === camera.id) {
    // ถ้าคลิกกล้องเดิม ให้ยกเลิกการกรอง
    selectedCameraFilter.value = null
  } else {
    // เลือกกล้องใหม่
    selectedCameraFilter.value = camera.id
  }
}

// ดึง log และ notification ทั้งหมดจากฐานข้อมูลและแสดงต่อเนื่อง (polling)
let allDataPollingInterval = null
let lastLogIds = new Set()
let lastNotificationIds = new Set()

function formatTimestamp(ts) {
  // รองรับ ISO string เช่น 2025-06-30T05:45:02.450678
  const d = new Date(ts)
  if (!isNaN(d)) {
    // แสดงวันที่และเวลาแบบ dd/MM/yyyy HH:mm:ss
    const pad = n => n.toString().padStart(2, '0')
    return `${pad(d.getDate())}/${pad(d.getMonth() + 1)}/${pad(d.getFullYear())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  }
  return ts // fallback
}

// ดึงข้อมูล logs ทั้งหมดจากฐานข้อมูล
async function fetchAllLogs() {
  try {
    const res = await logService.fetchAllLogs()

    // Check if response is directly an array of logs
    if (Array.isArray(res) && res.length > 0) {

      // Map database logs to display format
      const dbLogs = res.map(log => ({
        id: `log_${log.id}`,
        type: 'info',
        message: getDetectionResultText(log.detection_result),
        timestamp: formatTimestamp(log.timestamp),
        camera_id: log.camera_id,
        isLocal: false,
        activityType: 'log',
        detection_result: log.detection_result,
        confidence_score: log.confidence_score,
        risk_level: log.risk_level,
        person_count: log.person_count,
        camera_name: log.camera_name,
        room_name: log.room_name
      }))

      // Update combined activities
      updateCombinedActivities(dbLogs, 'log')
    } else if (res && res.logs && Array.isArray(res.logs)) {
      // Fallback: check if response has logs property (wrapped format)
      const dbLogs = res.logs.map(log => ({
        id: `log_${log.id}`,
        type: 'info',
        message: getDetectionResultText(log.detection_result),
        timestamp: formatTimestamp(log.timestamp),
        camera_id: log.camera_id,
        isLocal: false,
        activityType: 'log',
        detection_result: log.detection_result,
        confidence_score: log.confidence_score,
        risk_level: log.risk_level,
        person_count: log.person_count,
        camera_name: log.camera_name,
        room_name: log.room_name
      }))

      updateCombinedActivities(dbLogs, 'log')
    } else {
      // Clear existing database logs but keep local logs
      updateCombinedActivities([], 'log')
    }
  } catch (e) {
    console.error('fetchAllLogs - Error:', e)
    addLocalLog('error', 'ไม่สามารถโหลดบันทึกกิจกรรมได้', null, 'system')
  }
}

// ดึงข้อมูล notifications ทั้งหมดจากฐานข้อมูล
async function fetchAllNotifications() {
  try {
    const res = await logService.fetchAllNotification()

    let notificationsArray = null

    // Handle different response structures
    if (Array.isArray(res)) {
      // Response is directly an array
      notificationsArray = res
    } else if (res && res.notifications && Array.isArray(res.notifications)) {
      // Response is an object with notifications property
      notificationsArray = res.notifications
    }

    if (notificationsArray && notificationsArray.length > 0) {
      // Check for new notifications
      const newNotifications = notificationsArray.filter(notification =>
        !lastNotificationIds.has(notification.id)
      )
      // Update stored notification IDs
      notificationsArray.forEach(notification => {
        lastNotificationIds.add(notification.id)
      })

      // Store all notifications
      notifications.value = notificationsArray

      // Map notifications to display format for combined activities
      const notificationActivities = notificationsArray.map(notification => {
        const riskLevel = getRiskLevelFromDetectionType(notification.detection_type)
        const imageFileName = notification.image_path ? notification.image_path.toString().split('/').pop() : null
        const imageUrl = imageFileName ? `${import.meta.env.VITE_API_BASE_URL}/alert-images/${encodeURIComponent(imageFileName)}` : null
        setCameraRiskLevel(notification.camera_id, riskLevel, 15)
        return {
          id: `notification_${notification.id}`,
          type: 'alert',
          message: (() => {
            const camera = cameraStore.cameras.find(c => c.id === notification.camera_id)
            const cameraName = camera ? camera.name : `#${notification.camera_id}`
            const roomName = camera ? camera.room_name : ''
            const typeText = notification.detection_type === 'bed_exit' ? 'ตรวจจับการลุกจากเตียง' :
                           notification.detection_type === 'alone_yellow' ? 'ตรวจจับคนอยู่คนเดียว' :
                           notification.detection_type === 'fall_red' ? 'ตรวจจับการล้ม (อันตราย)' :
                           'ตรวจจับการล้ม'
            return `การแจ้งเตือน: ${typeText} (กล้อง: ${cameraName}${roomName ? ' - ' + roomName : ''})`
          })(),
          timestamp: formatTimestamp(notification.sent_at),
          camera_id: notification.camera_id,
          isLocal: false,
          activityType: 'notification',
          risk_level: riskLevel,
          detection_type: notification.detection_type,
          image_path: notification.image_path,
          image_url: imageUrl
        }
      })


      // Update combined activities
      updateCombinedActivities(notificationActivities, 'notification')

      // Trigger alerts for new notifications only
      newNotifications.forEach(notification => {
        triggerAlert(notification)
      })
    } else {      // Clear existing notifications but keep other activities
      notifications.value = []
      updateCombinedActivities([], 'notification')
    }
  } catch (e) {
    console.error('fetchAllNotifications - Error:', e)
    addLocalLog('error', 'ไม่สามารถโหลดการแจ้งเตือนได้')
  }
}// Update combined activities with new data
function updateCombinedActivities(newActivities, type) {
  // Remove old activities of this type (keep local logs and other types)
  combinedActivities.value = combinedActivities.value.filter(activity =>
    activity.activityType !== type || activity.isLocal
  )

  // Add new activities
  combinedActivities.value.push(...newActivities)

  // Sort by timestamp (newest first)
  combinedActivities.value.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))

  // Keep only the last 50 activities
  if (combinedActivities.value.length > 50) {
    combinedActivities.value.splice(50)
  }
}

// Start polling all data
function startAllDataPolling() {
  if (allDataPollingInterval) clearInterval(allDataPollingInterval)

  // Poll every 10 seconds
  allDataPollingInterval = setInterval(async () => {
    await fetchAllLogs()
    await fetchAllNotifications()
    // Check camera statuses every 30 seconds (every 3rd poll)
    if (Date.now() % 30000 < 10000) {
      await checkAllCameraStatuses()
    }
  }, 10000)

  // Fetch immediately on start
  fetchAllLogs()
  fetchAllNotifications()
}

function stopLogPolling() {
  if (allDataPollingInterval) {
    clearInterval(allDataPollingInterval)
  }
  allDataPollingInterval = null
  lastLogIds.clear()
  lastNotificationIds.clear()
}

// Function to remove camera
function removeCamera(camera) {
  if (confirm(`ต้องการลบกล้อง ${camera.name} ใช่หรือไม่?`)) {
    cameraStore.removeCamera(camera)

    // Show success notification
    notificationStore.sendNotification({
      title: 'ลบกล้องสำเร็จ',
      message: `ลบกล้อง ${camera.name} เรียบร้อยแล้ว`,
      type: 'success'
    })
  }
}

async function loadAvailableVideos() {
  try {
    testVideos.value = await cameraService.getTestVideos()
  } catch (error) {
    console.error('Failed to load test videos:', error)
  }
}

function startEditCamera(camera) {
  loadAvailableVideos()
  editingCameraSourceType.value = (camera.url || '').startsWith('/app/Test/') ? 'test' : 'url'

  editingCamera.value = {
    ...camera,
    notification_cooldown_sec: camera.notification_cooldown ?? 600,
    detection_type: camera.detection_type || 'bed_exit',
    alert_start_time: camera.alert_start_time || '21:00',
    alert_end_time: camera.alert_end_time || '05:00',
    ai_confidence_threshold: camera.ai_confidence_threshold ?? 0.5,
  }
  isEditing.value = true
  showEditModal.value = true
}

// ฟังก์ชันบันทึกการแก้ไขกล้อง
async function saveEditCamera() {
  if (!editingCamera.value.name || !editingCamera.value.url) {
    message.value = 'กรุณากรอกชื่อกล้องและ URL ให้ครบถ้วน'
    messageType.value = 'danger'
    return
  }

  const originalCamera = cameras.value.find(cam => cam.id === editingCamera.value.id)
  const wasMonitoring = activeMonitors.value[editingCamera.value.id]

  try {
    if (wasMonitoring && originalCamera) {
      stopMonitoring(originalCamera)
    }

    const updateData = {
      ...editingCamera.value,
      notification_cooldown: editingCamera.value.notification_cooldown_sec,
      detection_type: editingCamera.value.detection_type,
      alert_start_time: editingCamera.value.alert_start_time,
      alert_end_time: editingCamera.value.alert_end_time,
      ai_confidence_threshold: editingCamera.value.ai_confidence_threshold,
    }

    await cameraStore.updateCamera(editingCamera.value.id, updateData)

    message.value = `อัพเดทกล้อง ${editingCamera.value.name} เรียบร้อยแล้ว`
    messageType.value = 'success'
    showEditModal.value = false
    isEditing.value = false

    notificationStore.sendNotification({
      title: 'อัพเดทกล้องสำเร็จ',
      message: `อัพเดทกล้อง ${editingCamera.value.name} เรียบร้อยแล้ว`,
      type: 'success'
    })

    await cameraStore.loadCameras()

    if (wasMonitoring) {
      await nextTick()
      const updatedCamera = cameras.value.find(cam => cam.id === editingCamera.value.id)
      if (updatedCamera) {
        startMonitoring(updatedCamera)
      }
    }

    setTimeout(() => {
      message.value = ''
    }, 3000)
  } catch (err) {
    if (wasMonitoring && originalCamera) {
      startMonitoring(originalCamera)
    }

    message.value = 'เกิดข้อผิดพลาดในการอัพเดทกล้อง'
    messageType.value = 'danger'

    notificationStore.sendNotification({
      title: 'เกิดข้อผิดพลาด',
      message: 'ไม่สามารถอัพเดทกล้องได้',
      type: 'error'
    })
  }
}

function cancelEdit() {
  showEditModal.value = false
  isEditing.value = false
}

// Test function to add a sample notification (for debugging)
function addTestNotification() {
  const testNotification = {
    id: `test_notification_${Date.now()}`,
    type: 'alert',
    message: 'การแจ้งเตือนทดสอบ: ตรวจจับการลุกจากเตียง',
    timestamp: new Date().toLocaleTimeString(),
    camera_id: cameras.value[0]?.id || null,
    isLocal: false,
    activityType: 'notification'
  }

  combinedActivities.value.unshift(testNotification)
  combinedActivities.value.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
}

// Reset time filter
function resetTimeFilter() {
  timeFilterEnabled.value = false
  timeFilterStart.value = ''
  timeFilterEnd.value = ''
}

// ฟังก์ชันนับจำนวนกล้องของผู้ใช้
function getUserCameraCount(userId) {
  return allCameras.value.filter(camera =>
    camera.owner?.id === userId || camera.owner_id === userId
  ).length
}
</script>

<template>
  <div class="monitor-view">
    <div class="page-header">
      <h1 class="page-title">ระบบมอนิเตอร์</h1>
      <div class="page-header-controls">
        <!-- Blur All Toggle Button -->
        <button
          @click="toggleBlurAll"
          :class="isBlurAllActive ? 'btn btn-warning' : 'btn btn-outline-secondary'"
          class="blur-all-btn"
          :title="isBlurAllActive ? 'ยกเลิกการเบลอทั้งหมด' : 'เบลอภาพทั้งหมด'"
        >
          <i :class="isBlurAllActive ? 'fas fa-eye' : 'fas fa-eye-slash'"></i>
          {{ isBlurAllActive ? 'ยกเลิกเบลอทั้งหมด' : 'เบลอทั้งหมด' }}
        </button>
        <!-- ซ่อนปุ่มเพิ่มกล้องสำหรับ User ที่ไม่ใช่ Admin -->
        <button v-if="isAdmin" @click="goToAddCamera" class="btn btn-primary">
          <IconCamera class="btn-icon" /> เพิ่มกล้องใหม่
        </button>
      </div>
    </div>

    <div v-if="message" :class="`alert alert-${messageType} notification`">
      {{ message }}
    </div>

    <!-- แสดงข้อความเมื่อไม่มีกล้องเลย -->
    <div v-if="!hasCameras && selectedOwnerFilter === null" class="no-camera">
      <div class="no-camera-content">
        <IconCamera class="no-camera-icon" />
        <p v-if="isAdmin">ยังไม่มีกล้องในระบบ กรุณาเพิ่มกล้องก่อนใช้งานระบบมอนิเตอร์</p>
        <p v-else>ยังไม่มีกล้องในระบบของคุณ กรุณาติดต่อผู้ดูแลระบบ</p>
        <button v-if="isAdmin" @click="goToAddCamera" class="btn btn-primary">เพิ่มกล้อง</button>
      </div>
    </div>

    <div v-else class="dashboard">
      <!-- แสดงข้อความเมื่อผู้ใช้ที่เลือกไม่มีกล้อง (Admin only) -->
      <div v-if="showNoFilteredCamerasMessage && isAdmin" class="main-content">
        <div class="no-camera">
          <div class="no-camera-content">
            <IconCamera class="no-camera-icon" />
            <p>ผู้ใช้ที่เลือกยังไม่มีกล้องในระบบ</p>
            <button @click="selectedOwnerFilter = null" class="btn btn-secondary">แสดงกล้องทั้งหมด</button>
            <button @click="goToAddCamera" class="btn btn-primary">เพิ่มกล้องให้ผู้ใช้</button>
          </div>
        </div>
      </div>


      <div v-else class="main-content">
        <!-- กริดแสดงกล้อง -->
        <div class="cameras-grid" :class="{ 'fullscreen-mode': fullscreenCamera }">
          <div
            v-for="camera in cameras"
            :key="camera.id"
            class="camera-card"
            :class="{
              'motion-detected': motionDetected[camera.id],
              'camera-fullscreen': fullscreenCamera === camera.id,
              'camera-hidden': fullscreenCamera && fullscreenCamera !== camera.id,
              'camera-selected': selectedCameraFilter === camera.id,
            }"
            :data-camera-id="camera.id"
            @click="selectCameraFilter(camera)"
          >
            <div class="camera-header">
              <h3 class="camera-title">{{ camera.name }}</h3>
              <div class="camera-header-controls">
                <!-- แสดงข้อมูลเจ้าของกล้องสำหรับ admin -->
                <div v-if="isAdmin && camera.owner" class="camera-owner-info">
                  <span class="owner-text">เจ้าของ: {{ camera.owner.username }}</span>
                </div>

                <!-- ปุ่มขยายเต็มจอ - แสดงเมื่อไม่ได้อยู่ในโหมดเต็มจอ -->
                <button
                  v-if="fullscreenCamera !== camera.id"
                  @click="toggleFullscreen(camera)"
                  class="btn-icon fullscreen-toggle btn-icon-large"
                  title="ขยายเต็มจอ"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    width="20"
                    height="20"
                  >
                    <path
                      d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"
                    ></path>
                  </svg>
                </button>
                <!-- ปุ่มออกจากโหมดเต็มจอ - แสดงเมื่ออยู่ในโหมดเต็มจอ -->
                <button
                  v-if="fullscreenCamera === camera.id"
                  @click="toggleFullscreen(camera)"
                  class="btn-icon exit-fullscreen btn-icon-large"
                  title="ออกจากโหมดเต็มจอ"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    width="20"
                    height="20"
                  >
                    <path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"></path>
                  </svg>
                </button>
              </div>
            </div>

            <div class="video-container" @dblclick="fullscreenCamera === camera.id ? toggleFullscreen(camera) : null">
              <MediaViewer
                v-if="camera.id && camera.url"
                  :url="camera.url"
                  :camera-id="camera.id"
                  :use-stream-api="isRtspOrStreamUrl(camera.url)"
                  :alt-text="`Camera feed for ${camera.name}`"
                  :is-fullscreen="fullscreenCamera === camera.id"
                  :is-blurred="isBlurAllActive"
                  :muted="true"
                  :autoplay="true"
                  :loop="true"
                  :show-controls="fullscreenCamera === camera.id"
                  @load="handleMediaLoad(camera, $event)"
                  @error="handleMediaError(camera, $event)"
              />
            <div v-else class="video-placeholder">
              <span>กำลังโหลด...</span>
            </div>

              <div v-if="motionDetected[camera.id]" class="motion-alert">
                <IconAlert /> ตรวจพบการเคลื่อนไหว
              </div>

              <!-- ซ่อนปุ่มลบและแก้ไขเมื่ออยู่ในโหมดเต็มจอ หรือเมื่อกำลังตรวจจับ หรือไม่ใช่ Admin -->
              <div v-if="isAdmin && !fullscreenCamera && !activeMonitors[camera.id]" class="camera-action-buttons" @click.stop>
                  <button class="btn btn-danger btn-sm" @click="removeCamera(camera)">ลบ</button>
                  <button class="btn btn-secondary btn-sm" @click="startEditCamera(camera)">แก้ไข</button>
              </div>
            </div>

            <div class="camera-controls">
              <div v-if="riskLevel[camera.id] && riskLevel[camera.id] !== 'normal'" class="risk-indicator">
                <span class="risk-level" :class="getRiskLevelClass(riskLevel[camera.id])">
                  {{ getRiskLevelText(riskLevel[camera.id]) }}
                </span>
              </div>

              <!-- Camera Status Information -->
              <div v-if="cameraStatuses[camera.id]" class="camera-status-info">
                <div class="status-row">
                  <span class="status-label">สถานะ:</span>
                  <span class="status-value" :class="getStatusClass(cameraStatuses[camera.id].status)">
                    {{ getStatusText(cameraStatuses[camera.id].status) }}
                  </span>
                </div>
                <div v-if="cameraStatuses[camera.id].last_activity" class="status-row">
                  <span class="status-label">กิจกรรมล่าสุด:</span>
                  <span class="status-value">
                    {{ getDetectionResultText(cameraStatuses[camera.id].last_activity.detection_result) }}
                    <!-- ({{ Math.round(cameraStatuses[camera.id].last_activity.confidence * 100) }}%) -->
                  </span>
                </div>
              </div>

              <button
                v-if="!activeMonitors[camera.id]"
                @click="startMonitoring(camera)"
                :disabled="operationInProgress[camera.id]"
                class="btn btn-sm btn-success"
              >
                {{ operationInProgress[camera.id] ? 'กำลังเริ่ม...' : 'เริ่มการตรวจจับ' }}
              </button>
              <button
                v-else
                @click="stopMonitoring(camera)"
                :disabled="operationInProgress[camera.id]"
                class="btn btn-sm btn-danger"
              >
                {{ operationInProgress[camera.id] ? 'กำลังหยุด...' : 'หยุดการตรวจจับ' }}
              </button>
            </div>
          </div>
        </div>

        <!-- บันทึกกิจกรรม -->
        <div class="log-section card" :class="{ 'hidden-when-fullscreen': fullscreenCamera }">
          <div class="log-header">
            <div class="log-title-section">
              <h2 class="card-title">บันทึกกิจกรรม</h2>
              <div v-if="selectedCameraFilter" class="filter-indicator">
                <span class="filter-text">กรองจากกล้อง: {{ cameras.find(c => c.id === selectedCameraFilter)?.name }}</span>
                <button @click="selectedCameraFilter = null" class="btn-clear-filter">×</button>
              </div>
            </div>
          </div>

          <div class="log-entries">
            <div v-if="filteredActivities.length === 0" class="empty-log">
              <p>ยังไม่มีบันทึกกิจกรรม</p>
            </div>

            <div v-for="activity in filteredActivities" :key="activity.id" class="log-entry" :class="[activity.type, { 'db-log': !activity.isLocal, 'notification-entry': activity.activityType === 'notification' }]">
              <span class="log-time">{{ activity.timestamp }}</span>
              <div class="log-content">
                <span class="log-message">{{ activity.message }}</span>
                <div v-if="activity.image_url" class="log-image">
                  <a :href="activity.image_url" target="_blank" rel="noopener noreferrer">
                    <img :src="activity.image_url" alt="Alert image" class="alert-image" />
                  </a>
                </div>
                <!-- แสดงข้อมูลเพิ่มเติมสำหรับข้อมูลกิจกรรม -->
                <div v-if="activity.risk_level" class="log-details">
                  <span v-if="activity.person_count !== undefined" class="person-count">
                    {{ getPersonCountText(activity.person_count) }}
                  </span>
                  <!-- <span v-if="activity.confidence_score" class="confidence-score">
                    ความแม่นยำ: {{ Math.round(activity.confidence_score * 100) }}%
                  </span> -->
                  <span v-if="activity.risk_level" class="risk-level" :class="getRiskLevelClass(activity.risk_level)">
                    {{ getRiskLevelText(activity.risk_level) }}
                  </span>
                </div>
                <span v-if="activity.camera_id" class="camera-source">
                  จากกล้อง: {{ getCameraName(activity.camera_id) }}
                </span>
              </div>
              <span v-if="!activity.isLocal" class="log-source" :class="{ 'log-source-noti': activity.activityType === 'notification', 'log-source-log': activity.activityType !== 'notification' }">{{ activity.activityType === 'notification' ? 'NOTI' : 'LOG' }}</span>
            </div>
          </div>
        </div>
      </div>

      <div class="sidebar" :class="{ 'hidden-when-fullscreen': fullscreenCamera }">
        <!-- แถบเลื่อนผู้ใช้สำหรับ admin -->
        <div v-if="isAdmin && usersForFilter.length > 0" class="card user-filter-card">
          <h3 class="card-title">เลือกเจ้าของกล้อง</h3>
          <p class="filter-description">เลือกผู้ใช้เพื่อดูกล้องเฉพาะของผู้ใช้นั้น</p>

          <div class="user-filter-list">
            <button
              @click="selectedOwnerFilter = null"
              class="user-filter-btn"
              :class="{ 'active': selectedOwnerFilter === null }"
            >
              <span class="filter-text">แสดงทั้งหมด</span>
              <span class="camera-count">{{ allCameras.length }} กล้อง</span>
            </button>

            <button
              v-for="user in usersForFilter"
              :key="user.id"
              @click="selectedOwnerFilter = user.id"
              class="user-filter-btn"
              :class="{ 'active': selectedOwnerFilter === user.id }"
            >
              <span class="filter-text">{{ user.username }}</span>
              <span class="user-role">{{ user.role === 'admin' ? 'Admin' : 'User' }}</span>
              <span class="camera-count">{{ getUserCameraCount(user.id) }} กล้อง</span>
            </button>
          </div>
        </div>

        <!-- การตั้งค่าการแจ้งเตือน -->
        <!-- <div class="card settings-card">
          <h2 class="card-title">ตั้งค่าการแจ้งเตือน</h2>
          <p class="settings-description">การตั้งค่านี้จะอัปเดตเวลาการแจ้งเตือนสำหรับกล้องทุกตัวในระบบ</p>

          <div class="form-group">
            <label class="form-label">ช่วงเวลาแจ้งเตือน (รูปแบบ 24 ชั่วโมง)</label>
            <div class="time-range">
              <input
                type="text"
                v-model="notificationPeriod.start"
                class="form-input time-input"
                :disabled="isSavingSettings"
                pattern="^([01]?[0-9]|2[0-3]):[0-5][0-9]$"
                placeholder="21:00"
                maxlength="5"
              />
              <span>ถึง</span>
              <input
                type="text"
                v-model="notificationPeriod.end"
                class="form-input time-input"
                :disabled="isSavingSettings"
                pattern="^([01]?[0-9]|2[0-3]):[0-5][0-9]$"
                placeholder="05:00"
                maxlength="5"
              />
            </div>
            <small class="form-help">รูปแบบ: ชั่วโมง:นาที (00:00 - 23:59) เช่น 21:00 ถึง 05:00</small>
          </div>

          <button
            @click="saveNotificationSettings"
            class="btn btn-primary btn-save"
            :disabled="isSavingSettings"
            :class="{ 'loading': isSavingSettings }"
          >
            <span v-if="!isSavingSettings">บันทึกการตั้งค่า</span>
            <span v-else>กำลังบันทึก...</span>
          </button>
        </div>-->

        <!-- การตั้งค่าประเภทการตรวจจับ -->
        <!-- <div class="card settings-card">
          <h2 class="card-title">ตั้งค่าประเภทการตรวจจับ</h2>
          <p class="settings-description">การตั้งค่านี้จะอัปเดตประเภทการตรวจจับสำหรับกล้องทุกตัวในระบบ</p>

          <div class="form-group">
            <label class="form-label">ประเภทการตรวจจับ</label>
            <select
              v-model="globalDetectionType"
              class="form-input"
              :disabled="isSavingSettings"
            >
              <option v-for="opt in DETECTION_TYPE_OPTIONS" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
            </select>
          </div>

          <button
            @click="saveGlobalDetectionType"
            class="btn btn-primary btn-save"
            :disabled="isSavingSettings"
            :class="{ 'loading': isSavingSettings }"
          >
            <span v-if="!isSavingSettings">อัปเดตการตรวจจับ</span>
            <span v-else>กำลังอัปเดต...</span>
          </button>
        </div> -->

        <!-- การแจ้งเตือนล่าสุด -->
        <div class="card notifications-card">
          <h3 class="card-title">กิจกรรมล่าสุด</h3>
          <div class="notifications-list">
            <div v-if="filteredActivities.length === 0" class="empty-notifications">
              <p>ยังไม่มีกิจกรรม</p>
            </div>
            <div v-for="activity in combinedActivities.slice(0, 5)" :key="activity.id" class="notification-item" :class="{ 'alert-item': activity.activityType === 'notification' }">
              <div class="notification-type" :class="{ 'alert-type': activity.activityType === 'notification' }">
                {{ activity.activityType === 'notification' ? `🚨 ${activity.message}` : activity.message }}
              </div>
              <div class="notification-time">{{ activity.timestamp }}</div>
              <!-- <div class="activity-source">{{ activity.activityType === 'notification' ? 'แจ้งเตือน' : (activity.isLocal ? 'ระบบ' : 'ฐานข้อมูล') }}</div> -->
            </div>
          </div>
        </div>

        <!-- สถิติ -->
        <div class="stats-card card">
          <h3 class="card-title">สถิติการตรวจจับ</h3>
          <div class="stats-grid">
            <div class="stat-item">
              <span class="stat-value">{{ eventCount }}</span>
              <span class="stat-label">เหตุการณ์ทั้งหมด</span>
            </div>
            <div class="stat-item">
              <span class="stat-value">{{ alertCount }}</span>
              <span class="stat-label">การแจ้งเตือน</span>
            </div>
            <div class="stat-item">
              <span class="stat-value">{{ cameras.length }}</span>
              <span class="stat-label">กล้องทั้งหมด</span>
            </div>
            <div class="stat-item">
              <span class="stat-value">{{
                Object.values(activeMonitors).filter((v) => v).length
              }}</span>
              <span class="stat-label">กล้องที่ตรวจจับ</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Modal สำหรับแก้ไขกล้อง - กดออกได้โดยกดพื้นที่ว่างข้างๆ -->
    <div v-if="showEditModal" class="modal-overlay" @click.self="cancelEdit" @click.stop>
      <div class="modal-content">
        <div class="modal-header">
          <h3>แก้ไขกล้อง</h3>
          <button @click="cancelEdit" class="modal-close-btn">&times;</button>
        </div>
        <div class="modal-body">
          <form @submit.prevent="saveEditCamera">

            <div class="form-group">
              <label for="edit-camera-name" class="form-label"
                >ชื่อกล้อง <span class="required">*</span></label
              >
              <input
                type="text"
                id="edit-camera-name"
                v-model="editingCamera.name"
                class="form-input"
                placeholder="เช่น กล้องหน้าบ้าน, กล้องหลังบ้าน"
                autofocus
              />
            </div>

            <div class="form-group">
              <label for="edit-room-name" class="form-label">ชื่อห้อง/โซน</label>
              <input
                type="text"
                id="edit-room-name"
                v-model="editingCamera.room_name"
                class="form-input"
                placeholder="เช่น ห้อง 101, โถงกลาง, ห้องผู้ป่วย 2"
              />
            </div>

            <div class="form-group">
              <label class="form-label">แหล่งวิดีโอ</label>
              <div class="source-type-toggle">
                <label>
                  <input type="radio" value="url" v-model="editingCameraSourceType" />
                  กล้องจริง (RTSP/URL)
                </label>
                <label>
                  <input type="radio" value="test" v-model="editingCameraSourceType" />
                  ไฟล์วิดีโอทดสอบ
                </label>
              </div>
            </div>

            <div class="form-group" v-if="editingCameraSourceType === 'url'">
              <label for="edit-camera-url" class="form-label"
                >URL การเชื่อมต่อ <span class="required">*</span></label
              >
              <input
                type="text"
                id="edit-camera-url"
                v-model="editingCamera.url"
                class="form-input"
                placeholder="เช่น rtsp://username:password@ip:port/path"
              />
            </div>

            <div class="form-group" v-else>
              <label for="edit-camera-test-video" class="form-label"
                >เลือกไฟล์วิดีโอทดสอบ <span class="required">*</span></label
              >
              <select id="edit-camera-test-video" v-model="editingCamera.url" class="form-input">
                <option value="" disabled>เลือกไฟล์วิดีโอ</option>
                <option v-for="v in testVideos" :key="v.filename" :value="v.url">
                  {{ v.filename }}
                </option>
              </select>
              <small class="form-help">ไฟล์จากโฟลเดอร์ Test/ ของ backend</small>
            </div>

            <div class="form-group">
              <label for="edit-detection-type" class="form-label">ประเภทการตรวจจับ</label>
              <select id="edit-detection-type" v-model="editingCamera.detection_type" class="form-input">
                <option v-for="opt in DETECTION_TYPE_OPTIONS" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
              </select>
              <small class="form-help">{{ DETECTION_TYPE_FORM_HELP }}</small>
            </div>

            <div class="form-group">
              <label for="edit-alert-start-time" class="form-label">เวลาเริ่มการแจ้งเตือน (24 ชั่วโมง)</label>
              <input
                type="text"
                id="edit-alert-start-time"
                v-model="editingCamera.alert_start_time"
                class="form-input"
                pattern="^([01]?[0-9]|2[0-3]):[0-5][0-9]$"
                placeholder="เช่น 21:00"
                maxlength="5"
              />
              <small class="form-help">รูปแบบ: ชั่วโมง:นาที (00:00 - 23:59) เช่น 21:00 หรือ 09:00</small>
            </div>

            <div class="form-group">
              <label for="edit-alert-end-time" class="form-label">เวลาสิ้นสุดการแจ้งเตือน (24 ชั่วโมง)</label>
              <input
                type="text"
                id="edit-alert-end-time"
                v-model="editingCamera.alert_end_time"
                class="form-input"
                pattern="^([01]?[0-9]|2[0-3]):[0-5][0-9]$"
                placeholder="เช่น 05:00"
                maxlength="5"
              />
              <small class="form-help">รูปแบบ: ชั่วโมง:นาที (00:00 - 23:59) เช่น 05:00 หรือ 17:00</small>
            </div>

            <div class="form-group">
              <label for="edit-notification-cooldown" class="form-label">ระยะห่างการแจ้งเตือน (วินาที)</label>
              <input type="number" id="edit-notification-cooldown" v-model.number="editingCamera.notification_cooldown_sec" class="form-input" min="1" step="1" />
              <small class="form-help">เวลาที่ต้องรอก่อนแจ้งเตือนครั้งต่อไป (ป้องกันการแจ้งเตือนซ้ำเร็วเกินไป) เช่น 30 วินาที, 60 วินาที (1 นาที), 600 วินาที (10 นาที)</small>
            </div>

            <div class="form-group">
              <label for="edit-ai-confidence-threshold" class="form-label">ความแม่นยำของ AI (0.0-1.0)</label>
              <input type="number" id="edit-ai-confidence-threshold" v-model.number="editingCamera.ai_confidence_threshold" class="form-input" min="0" max="1" step="0.01" />
              <small class="form-help">ระดับความมั่นใจของ AI ที่จะแจ้งเตือน (0.5 = 50%, ยิ่งสูงยิ่งแม่นยำ)</small>
            </div>

            <div class="form-actions">
              <button type="button" @click="cancelEdit" class="btn btn-secondary">ยกเลิก</button>
              <button type="submit" class="btn btn-primary">บันทึกการเปลี่ยนแปลง</button>
            </div>
          </form>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.monitor-view {
  padding-bottom: 2rem;
}


/* Activity Cards: Modern, clean design */
.activity-card {
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  border-radius: 12px;
  margin-bottom: 16px;
  box-shadow: 0 2px 8px rgba(30,41,59,0.08);
  padding: 16px;
  border: 1px solid #e2e8f0;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.activity-card:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(30,41,59,0.12);
}

.activity-card.alert-activity {
  background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
  border-color: #fecaca;
}

.activity-content {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.activity-main {
  flex: 1;
  min-width: 0; /* Allow text to wrap */
}

.activity-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  gap: 12px;
}

.activity-result {
  font-size: 1.1rem;
  font-weight: 600;
  color: #1e293b;
  line-height: 1.4;
  word-break: break-word;
}

.activity-result.alert-result {
  color: #dc2626;
  font-weight: 700;
}

.activity-time {
  font-size: 0.85rem;
  font-weight: 500;
  color: #9ca3af;
  flex-shrink: 0;
}

.activity-camera {
  font-size: 0.9rem;
  font-weight: 600;
  color: #059669;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.activity-indicator {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #3b82f6;
  margin-top: 6px;
  flex-shrink: 0;
}

.activity-indicator.alert-indicator {
  background: #dc2626;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.page-header-controls {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.blur-all-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.875rem;
  font-weight: 500;
  transition: all 0.2s ease;
}

.blur-all-btn:hover {
  transform: translateY(-1px);
}

.btn-icon {
  width: 16px;
  height: 16px;
  margin-right: 4px;
}

.btn-icon-large {
  width: 40px !important;
  height: 40px !important;
  padding: 8px;
  background: rgba(0, 0, 0, 0.7);
  border: none;
  border-radius: 6px;
  color: white;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  justify-content: center;
}

.btn-icon-large:hover {
  background: rgba(0, 0, 0, 0.9);
  transform: scale(1.05);
}

.no-camera {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.no-camera-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
  padding: 2rem;
}

.no-camera-icon {
  width: 4rem;
  height: 4rem;
  color: #d1d5db;
}

.dashboard {
  display: flex;
  gap: 1.5rem;
}

.main-content {
  flex: 3;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.sidebar {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.cameras-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1.5rem;
}

.camera-card {
  background-color: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
  transition: all 0.3s ease;
  cursor: pointer;
}

.camera-card:hover {
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

.camera-card.motion-detected {
  box-shadow: 0 0 0 3px #dc2626;
}

.camera-card.camera-selected {
  box-shadow: 0 0 0 3px #2563eb;
  background-color: #eff6ff;
}

.camera-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0.75rem;
  background-color: #f3f4f6;
  min-height: auto;
}

.camera-title {
  font-size: 0.875rem;
  font-weight: 600;
  margin: 0;
  line-height: 1.2;
}

.camera-status {
  font-size: 0.75rem;
  font-weight: 500;
  padding: 0.25rem 0.5rem;
  border-radius: 9999px;
}

.camera-status.online {
  background-color: #d1fae5;
  color: #065f46;
}

.camera-status.offline {
  background-color: #f3f4f6;
  color: #6b7280;
}

.video-container {
  position: relative;
  width: 100%;
  height: 180px;
  background-color: #000;
}

.video-container video {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.video-container video::-webkit-media-controls {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(0, 0, 0, 0.7);
  z-index: 5;
}

.video-container img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.video-container .loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.875rem;
  z-index: 6;
}

.motion-alert {
  position: absolute;
  top: 8px;
  left: 8px;
  background-color: rgba(220, 38, 38, 0.9);
  color: white;
  padding: 4px 8px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.875rem;
  font-weight: 500;
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}

.camera-controls {
  padding: 0.75rem 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.risk-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.risk-level {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 4px;
}

.risk-low {
  background-color: #d1fae5;
  color: #065f46;
}

.risk-medium {
  background-color: #fff7ed;
  color: #9a3412;
}

.risk-high {
  background-color: #fee2e2;
  color: #b91c1c;
}

.risk-value {
  font-size: 0.875rem;
  color: #6b7280;
}

.camera-status-info {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.75rem;
  margin: 0.5rem 0;
}

.status-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.status-label {
  color: #6b7280;
  font-weight: 500;
  min-width: 80px;
}

.status-value {
  font-weight: 600;
}

.status-running {
  color: #059669;
}

.status-stopped {
  color: #6b7280;
}

.status-error {
  color: #dc2626;
}

.status-unknown {
  color: #f59e0b;
}

/* Blur button styles */
.blur-toggle-btn {
  margin-left: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.375rem;
  font-size: 0.875rem;
  transition: all 0.2s ease;
}

.blur-toggle-btn:hover {
  transform: translateY(-1px);
}

.log-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.log-title-section {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.filter-indicator {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.375rem 0.75rem;
  background-color: #dbeafe;
  border: 1px solid #3b82f6;
  border-radius: 0.375rem;
  font-size: 0.875rem;
}

.filter-text {
  color: #1e40af;
  font-weight: 500;
}

.btn-clear-filter {
  background: none;
  border: none;
  color: #1e40af;
  font-size: 1.25rem;
  font-weight: bold;
  cursor: pointer;
  padding: 0;
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: background-color 0.2s;
}

.btn-clear-filter:hover {
  background-color: #1e40af;
  color: white;
}

.log-controls {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-bottom: 1rem;
}

.time-filter-section {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 1rem;
  background-color: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 0.5rem;
}

.time-filter-toggle {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  font-weight: 500;
  color: #374151;
}

.time-filter-toggle input[type="checkbox"] {
  width: 16px;
  height: 16px;
  accent-color: #3b82f6;
}

.filter-label {
  font-size: 0.875rem;
  user-select: none;
}

.time-filter-inputs {
  display: flex;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
  margin-top: 0.5rem;
}

.time-input-group {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.time-input-group label {
  font-size: 0.875rem;
  color: #6b7280;
  font-weight: 500;
  min-width: 50px;
}

.time-input {
  padding: 0.375rem 0.5rem;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  background-color: white;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.time-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.action-buttons {
  display: flex;
  gap: 0.5rem;
  justify-content: flex-end;
  align-items: center;
}

.action-buttons .btn {
  font-size: 0.875rem;
  padding: 0.375rem 0.75rem;
}

.log-entries {
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid #e5e7eb;
  border-radius: 0.375rem;
  background-color: white;
}

.log-entry {
  padding: 0.75rem;
  border-bottom: 1px solid #e5e7eb;
  font-size: 0.875rem;
  display: grid;
  grid-template-columns: 80px 1fr auto;
  gap: 1rem;
  align-items: start;
}

.log-entry:last-child {
  border-bottom: none;
}

.log-entry.alert {
  border-left: 3px solid #dc2626;
}

.log-entry.motion {
  border-left: 3px solid #f59e0b;
}

.log-entry.info {
  border-left: 3px solid #3b82f6;
}

.log-entry.db-log {
  background-color: #f8fafc;
}

.log-entry.notification-entry {
  background-color: #fef2f2;
  border-left: 3px solid #dc2626;
}

.log-content {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  justify-content: center;
  min-height: 1.25rem;
}

.log-time {
  font-size: 0.75rem;
  color: #6b7280;
  font-weight: 500;
  align-self: start;
  padding-top: 0.125rem;
}

.log-message {
  font-weight: 500;
  line-height: 1.25rem;
}

.log-image {
  margin-top: 0.5rem;
}

.alert-image {
  max-width: 160px;
  max-height: 90px;
  border-radius: 8px;
  border: 1px solid #e5e7eb;
  object-fit: cover;
}

.camera-source {
  font-size: 0.75rem;
  color: #6b7280;
  font-style: italic;
}

.log-details {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  margin-top: 0.25rem;
  flex-wrap: wrap;
}

.person-count,
.confidence-score {
  font-size: 0.75rem;
  color: #4b5563;
  background-color: #f3f4f6;
  padding: 2px 6px;
  border-radius: 12px;
  font-weight: 500;
}

.risk-level {
  font-size: 0.75rem;
  padding: 2px 8px;
  border-radius: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.025em;
}

.risk-green {
  background-color: #dcfce7;
  color: #166534;
  border: 1px solid #bbf7d0;
}

.risk-yellow {
  background-color: #fef3c7;
  color: #92400e;
  border: 1px solid #fde68a;
}

.risk-red {
  background-color: #fecaca;
  color: #b91c1c;
  border: 1px solid #f87171;
}

.risk-unknown {
  background-color: #e5e7eb;
  color: #6b7280;
  border: 1px solid #d1d5db;
}

.log-source {
  font-size: 0.75rem;
  background-color: #e2e8f0;
  color: #475569;
  padding: 2px 6px;
  border-radius: 12px;
  font-weight: 500;
  align-self: start;
  margin-top: 0.125rem;
}

.log-source-noti {
  background-color: #fecaca;
  color: #b91c1c;
  border: 1px solid #f87171;
}

.log-source-log {
  background-color: #dbeafe;
  color: #1e40af;
  border: 1px solid #60a5fa;
}

.empty-log {
  padding: 2rem;
  text-align: center;
  color: #6b7280;
}

.time-range {
  display: flex;
  align-items: center;
  gap: 8px;
}

.time-input {
  width: 100%;
  max-width: 120px;
}

.btn-save {
  margin-top: 1rem;
  transition: all 0.3s ease;
}

.btn-save:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-save.loading {
  background-color: #9ca3af;
}

.settings-description {
  font-size: 0.875rem;
  color: #6b7280;
  margin-bottom: 1rem;
  line-height: 1.4;
}

.settings-card,
.stats-card,
.notifications-card {
  margin-bottom: 0;
}

.notifications-list {
  max-height: 200px;
  overflow-y: auto;
}

/* Empty notifications state */
.empty-notifications {
  padding: 2rem 1rem;
  text-align: center;
  color: #64748b;
  font-size: 0.95rem;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  border-radius: 12px;
  border: 2px dashed #e2e8f0;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.75rem;
}

.stat-item {
  padding: 0.75rem;
  background-color: #f9fafb;
  border-radius: 6px;
  text-align: center;
}

.stat-value {
  display: block;
  font-size: 1.5rem;
  font-weight: 700;
  color: #2563eb;
}

.stat-label {
  font-size: 0.75rem;
  color: #6b7280;
}

.camera-header-controls {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 0.25rem;
}

.camera-owner-info {
  font-size: 0.7rem;
  margin-bottom: 0.125rem;
}

.owner-text {
  color: #1e40af;
  font-weight: 500;
  line-height: 1.1;
}

.fullscreen-toggle {
  background: none;
  border: none;
  color: #6b7280;
  cursor: pointer;
  padding: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.fullscreen-toggle:hover {
  background-color: rgba(0, 0, 0, 0.1);
  color: #1e40af;
}

.fullscreen-toggle svg {
  width: 16px;
  height: 16px;
}

.camera-fullscreen {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  width: 100%;
  height: 100vh;
  z-index: 1000;
  margin: 0;
  border-radius: 0;
  background-color: black;
  padding: 0;
  display: flex;
  flex-direction: column;
}

.camera-fullscreen .camera-header {
  background-color: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 12px 16px;
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1001;
}

.camera-fullscreen .camera-controls {
  background-color: rgba(0, 0, 0, 0.7);
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 1001;
  padding: 12px 16px;
}

.camera-fullscreen .video-container {
  flex: 1;
  height: 100%;
}

.camera-fullscreen .video-container video {
  object-fit: contain;
  width: 100%;
  height: 100%;
}

.camera-hidden {
  display: none;
}

.hidden-when-fullscreen {
  display: none;
}

.fullscreen-mode {
  display: block;
}

.camera-action-buttons {
  position: absolute;
  top: 8px;
  right: 8px;
  display: flex;
  gap: 0.5rem;
  z-index: 10;
  background: rgba(0, 0, 0, 0.7);
  padding: 0.25rem 0.5rem;
  border-radius: 0.375rem;
  backdrop-filter: blur(4px);
}

.camera-action-buttons .btn {
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
  border: none;
  border-radius: 0.25rem;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s ease;
  color: white;
  min-width: 40px;
}

.camera-action-buttons .btn-danger {
  background-color: #dc2626;
}

.camera-action-buttons .btn-danger:hover {
  background-color: #b91c1c;
  transform: translateY(-1px);
}

.camera-action-buttons .btn-secondary {
  background-color: #6b7280;
}

.camera-action-buttons .btn-secondary:hover {
  background-color: #4b5563;
  transform: translateY(-1px);
}

.camera-action-buttons .btn:active {
  transform: translateY(0);
}

/* ปรับหน้าจอโทรศัพท์ */
@media (max-width: 768px) {
  .dashboard {
    flex-direction: column;
  }

  .cameras-grid {
    grid-template-columns: 1fr;
  }

  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .time-filter-inputs {
    flex-direction: column;
    align-items: stretch;
    gap: 0.75rem;
  }

  .time-input-group {
    justify-content: space-between;
  }

  .time-input-group label {
    min-width: auto;
    flex-shrink: 0;
  }

  .action-buttons {
    flex-direction: column;
    gap: 0.5rem;
  }

  .action-buttons .btn {
    width: 100%;
  }
}

/* Modal styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.modal-content {
  background-color: white;
  border-radius: 8px;
  max-width: 500px;
  width: 100%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  animation: modal-appear 0.3s ease-out;
}

@keyframes modal-appear {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.modal-header {
  padding: 1rem;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.modal-header h3 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
}

.modal-close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: #6b7280;
}

.modal-body {
  padding: 1.5rem;
}

.form-group {
  margin-bottom: 1rem;
}

.form-label {
  display: block;
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
  margin-bottom: 0.25rem;
}

.form-input {
  width: 100%;
  padding: 0.5rem 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
}

.form-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.form-help {
  display: block;
  color: #6b7280;
  font-size: 0.75rem;
  margin-top: 0.25rem;
}

.source-type-toggle {
  display: flex;
  gap: 1.5rem;
  padding: 0.5rem 0;
}

.source-type-toggle label {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-weight: normal;
  cursor: pointer;
}

.required {
  color: #dc2626;
}

.form-actions {
  display: flex;
  justify-content: flex-end;
  gap: 1rem;
  margin-top: 1.5rem;
}

.notification {
  margin-bottom: 1rem;
  padding: 0.75rem 1rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
}

.alert-success {
  background-color: #d1fae5;
  border: 1px solid #a7f3d0;
  color: #065f46;
}

.alert-danger {
  background-color: #fee2e2;
  border: 1px solid #fca5a5;
  color: #b91c1c;
}

.alert-info {
  background-color: #dbeafe;
  border: 1px solid #93c5fd;
  color: #1e3a8a;
}

/* User filter styles */
.user-filter-card {
  background: white;
  border-radius: 8px;
  padding: 1rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.user-filter-card .card-title {
  margin: 0 0 0.5rem 0;
  font-size: 1.125rem;
  font-weight: 600;
  color: #1f2937;
}

.filter-description {
  font-size: 0.875rem;
  color: #6b7280;
  margin-bottom: 1rem;
}

.user-filter-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  max-height: 200px;
  overflow-y: auto;
}

.user-filter-btn {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: 0.75rem;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  background: #f9fafb;
  cursor: pointer;
  transition: all 0.2s ease;
  text-align: left;
  width: 100%;
}

.user-filter-btn:hover {
  background: #f3f4f6;
  border-color: #d1d5db;
}

.user-filter-btn.active {
  background: #dbeafe;
  border-color: #3b82f6;
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
}

.filter-text {
  font-weight: 600;
  color: #1f2937;
  font-size: 0.875rem;
}

.user-role {
  font-size: 0.75rem;
  color: #6b7280;
  margin-top: 0.25rem;
}

.camera-count {
  font-size: 0.75rem;
  color: #059669;
  font-weight: 500;
  margin-top: 0.25rem;
}

.btn-info {
  background-color: #0ea5e9;
  color: white;
  border: 1px solid #0ea5e9;
}

.btn-info:hover {
  background-color: #0284c7;
  border-color: #0284c7;
}

.video-examples {
  margin-top: 0.75rem;
  padding: 0.75rem;
  background-color: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 0.375rem;
}

.form-label-small {
  display: block;
  font-size: 0.75rem;
  font-weight: 600;
  color: #374151;
  margin-bottom: 0.5rem;
}

.video-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.video-example-btn {
  background-color: white;
  border: 1px solid #d1d5db;
  border-radius: 0.25rem;
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
  color: #374151;
}

.video-example-btn:hover {
  background-color: #3b82f6;
  border-color: #3b82f6;
  color: white;
  transform: translateY(-1px);
}

.more-videos {
  color: #6b7280;
  font-style: italic;
  align-self: center;
  margin-left: 0.5rem;
}
</style>
