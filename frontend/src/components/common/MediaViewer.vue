<template>
  <div class="media-viewer" :class="{ 'fullscreen': isFullscreen, 'blurred': isBlurred }">
    <!-- MJPEG Stream Display -->
    <template v-if="isMjpegStream(processedUrl) || isHttpStream(processedUrl)">
      <img
        :src="processedUrl"
        :alt="altText"
        :class="['media-content', 'mjpeg-stream', { 'blurred-content': isBlurred }]"
        @error="handleError('mjpeg')"
        @load="handleLoad('mjpeg')"
        @loadstart="handleLoadStart('mjpeg')"
        ref="mjpegElement"
        :key="processedUrl"
      />
    </template>

    <!-- Regular Image Display -->
    <template v-else-if="isImageUrl(processedUrl)">
      <img
        :src="processedUrl"
        :alt="altText"
        :class="['media-content', { 'blurred-content': isBlurred }]"
        @error="handleError('image')"
        @load="handleLoad('image')"
        :key="processedUrl"
      />
    </template>

    <!-- Video File Display -->
    <template v-else-if="isVideoFile(processedUrl) || isStreamingUrl(processedUrl)">
      <video
        ref="videoElement"
        :muted="muted"
        :autoplay="autoplay"
        :loop="loop"
        :controls="showControls"
        :src="processedUrl"
        :class="['media-content', { 'blurred-content': isBlurred }]"
        @error="handleError('video')"
        @loadstart="handleLoadStart('video')"
        @loadeddata="handleLoad('video')"
        @loadedmetadata="handleMetadataLoaded"
        @canplay="handleCanPlay"
        :key="processedUrl"
      >
        <p>เบราว์เซอร์ไม่รองรับการเล่นวิดีโอนี้</p>
      </video>
    </template>

    <!-- Fallback Display -->
    <template v-else>
      <img
        :src="placeholderImage"
        :alt="altText"
        :class="['media-content', { 'blurred-content': isBlurred }]"
      />
    </template>

    <!-- Loading Overlay with Fallback Image -->
    <div v-if="isLoading" class="loading-overlay">
      <img :src="loadingFallbackImage" class="fallback-bg" />
      <div class="logo-container">
        <img :src="v89Logo" class="v89-logo" alt="V89 Logo" />
      </div>
      <div class="loading-content">
        <div class="loading-spinner"></div>
        <span>กำลังโหลด...</span>
      </div>
    </div>

    <!-- Error Overlay with Fallback Image -->
    <div v-if="hasError" class="error-overlay">
      <img :src="errorFallbackImage" class="fallback-bg" />
      <div class="logo-container error-logo-container">
        <img :src="v89Logo" class="v89-logo error-logo" alt="V89 Logo" />
      </div>
      <div class="error-content">
        <div class="error-icon">⚠️</div>
        <span>{{ errorMessage }}</span>
        <div class="error-suggestion" v-if="errorMessage">
          กรุณาตรวจสอบ URL หรือลองใหม่อีกครั้ง
        </div>
      </div>
    </div>
  </div>
</template>

<!-- SETUP -->
<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { convertToServerUrl, isLocalPath, getSuggestedServerUrl } from '@/utils/videoUtils'
import v89Logo from '@/assets/V89_logo.png'

const props = defineProps({
  url: {
    type: String,
    required: true
  },
  cameraId: {
    type: [Number, String],
    default: null
  },
  altText: {
    type: String,
    default: 'Media content'
  },
  muted: {
    type: Boolean,
    default: true
  },
  autoplay: {
    type: Boolean,
    default: true
  },
  loop: {
    type: Boolean,
    default: true
  },
  showControls: {
    type: Boolean,
    default: true
  },
  isFullscreen: {
    type: Boolean,
    default: false
  },
  useStreamApi: {
    type: Boolean,
    default: false
  },
  isBlurred: {
    type: Boolean,
    default: false
  },
  overlay: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits([
  'load',
  'error',
  'loadstart',
  'canplay',
  'metadata-loaded'
])

const isLoading = ref(false)
const hasError = ref(false)
const errorMessage = ref('')
const videoElement = ref(null)
const mjpegElement = ref(null)
const showLocalPathWarning = ref(false)

const processedUrl = computed(() => {
  if (!props.url) return ''

  // ถ้าใช้ Stream API และมี cameraId ให้ใช้ stream endpoint
  if (props.useStreamApi && props.cameraId) {
    const streamUrl = getMjpegStreamUrl(props.cameraId, props.overlay)
    // ถ้า stream URL ไม่สามารถสร้างได้ ให้ fallback ไปใช้ URL เดิม
    return streamUrl || props.url
  }

  if (isLocalPath(props.url)) {
    showLocalPathWarning.value = true
    const suggested = getSuggestedServerUrl(props.url)
    // console.warn(`Local file path detected: ${props.url}. Suggested server URL: ${suggested}`)
    return convertToServerUrl(props.url)
  }

  showLocalPathWarning.value = false
  return props.url
})

// Import stream service
import streamService from '@/services/streamService'

// Helper function to get MJPEG stream URL
function getMjpegStreamUrl(cameraId, overlay) {
  try {
    return streamService.getCameraStreamUrl(cameraId, overlay)
  } catch (error) {
    console.error(`Failed to get MJPEG stream URL for camera ${cameraId}:`, error)
    return ''
  }
}

// Function to check if URL is MJPEG stream
function isMjpegStream(url) {
  if (!url) return false
  return props.useStreamApi && props.cameraId ||
         url.includes('/stream/camera/') ||
         url.includes('multipart/x-mixed-replace') ||
         url.includes('/video') && !isVideoFile(url) // ถือว่า /video endpoint เป็น stream
}

// Function to check if URL is HTTP stream
function isHttpStream(url) {
  if (!url) return false
  const urlLower = url.toLowerCase()
  return (urlLower.startsWith('http://') || urlLower.startsWith('https://')) &&
         (urlLower.includes('/video') || urlLower.includes('/stream')) &&
         !isVideoFile(url)
}

// Reset loading state function
function resetLoadingState() {
  isLoading.value = false
  hasError.value = false
  errorMessage.value = ''
}

watch(() => props.url, (newUrl, oldUrl) => {
  // รีเซ็ต state เมื่อเปลี่ยน URL
  resetLoadingState()

  if (newUrl && newUrl !== oldUrl) {
    // ตรวจสอบประเภทของ URL และเริ่ม loading ใหม่
    if (isMjpegStream(newUrl) || isHttpStream(newUrl)) {
      handleLoadStart('mjpeg')
    } else if (isVideoFile(newUrl) || isStreamingUrl(newUrl)) {
      handleLoadStart('video')
    } else if (isImageUrl(newUrl)) {
      handleLoadStart('image')
    }
  }
})

watch(() => props.cameraId, (newCameraId, oldCameraId) => {
  if (newCameraId && newCameraId !== oldCameraId && props.useStreamApi) {
    resetLoadingState()
    handleLoadStart('mjpeg')
  }
})

// Placeholder images for different states
const placeholderImage = v89Logo

// Fallback image for loading state - using V89 logo with gradient overlay
const loadingFallbackImage = `data:image/svg+xml,%3Csvg width='320' height='180' xmlns='http://www.w3.org/2000/svg'%3E%3CdRect width='100%25' height='100%25' fill='%23111827'/%3E%3Cdefs%3E%3ClinearGradient id='grad' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%233b82f6;stop-opacity:0.1'/%3E%3Cstop offset='100%25' style='stop-color:%231e40af;stop-opacity:0.2'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='100%25' height='100%25' fill='url(%23grad)'/%3E%3Ctext x='50%25' y='140' font-family='Arial' font-size='13' fill='%239ca3af' text-anchor='middle'%3ELoading camera feed..%3C/text%3E%3C/svg%3E`

// Fallback image for error state - using V89 logo with error overlay
const errorFallbackImage = `data:image/svg+xml,%3Csvg width='320' height='180' xmlns='http://www.w3.org/2000/svg'%3E%3CdRect width='100%25' height='100%25' fill='%237f1d1d'/%3E%3Cdefs%3E%3ClinearGradient id='grad2' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%23ef4444;stop-opacity:0.15'/%3E%3Cstop offset='100%25' style='stop-color:%23dc2626;stop-opacity:0.25'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='100%25' height='100%25' fill='url(%23grad2)'/%3E%3Ccircle cx='160' cy='70' r='40' fill='none' stroke='%23fca5a5' stroke-width='2'/%3E%3Ctext x='160' y='85' font-family='Arial' font-size='36' fill='%23ef4444' text-anchor='middle' font-weight='bold' dominant-baseline='middle'%3E!%3C/text%3E%3Ctext x='50%25' y='140' font-family='Arial' font-size='13' fill='%23fca5a5' text-anchor='middle'%3ECannot load camera feed%3C/text%3E%3C/svg%3E`

function isImageUrl(url) {
  if (!url) return false
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
  const urlLower = url.toLowerCase()
  return imageExtensions.some(ext => urlLower.includes(ext)) ||
         urlLower.includes('snapshot') ||
         urlLower.includes('image')
}

function isVideoFile(url) {
  if (!url) return false
  const videoExtensions = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v']
  const urlLower = url.toLowerCase()
  return videoExtensions.some(ext => urlLower.endsWith(ext))
}

function isStreamingUrl(url) {
  if (!url) return false
  const urlLower = url.toLowerCase()
  return urlLower.startsWith('rtsp://') ||
         urlLower.startsWith('rtmp://') ||
         urlLower.startsWith('http://') && urlLower.includes('.m3u8') ||
         urlLower.startsWith('https://') && urlLower.includes('.m3u8')
}

function handleLoadStart(mediaType) {
  isLoading.value = true
  hasError.value = false
  errorMessage.value = ''
  emit('loadstart', { mediaType, url: processedUrl.value })
}

function handleLoad(mediaType) {
  isLoading.value = false
  hasError.value = false
  errorMessage.value = ''
  emit('load', { mediaType, url: processedUrl.value })
}

function handleError(mediaType) {
  isLoading.value = false
  hasError.value = true

  if (isLocalPath(props.url)) {
    errorMessage.value = `ไม่สามารถโหลดไฟล์ local ได้ กรุณาใช้ server URL แทน`
  } else if (mediaType === 'mjpeg') {
    errorMessage.value = `ไม่สามารถเชื่อมต่อ stream ได้`
  } else {
    errorMessage.value = `ไม่สามารถโหลด${mediaType === 'video' ? 'วิดีโอ' : 'รูปภาพ'}ได้`
  }

  emit('error', {
    mediaType,
    url: processedUrl.value,
    originalUrl: props.url,
    message: errorMessage.value
  })
}

function handleMetadataLoaded() {
  emit('metadata-loaded', {
    url: processedUrl.value,
    videoElement: videoElement.value
  })
}

function handleCanPlay() {
  emit('canplay', {
    url: processedUrl.value,
    videoElement: videoElement.value
  })
}

function play() {
  if (videoElement.value) {
    return videoElement.value.play()
  }
}

function pause() {
  if (videoElement.value) {
    videoElement.value.pause()
  }
}

function getCurrentTime() {
  return videoElement.value?.currentTime || 0
}

function setCurrentTime(time) {
  if (videoElement.value) {
    videoElement.value.currentTime = time
  }
}

defineExpose({
  play,
  pause,
  getCurrentTime,
  setCurrentTime,
  videoElement,
  mjpegElement
})

onMounted(() => {
  if (processedUrl.value) {
    if (isMjpegStream(processedUrl.value) || isHttpStream(processedUrl.value)) {
      handleLoadStart('mjpeg')
    } else if (isVideoFile(processedUrl.value) || isStreamingUrl(processedUrl.value)) {
      handleLoadStart('video')
    } else if (isImageUrl(processedUrl.value)) {
      handleLoadStart('image')
    }
  }
})
</script>

<style scoped>
.media-viewer {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background-color: #000;
}

.media-content {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  transition: filter 0.3s ease;
}

.blurred-content {
  filter: blur(20px);
}

.mjpeg-stream {
  /* สำหรับ MJPEG stream ไม่ต้องใช้ smooth transition */
  image-rendering: auto;
  image-rendering: -webkit-optimize-contrast;
}

.media-viewer.fullscreen .media-content {
  object-fit: contain;
}

.stream-indicator {
  position: absolute;
  top: 12px;
  right: 12px;
  z-index: 20;
}

.live-badge {
  background: #ef4444;
  color: white;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  animation: pulse 2s infinite;
}

.media-content video::-webkit-media-controls {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(0, 0, 0, 0.8);
  z-index: 5;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 10;
  gap: 8px;
}

.loading-overlay .fallback-bg {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  z-index: -1;
}

.loading-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: white;
  font-size: 0.875rem;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
  background: rgba(0, 0, 0, 0.5);
  padding: 12px 24px;
  border-radius: 8px;
}

.loading-spinner {
  width: 24px;
  height: 24px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top: 2px solid #fff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.error-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  z-index: 10;
  gap: 8px;
}

.error-overlay .fallback-bg {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.5;
  z-index: -1;
}

.error-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: white;
  font-size: 0.875rem;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
  background: rgba(220, 38, 38, 0.8);
  padding: 16px 24px;
  border-radius: 8px;
  border: 2px solid rgba(255, 255, 255, 0.2);
}

.error-icon {
  font-size: 2rem;
  line-height: 1;
}

.error-suggestion {
  margin-top: 4px;
  opacity: 0.9;
  font-size: 0.75rem;
}

.local-path-warning {
  position: absolute;
  top: 8px;
  left: 8px;
  right: 8px;
  background: rgba(251, 191, 36, 0.95);
  color: #92400e;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 0.75rem;
  z-index: 15;
  display: flex;
  align-items: flex-start;
  gap: 8px;
  border: 1px solid #f59e0b;
}

.warning-icon {
  flex-shrink: 0;
  font-size: 1rem;
}

.warning-text {
  flex: 1;
  line-height: 1.3;
}

.warning-text code {
  background: rgba(0, 0, 0, 0.1);
  padding: 1px 4px;
  border-radius: 3px;
  font-family: monospace;
  font-size: 0.7rem;
}

.fallback-bg {
  position: absolute !important;
  top: 0 !important;
  left: 0 !important;
  width: 100% !important;
  height: 100% !important;
  object-fit: cover !important;
  z-index: -1 !important;
}

.logo-container {
  position: absolute;
  top: 40%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 11;
  opacity: 0.3;
  pointer-events: none;
}

.v89-logo {
  width: 120px;
  height: 120px;
  object-fit: contain;
  filter: brightness(1.2);
}

.error-logo-container {
  opacity: 0.25;
}

.error-logo {
  filter: brightness(0.9) saturate(0.5);
}
</style>
