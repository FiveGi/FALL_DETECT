import { getApiBaseUrl, API_ENDPOINTS } from '@/config/api'

function getAuthToken() {
    return localStorage.getItem('authToken')
}

export default {
  fetchAllLogs: async () => {
    try {
      const token = getAuthToken()
      const response = await fetch(`${getApiBaseUrl()}${API_ENDPOINTS.LOG.BASE}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
      })
      if (!response.ok) {
        throw new Error('Failed to fetch notification')
      }
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching notification:', error)
      throw error
    }
  },
  // ดึง log ตาม id กล้อง
  fetchLogsByCameraId: async (cameraId) => {
    try {
      const token = getAuthToken()
      const response = await fetch(`${getApiBaseUrl()}${API_ENDPOINTS.LOG.DETAIL(cameraId)}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
      })
      if (!response.ok) {
        throw new Error('Failed to fetch logs for camera')
      }
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching logs for camera:', error)
      throw error
    }
  },

  fetchAllNotification: async () => {
    try {
      const token = getAuthToken()
      const response = await fetch(`${getApiBaseUrl()}${API_ENDPOINTS.NOTIFICATIONS.BASE}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
      })
      if (!response.ok) {
        throw new Error('Failed to fetch notification')
      }
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching notification:', error)
      throw error
    }
  },

  fetchNotificationByCameraId: async (cameraId) => {
    try {
      const token = getAuthToken()
      const response = await fetch(`${getApiBaseUrl()}${API_ENDPOINTS.NOTIFICATIONS.DETAIL(cameraId)}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
      })
      if (!response.ok) {
        throw new Error('Failed to fetch notification')
      }
      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching notification:', error)
      throw error
    }
  },
}
