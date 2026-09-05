import { getApiBaseUrl, API_ENDPOINTS } from '@/config/api'

function getAuthToken() {
  return localStorage.getItem('authToken')
}

export default {
  // ดึงการตั้งค่า Telegram
  fetchTelegramSettings: async () => {
    try {
      const token = getAuthToken()
      const response = await fetch(`${getApiBaseUrl()}${API_ENDPOINTS.TELEGRAM.SETTINGS}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
      })

      if (!response.ok) {
        throw new Error('Failed to fetch telegram settings')
      }

      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error fetching telegram settings:', error)
      throw error
    }
  },

  // อัปเดตการตั้งค่า Telegram
  updateTelegramSettings: async (settings) => {
    try {
      const token = getAuthToken()
      const response = await fetch(`${getApiBaseUrl()}${API_ENDPOINTS.TELEGRAM.SETTINGS}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(settings),
      })

      if (!response.ok) {
        throw new Error('Failed to update telegram settings')
      }

      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error updating telegram settings:', error)
      throw error
    }
  },

  // ทดสอบการส่งข้อความ Telegram
  testTelegramSettings: async () => {
    try {
      const token = getAuthToken()
      const response = await fetch(`${getApiBaseUrl()}${API_ENDPOINTS.TELEGRAM.TEST}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
      })

      if (!response.ok) {
        throw new Error('Failed to test telegram settings')
      }

      const data = await response.json()
      return data
    } catch (error) {
      console.error('Error testing telegram settings:', error)
      throw error
    }
  },
}
