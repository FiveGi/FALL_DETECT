import { copyFile, mkdir, readdir, stat } from 'fs/promises'
import { join, extname } from 'path'
import { existsSync } from 'fs'

const SOURCE_VIDEO_DIR = 'C:/Users/TUF_12500H/Desktop/V89/OF_Pose_Fall_Detection/videos'
const TARGET_VIDEO_DIR = './public/videos'
const VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v', '.3gp']

async function copyVideos() {
  try {
    console.log('🎬 เริ่มต้นการคัดลอกไฟล์วิดีโอ...')

    if (!existsSync(TARGET_VIDEO_DIR)) {
      await mkdir(TARGET_VIDEO_DIR, { recursive: true })
      console.log(`📁 สร้างโฟลเดอร์: ${TARGET_VIDEO_DIR}`)
    }

    if (!existsSync(SOURCE_VIDEO_DIR)) {
      console.error(`❌ ไม่พบโฟลเดอร์ต้นทาง: ${SOURCE_VIDEO_DIR}`)
      return
    }

    const files = await readdir(SOURCE_VIDEO_DIR)
    const videoFiles = files.filter(file => {
      const ext = extname(file).toLowerCase()
      return VIDEO_EXTENSIONS.includes(ext)
    })

    if (videoFiles.length === 0) {
      console.log('📝 ไม่พบไฟล์วิดีโอในโฟลเดอร์ต้นทาง')
      return
    }

    console.log(`📊 พบไฟล์วิดีโอ ${videoFiles.length} ไฟล์`)

    let copiedCount = 0
    let skippedCount = 0
    let errorCount = 0

    for (const file of videoFiles) {
      const sourcePath = join(SOURCE_VIDEO_DIR, file)
      const targetPath = join(TARGET_VIDEO_DIR, file)

      try {
        if (existsSync(targetPath)) {
          const sourceStat = await stat(sourcePath)
          const targetStat = await stat(targetPath)

          if (sourceStat.size === targetStat.size) {
            console.log(`⏭️  ข้าม: ${file} (มีอยู่แล้วและขนาดเท่ากัน)`)
            skippedCount++
            continue
          }
        }

        await copyFile(sourcePath, targetPath)
        console.log(`✅ คัดลอก: ${file}`)
        copiedCount++

      } catch (error) {
        console.error(`❌ ข้อผิดพลาดขณะคัดลอก ${file}:`, error.message)
        errorCount++
      }
    }

    console.log('\n📈 สรุปผลการคัดลอก:')
    console.log(`   ✅ คัดลอกสำเร็จ: ${copiedCount} ไฟล์`)
    console.log(`   ⏭️  ข้าม: ${skippedCount} ไฟล์`)
    console.log(`   ❌ ข้อผิดพลาด: ${errorCount} ไฟล์`)

    if (copiedCount > 0) {
      console.log('\n🎉 การคัดลอกเสร็จสิ้น!')
      console.log('💡 ตอนนี้คุณสามารถใช้ URL ในรูปแบบ:')
      console.log('   http://localhost:3001/videos/[ชื่อไฟล์]')
      console.log('\n📋 ตัวอย่าง URLs:')

      videoFiles.slice(0, 3).forEach(file => {
        console.log(`   http://localhost:3001/videos/${file}`)
      })

      if (videoFiles.length > 3) {
        console.log(`   ... และอีก ${videoFiles.length - 3} ไฟล์`)
      }
    }

  } catch (error) {
    console.error('💥 เกิดข้อผิดพลาดในการคัดลอก:', error.message)
  }
}

copyVideos().catch(console.error)
