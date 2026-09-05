import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [vue()],
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url))
        }
    },
    server: {
        proxy: {
            '/api': {
                target: 'http://backend:8932',
                changeOrigin: true,
                rewrite: (path) => path,
            },
        },
        host: '0.0.0.0',
        port: 3000,
        // Lets this dev server be reached through a tunnel (e.g. cloudflared's
        // `tunnel --url`) for sharing with someone off this network -- Vite
        // otherwise rejects any request whose Host header isn't localhost, to
        // guard against DNS-rebinding attacks. Scoped to trycloudflare.com's
        // subdomains rather than allowing every host.
        allowedHosts: ['.trycloudflare.com'],
        fs: {
            // อนุญาตให้เข้าถึงไฟล์นอก project root
            strict: false,
            allow: ['..']
        }
    },
    build: {
        target: 'esnext'
    },
    // เพิ่ม static assets สำหรับ videos
    publicDir: 'public'
})
