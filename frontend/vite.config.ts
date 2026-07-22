import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies API and media calls to the FastAPI backend, so the
// frontend code can use same-origin relative URLs in both dev and production.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/media': 'http://127.0.0.1:8000',
    },
  },
})
