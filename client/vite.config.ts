import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/conversations': 'http://localhost:8000',
      '/documents': 'http://localhost:8000',
    }
  }
})
