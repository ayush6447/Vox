import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite configuration for the SignSpeak frontend
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Allow backend on another port, e.g. 8000
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false
      }
    }
  }
});


