import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Relative base so the built bundle works wherever FastAPI mounts it, and so
// opening dist/index.html directly does not 404 on absolute /assets paths.
export default defineConfig({
  plugins: [react()],
  base: './',
  build: { outDir: 'dist', emptyOutDir: true, assetsInlineLimit: 0 },
  // Dev only. The shipped shape serves this bundle from the FastAPI app itself,
  // single origin, so there is no proxy and no CORS in the demo path at all.
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: false } },
  },
  test: { environment: 'jsdom', globals: true },
})
