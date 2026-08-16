import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The API key lives on the server, so the browser always talks to our own origin
// and never to Google. In production uvicorn serves this build from web/dist and
// the paths line up; in development the dev server has to forward them.
export default defineConfig({
  plugins: [react()],
  // Kept in step with the `paths` entry in tsconfig.app.json.
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  build: {
    outDir: 'dist',
  },
  server: {
    host: true,
    // A bind-mounted Windows directory delivers no file events into a Linux
    // container, so the watcher has to look for itself. Compose sets this; a
    // dev server run on the host doesn't, and shouldn't pay for the polling.
    watch: process.env.VITE_POLL ? { usePolling: true, interval: 300 } : undefined,
    proxy: {
      '/api': {
        // host.docker.internal when this dev server itself runs in a container.
        target: process.env.API_TARGET ?? 'http://localhost:7860',
        changeOrigin: true,
      },
    },
  },
});
