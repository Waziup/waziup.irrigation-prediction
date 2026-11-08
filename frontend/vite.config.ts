import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';
import type { ProxyOptions } from 'vite';

export default defineConfig(({ mode }) => {
  const root = fileURLToPath(new URL('.', import.meta.url));
  const env = loadEnv(mode, root, '');
  const socketPath = resolve(root, env.WAZIFARM_API_SOCKET || '../proxy.sock');
  const backend: ProxyOptions = {
    target: { protocol: 'http:', host: 'localhost', port: 80, socketPath },
    // Preserve the browser Host so existing same-origin write checks still apply.
    changeOrigin: false,
    configure(proxy) {
      proxy.on('error', (_error, _request, response) => {
        if ('writeHead' in response && !response.headersSent) {
          response.writeHead(503, { 'Content-Type': 'application/json' });
          response.end(
            JSON.stringify({
              error:
                'Backend unavailable. Start the Python application, or set WAZIFARM_API_SOCKET to its Unix socket and restart Vite.',
            }),
          );
        }
      });
    },
  };
  return {
    base: './',
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      strictPort: true,
      proxy: { '/api': backend, '/legacy-ui': backend },
    },
    preview: { port: 4173, strictPort: true },
  };
});
