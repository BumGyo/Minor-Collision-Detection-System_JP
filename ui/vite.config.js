import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // WSL2などからWindowsブラウザがアクセスできるように0.0.0.0をバインド
    host: true,
    // VSCodeポートフォワーディング(dev tunnel)・外部共有時のトンネルドメインを許可
    // (Viteが不明なHostヘッダー要求を遮断するのを防止)
    allowedHosts: ['.devtunnels.ms'],
    proxy: {
      // バックエンドFastAPIへのプロキシ (開発サーバー)
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
