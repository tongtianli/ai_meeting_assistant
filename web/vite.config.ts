import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// 开发时 /api 代理到后端，避免 CORS；生产由 FastAPI 托管 dist（同源）
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
