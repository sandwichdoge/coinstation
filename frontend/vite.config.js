import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// In dev, proxy /api to the FastAPI backend so the browser makes same-origin
// requests (no CORS). Override the backend target with VITE_PROXY_TARGET.
var target = process.env.VITE_PROXY_TARGET || "http://127.0.0.1:8000";
export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        proxy: {
            "/api": { target: target, changeOrigin: true },
        },
    },
});
