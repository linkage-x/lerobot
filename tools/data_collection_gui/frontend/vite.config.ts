import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const gatewayTarget = process.env.GUI_API_TARGET ?? "http://192.168.111.122:8765";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: gatewayTarget,
        changeOrigin: true
      }
    }
  }
});
