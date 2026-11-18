import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Which gateway /api goes to. run/deploy.sh always sets it (production, sandbox or the workstation).
// Without it a hand-started dev server talks to a gateway on this computer, never to a production
// Thor: an unmerged page must not drive production's recorder and arms because a variable was forgotten.
const gatewayTarget = process.env.GUI_API_TARGET ?? "http://127.0.0.1:8765";
if (!process.env.VITEST) {
  console.log(`[gui] /api -> ${gatewayTarget}${process.env.GUI_API_TARGET ? "" : " (GUI_API_TARGET unset)"}`);
}

// `server.proxy` also serves `vite preview` (preview.proxy defaults to it), which is how production
// runs: a built dist/ that only changes when it is rebuilt (run/README.md).
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
