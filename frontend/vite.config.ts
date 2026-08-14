import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

function readPortConfig(): Record<string, string> {
  const configPath = resolve(process.cwd(), "../config/ports.env");
  if (!existsSync(configPath)) {
    return {};
  }

  return readFileSync(configPath, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("#"))
    .reduce<Record<string, string>>((config, line) => {
      const index = line.indexOf("=");
      if (index > 0) {
        config[line.slice(0, index).trim()] = line.slice(index + 1).trim();
      }
      return config;
    }, {});
}

const portConfig = readPortConfig();
const frontendPort = Number(process.env.PORT ?? portConfig.FRONTEND_PORT ?? 5173);
const backendPort = Number(process.env.BACKEND_PORT ?? portConfig.BACKEND_PORT ?? 8001);
const backendHost = process.env.API_PROXY_HOST ?? portConfig.API_PROXY_HOST ?? "127.0.0.1";
const apiProxyTarget = process.env.API_PROXY_TARGET ?? `http://${backendHost}:${backendPort}`;

export default defineConfig({
  plugins: [react()],
  server: {
    host: portConfig.FRONTEND_HOST ?? "0.0.0.0",
    port: frontendPort,
    proxy: {
      "/api": {
        target: apiProxyTarget,
        changeOrigin: true
      }
    }
  }
});
