import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e-static",
  timeout: 120_000,
  expect: { timeout: 30_000 },
  use: { baseURL: process.env.TWINLENS_URL ?? "http://localhost:8001/twinlens/", viewport: { width: 1440, height: 1000 } },
  reporter: "list",
});
