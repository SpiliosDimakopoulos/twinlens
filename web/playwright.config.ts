import { defineConfig } from "@playwright/test";

// Expects the API (with the built dashboard) on :8000, e.g. `make serve`.
export default defineConfig({
  testDir: "./e2e",
  timeout: 180_000,
  expect: { timeout: 120_000 },
  use: { baseURL: process.env.TWINLENS_URL ?? "http://localhost:8000", viewport: { width: 1440, height: 1000 } },
  reporter: "list",
});
