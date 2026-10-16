import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e-tour",
  timeout: 120_000,
  outputDir: "./tour-output",
  use: {
    baseURL: process.env.TWINLENS_URL ?? "https://spiliosdimakopoulos.github.io/twinlens/",
    viewport: { width: 1280, height: 800 },
    video: { mode: "on", size: { width: 1280, height: 800 } },
  },
  reporter: "list",
});
