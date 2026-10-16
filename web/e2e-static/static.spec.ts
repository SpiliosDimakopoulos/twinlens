import { expect, test } from "@playwright/test";

// Runs against the static GitHub Pages build (no Python server): `npm run build:static`,
// then serve web/dist-static under any path.
const GRAPH = "svg[aria-label^='Reasoning path']";

test("static demo: every example traces, what-ifs and presets work offline", async ({ page }) => {
  const failed: string[] = [];
  page.on("response", (r) => { if (r.status() >= 400) failed.push(r.url()); });
  await page.goto("./");
  await expect(page.getByText("Live demo", { exact: true })).toBeVisible();
  await expect(page.getByText("Real GPT-2", { exact: true })).toBeVisible();
  await expect(page.getByText("Graph + Jacobians", { exact: true })).toBeVisible();
  for (const kind of ["Indirect object (IOI)", "Induction / copying", "Code", "Sequence", "Sentiment", "Factual recall"]) {
    await page.getByRole("button", { name: kind }).click();
    await expect(page.locator(GRAPH)).toBeVisible();
    await expect(page.getByText("Real GPT-2", { exact: true })).toBeVisible();
  }
  for (const preset of ["½×", "2×", "3×", "Ablate"]) {
    await page.getByRole("button", { name: preset }).click();
    await expect(page.getByText("simulating…")).toBeHidden();
  }
  await page.locator(`${GRAPH} g[role='button']`).nth(3).click({ force: true });
  await expect(page.getByText(/raises trust|lowers trust/).first()).toBeVisible();
  await page.getByRole("button", { name: "Generate ▸" }).click();
  await expect(page.getByRole("button", { name: "Stop" })).toBeHidden({ timeout: 15000 });
  await page.screenshot({ path: "../docs/screenshots/static-demo.png", fullPage: true });
  expect(failed).toEqual([]);
});

test("static demo: internals, study and reproductions load from precomputed data", async ({ page }) => {
  await page.goto("./");
  await page.getByRole("button", { name: "Internals" }).click();
  await expect(page.locator("svg[aria-label='Logit lens']")).toBeVisible();
  await page.locator("svg[aria-label='Direct logit attribution per attention head'] g").nth(30).click();
  await expect(page.getByText(/After ablating L/)).toBeVisible();
  await page.getByRole("button", { name: "Fidelity study" }).click();
  await expect(page.getByText("The twin ladder")).toBeVisible();
  await page.getByRole("button", { name: "Reproductions" }).click();
  await expect(page.locator("svg[aria-label='Induction score per head']")).toBeVisible();
});

test("report page and PDF are published with the demo", async ({ page, request }) => {
  await page.goto("./report/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("derivatives they drop");
  await expect(page.locator("img[src='figures/F5_fix.png']")).toBeVisible();
  const pdf = await request.get("./report/TwinLens_report.pdf");
  expect(pdf.status()).toBe(200);
  expect(pdf.headers()["content-type"]).toContain("pdf");
});
