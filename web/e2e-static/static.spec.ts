import { expect, test } from "@playwright/test";

// Runs against the static GitHub Pages build (no Python server): `npm run build:static`,
// then serve web/dist-static under any path.
const GRAPH = "svg[aria-label^='Reasoning path']";
const GAP = "svg[aria-label^='Predicted change at each twin level']";

test("static demo: every specimen traces, what-ifs and presets work offline", async ({ page }) => {
  const failed: string[] = [];
  page.on("response", (r) => { if (r.status() >= 400) failed.push(r.url()); });
  await page.goto("./");
  await expect(page.getByText(/Static demo with precomputed GPT-2 runs/)).toBeVisible();
  await expect(page.locator(GAP)).toBeVisible();
  await expect(page.getByText("PROPOSED FIX · STILL LINEAR")).toBeVisible();
  const tabs = page.getByRole("tab");
  for (let i = 1; i < (await tabs.count()); i++) {
    await tabs.nth(i).click();
    await expect(tabs.nth(i)).toHaveAttribute("aria-selected", "true");
    await expect(page.locator(GRAPH)).toBeVisible();
    await expect(page.locator(GAP)).toBeVisible();
  }
  for (const preset of ["½×", "2×", "3×", "Ablate"]) {
    await page.getByRole("button", { name: preset }).click();
    await expect(page.getByText("simulating…")).toBeHidden();
    await expect(page.locator(GAP)).toBeVisible();
  }
  await page.locator(`${GRAPH} g[role='button']`).nth(3).click({ force: true });
  await expect(page.getByText(/raises it|lowers it/).first()).toBeVisible();
  await page.getByRole("button", { name: "Generate" }).click();
  await expect(page.getByRole("button", { name: "Stop" })).toBeHidden({ timeout: 15000 });
  await page.screenshot({ path: "../docs/screenshots/static-demo.png", fullPage: true });
  expect(failed).toEqual([]);
});

test("static demo: deep link opens the IOI hero example", async ({ page }) => {
  await page.goto("./?p=2&f=0-3-5545");
  await expect(page.locator(GAP)).toBeVisible();
  await expect(page.getByText(/the opposite direction/)).toBeVisible();
  await expect(page.getByRole("tab", { selected: true })).toContainText("Indirect object");
});

test("static demo: internals, study and reproductions load from precomputed data", async ({ page }) => {
  await page.goto("./?view=internals");
  await expect(page.locator("svg[aria-label='Logit lens']")).toBeVisible();
  await page.locator("svg[aria-label='Direct logit attribution per attention head'] g").nth(30).click();
  await expect(page.getByText(/After ablating L/)).toBeVisible();
  await page.getByRole("button", { name: "The study" }).click();
  await expect(page.getByRole("heading", { name: "Keep the derivatives, keep the linearity" })).toBeVisible();
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
