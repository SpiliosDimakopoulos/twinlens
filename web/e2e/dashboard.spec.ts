import { expect, test } from "@playwright/test";

const SHOTS = "../docs/screenshots";
const GRAPH = "svg[aria-label^='Reasoning path']";

test("loads with a traced prompt and an automatic what-if", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(GRAPH)).toBeVisible();
  // The strongest supporting feature is pre-selected and simulated without any click.
  await expect(page.getByText("Real GPT-2", { exact: true })).toBeVisible();
  await expect(page.getByText(/Trust in the graph's prediction/)).toBeVisible();
  await expect(page.getByText(/The graph says the edit|Neither the graph/)).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/twin.png`, fullPage: true });
});

test("clicking another node and a preset re-simulates", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Real GPT-2", { exact: true })).toBeVisible();
  const nodes = page.locator(`${GRAPH} g[role='button']`);
  const n = await nodes.count();
  expect(n).toBeGreaterThan(5);
  await nodes.nth(n - 1).click({ force: true });
  await page.getByRole("button", { name: "2×" }).click();
  await expect(page.getByRole("button", { name: "2×" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("simulating…")).toBeHidden();
  await expect(page.getByText(/raises trust|lowers trust/).first()).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/whatif.png`, fullPage: true });
});

test("nodes are keyboard accessible", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Real GPT-2", { exact: true })).toBeVisible();
  const node = page.locator(`${GRAPH} g[role='button']`).first();
  await node.focus();
  await page.keyboard.press("Enter");
  await expect(node).toHaveAttribute("aria-pressed", "true");
});

test("internals: logit lens, attention, click-to-ablate", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Internals" }).click();
  await expect(page.locator("svg[aria-label='Logit lens']")).toBeVisible();
  await page.locator("svg[aria-label='Direct logit attribution per attention head'] g").nth(30).click();
  await expect(page.getByText(/After ablating L/)).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/internals.png`, fullPage: true });
});

test("fidelity study and reproductions render", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Fidelity study" }).click();
  await expect(page.getByText("The twin ladder")).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/study.png`, fullPage: true });
  await page.getByRole("button", { name: "Reproductions" }).click();
  await expect(page.locator("svg[aria-label='Induction score per head']")).toBeVisible();
  await expect(page.locator("svg[aria-label='IOI direct logit attribution per head']")).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/reproductions.png`, fullPage: true });
});

test("dark mode via the theme toggle", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("radio", { name: "Dark theme" }).click();
  await expect(page.getByText("Real GPT-2", { exact: true })).toBeVisible();
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(bg).toBe("rgb(13, 13, 13)");
  await page.screenshot({ path: `${SHOTS}/twin-dark.png`, fullPage: true });
  await page.getByRole("radio", { name: "System theme" }).click();
});

test("phone layout has no horizontal page scroll", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByText("Real GPT-2", { exact: true })).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await page.screenshot({ path: `${SHOTS}/mobile.png`, fullPage: true });
});
