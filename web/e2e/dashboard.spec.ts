import { expect, test, type Page } from "@playwright/test";

const SHOTS = "../docs/screenshots";
const GRAPH = "svg[aria-label^='Reasoning path']";
const GAP = "svg[aria-label^='Predicted change at each twin level']";

async function ready(page: Page) {
  await expect(page.locator(GRAPH)).toBeVisible();
  await expect(page.locator(GAP)).toBeVisible();
  await expect(page.getByText("simulating…")).toBeHidden();
}

test("loads with a traced prompt and an automatic what-if", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("derivatives they drop");
  await expect(page.getByText(/The graph predicts|Neither the graph/)).toBeVisible();
  await expect(page.getByText("Should you trust the graph here?")).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/twin.png`, fullPage: true });
});

test("clicking another node and a preset re-simulates; the URL is shareable", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  const nodes = page.locator(`${GRAPH} g[role='button']`);
  const n = await nodes.count();
  expect(n).toBeGreaterThan(5);
  await nodes.nth(n - 1).click({ force: true });
  await page.getByRole("button", { name: "2×" }).click();
  await expect(page.getByRole("button", { name: "2×" })).toHaveAttribute("aria-pressed", "true");
  await ready(page);
  await expect(page.getByText(/raises it|lowers it/).first()).toBeVisible();
  await expect(page).toHaveURL(/[?&]f=\d+-\d+-\d+/);
  await expect(page).toHaveURL(/[?&]m=2/);
  await page.screenshot({ path: `${SHOTS}/whatif.png`, fullPage: true });
  // Re-opening the shared link restores the same feature and edit.
  await page.goto(page.url());
  await ready(page);
  await expect(page.getByRole("button", { name: "2×" })).toHaveAttribute("aria-pressed", "true");
});

test("keyboard: nodes are focusable, arrows step features, brackets change specimen", async ({ page }) => {
  await page.goto("/");
  await ready(page);
  const node = page.locator(`${GRAPH} g[role='button']`).first();
  await node.focus();
  await page.keyboard.press("Enter");
  await expect(node).toHaveAttribute("aria-pressed", "true");
  await page.locator("main").click({ position: { x: 5, y: 5 } });
  await page.keyboard.press("ArrowRight");
  await expect(node).toHaveAttribute("aria-pressed", "false");
  await page.keyboard.press("]");
  await expect(page.getByRole("tab", { selected: true })).toContainText("Indirect object");
});

test("internals: logit lens, attention, click-to-ablate", async ({ page }) => {
  await page.goto("/?view=internals");
  await expect(page.locator("svg[aria-label='Logit lens']")).toBeVisible();
  await page.locator("svg[aria-label='Direct logit attribution per attention head'] g").nth(30).click();
  await expect(page.getByText(/After ablating L/)).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/internals.png`, fullPage: true });
});

test("the study and reproductions render", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "The study" }).click();
  await expect(page.getByRole("heading", { name: "Keep the derivatives, keep the linearity" })).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/study.png`, fullPage: true });
  await page.getByRole("button", { name: "Reproductions" }).click();
  await expect(page.locator("svg[aria-label='Induction score per head']")).toBeVisible();
  await expect(page.locator("svg[aria-label='IOI direct logit attribution per head']")).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/reproductions.png`, fullPage: true });
});

test("dark mode via the theme toggle", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("radio", { name: "Dark theme" }).click();
  await ready(page);
  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(bg).toBe("rgb(15, 15, 14)");
  await page.screenshot({ path: `${SHOTS}/twin-dark.png`, fullPage: true });
  await page.getByRole("radio", { name: "System theme" }).click();
});

test("phone layout: no horizontal page scroll, what-if follows the graph", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await ready(page);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  const gy = (await page.locator(GRAPH).boundingBox())!.y;
  const wy = (await page.locator(GAP).boundingBox())!.y;
  const py = (await page.getByText("What GPT-2 predicts").boundingBox())!.y;
  expect(gy).toBeLessThan(wy);
  expect(wy).toBeLessThan(py);
  await page.screenshot({ path: `${SHOTS}/mobile.png`, fullPage: true });
});
