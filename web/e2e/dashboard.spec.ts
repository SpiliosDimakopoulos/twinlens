import { expect, test } from "@playwright/test";

const SHOTS = "../docs/screenshots";

test("trace a prompt, run a what-if, and capture the dashboard", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Trace" }).click();
  await expect(page.getByText("Reasoning path")).toBeVisible();
  await expect(page.locator("svg[aria-label^='Features with the largest direct effect'] circle").first()).toBeVisible();
  await page.screenshot({ path: `${SHOTS}/twin.png`, fullPage: true });

  // Click the biggest node in the graph (the one with the largest radius).
  const nodes = page.locator("svg[aria-label^='Features with the largest direct effect'] g[style*='pointer'] circle:nth-child(2)");
  const radii = await nodes.evaluateAll((els) => els.map((e) => Number(e.getAttribute("r"))));
  await nodes.nth(radii.indexOf(Math.max(...radii))).click({ force: true });
  await page.getByRole("button", { name: "Ask the twin, then GPT-2" }).click();
  await expect(page.getByText("Real GPT-2")).toBeVisible();
  await expect(page.getByText(/Twin trust score|Trust model not trained/)).toBeVisible();
  await page.getByText("What if: L").scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${SHOTS}/whatif.png`, fullPage: true });
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
