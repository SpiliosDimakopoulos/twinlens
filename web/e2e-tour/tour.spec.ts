import { test, type Page } from "@playwright/test";

// A scripted ~25 s tour of the live demo, recorded to video and turned into docs/demo.gif.
const GRAPH = "svg[aria-label^='Reasoning path']";
const GAP = "svg[aria-label^='Predicted change at each twin level']";
const pause = (page: Page, ms: number) => page.waitForTimeout(ms);

test("tour", async ({ page }) => {
  await page.goto("./?p=2&f=0-3-5545");
  await page.locator(GAP).waitFor();
  await pause(page, 2600);
  await page.locator(GRAPH).scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollBy({ top: 120, behavior: "smooth" }));
  await pause(page, 1600);
  const nodes = page.locator(`${GRAPH} g[role='button'] circle:nth-child(2)`);
  for (const i of [4, 8, 12]) {
    const box = await nodes.nth(i).boundingBox();
    if (box) await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 14 });
    await pause(page, 800);
  }
  await page.mouse.move(20, 20);
  await page.getByRole("button", { name: "2×" }).click();
  await pause(page, 1800);
  await page.getByRole("button", { name: "Ablate" }).click();
  await pause(page, 2200);
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await pause(page, 900);
  await page.getByRole("tab").first().click();
  await page.locator(GAP).waitFor();
  await pause(page, 1200);
  await page.locator(GRAPH).scrollIntoViewIfNeeded();
  await pause(page, 2200);
  await page.getByRole("button", { name: "The study" }).click();
  await pause(page, 1600);
  for (let y = 0; y < 7; y++) {
    await page.mouse.wheel(0, 300);
    await pause(page, 600);
  }
  await pause(page, 1400);
});
