import { test } from "@playwright/test";

// A scripted ~20 s tour of the live demo, recorded to video and turned into docs/demo.gif.
const GRAPH = "svg[aria-label^='Reasoning path']";
const pause = (page: import("@playwright/test").Page, ms: number) => page.waitForTimeout(ms);

test("tour", async ({ page }) => {
  await page.goto("./");
  await page.getByText("Real GPT-2", { exact: true }).waitFor();
  await pause(page, 2200);
  const nodes = page.locator(`${GRAPH} g[role='button'] circle:nth-child(2)`);
  for (const i of [5, 9, 2]) {
    const box = await nodes.nth(i).boundingBox();
    if (box) await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 12 });
    await pause(page, 700);
  }
  await page.mouse.move(10, 10);
  await page.getByRole("button", { name: "2×" }).click();
  await pause(page, 1800);
  await page.getByRole("button", { name: "Ablate" }).click();
  await pause(page, 1500);
  await page.getByText("Trust in the graph's prediction").scrollIntoViewIfNeeded();
  await pause(page, 2200);
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await pause(page, 800);
  await page.getByRole("button", { name: "Indirect object (IOI)" }).click();
  await page.getByText("Real GPT-2", { exact: true }).waitFor();
  await pause(page, 2500);
  await page.getByRole("button", { name: "Fidelity study" }).click();
  await pause(page, 1800);
  for (let y = 0; y < 5; y++) {
    await page.mouse.wheel(0, 260);
    await pause(page, 650);
  }
  await pause(page, 1200);
});
