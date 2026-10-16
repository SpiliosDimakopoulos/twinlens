// Print the rendered report (web/public/report/index.html) to PDF with headless Chromium.
import { chromium } from "@playwright/test";
import { copyFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const html = resolve("public/report/index.html");
const out = resolve("public/report/TwinLens_report.pdf");
const browser = await chromium.launch();
const page = await browser.newPage();
await page.emulateMedia({ media: "print", colorScheme: "light" });
await page.goto(pathToFileURL(html).href, { waitUntil: "load" });
await page.pdf({
  path: out, format: "A4", printBackground: true,
  margin: { top: "16mm", bottom: "16mm", left: "14mm", right: "14mm" },
  displayHeaderFooter: true,
  headerTemplate: "<span></span>",
  footerTemplate: '<div style="font-size:8px;color:#898781;width:100%;text-align:center"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
});
await browser.close();
copyFileSync(out, resolve("../docs/TwinLens_report.pdf"));
console.log("wrote", out);
