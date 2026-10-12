// Headless Chromium against gui_server.py --fake: node test/gui_e2e.mjs <url> [screenshot dir]
import { createRequire } from "module";
const require = createRequire(import.meta.url);
let pw;
try { pw = require("playwright"); } catch { pw = require("/opt/node22/lib/node_modules/playwright"); }
const url = process.argv[2];
const shots = process.argv[3] || "";
const results = [];
const check = (name, cond, detail = "") => { results.push(!!cond); console.log(`${cond ? "PASS" : "FAIL"} ${name}${detail ? ": " + detail : ""}`); };
const api = async (page, path) => page.evaluate(async (p) => (await fetch(p)).json(), path);

const browser = await pw.chromium.launch({ executablePath: process.env.PW_CHROMIUM || undefined });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error" && !/stream\.mjpg|tts/.test(m.text())) errors.push(m.text()); });
  await page.goto(url);
  await page.waitForSelector("#chip-conn.ok", { timeout: 5000 });
  check("page loads and connects", true);
  await page.waitForFunction(() => document.querySelectorAll("#dets li").length === 3, null, { timeout: 5000 });
  check("detections listed", true);

  // instruction -> proposal with countdown -> cancel
  await page.fill("#instruction", "grasp the red flower");
  await page.click("#say-form button[type=submit]");
  await page.waitForSelector("#proposal:not([hidden])", { timeout: 3000 });
  const title = await page.textContent("#proposal-title");
  check("proposal shown for the instruction", /Grasp #1 red flower/.test(title), title);
  await page.waitForTimeout(600);
  const scale = await page.$eval("#countdown-bar", (e) => e.style.transform);
  check("countdown running", /scaleX\(0\.[0-9]+\)/.test(scale), scale);
  await page.click("#cancel");
  await page.waitForSelector("#proposal", { state: "hidden", timeout: 3000 });
  check("cancel hides the proposal", true);

  // a proposal that is left alone commits after 2 s
  await page.fill("#instruction", "grasp the yellow flower");
  await page.press("#instruction", "Enter");
  await page.waitForSelector("#proposal:not([hidden])", { timeout: 3000 });
  await page.waitForSelector("#proposal", { state: "hidden", timeout: 4000 });
  const log = await page.textContent("#log");
  check("proposal commits after the countdown", /Grasping\./.test(log));

  // clarification question with options
  await page.fill("#instruction", "grasp the flower");
  await page.press("#instruction", "Enter");
  await page.waitForSelector("#question:not([hidden]) button", { timeout: 3000 });
  const opts = await page.$$eval("#question-opts button", (bs) => bs.map((b) => b.textContent));
  check("question offers the options", opts.length === 2, opts.join(" | "));
  await page.click("#question-opts button >> nth=1");
  await page.waitForSelector("#proposal:not([hidden])", { timeout: 3000 });
  check("answering the question gives a proposal", /yellow/.test(await page.textContent("#proposal-title")));
  await page.click("#confirm");
  await page.waitForSelector("#proposal", { state: "hidden", timeout: 3000 });

  // drive: refused until enabled, then hold-to-move and a zero on release
  await page.click('.tabs button[data-tab="drive"]');
  await page.check("#drive-enable");
  await page.waitForTimeout(300);
  const btn = page.locator('#base-pad button[aria-label="forward"]');
  const box = await btn.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.waitForTimeout(450);
  await page.mouse.up();
  await page.waitForTimeout(300);
  const st = await api(page, "api/debug_log");
  const drives = st.filter((e) => e.kind === "drive");
  check("hold-to-move sends repeated commands", drives.filter((d) => d.v > 0).length >= 3, `${drives.length} drive commands`);
  check("release sends a stop", drives.length && drives[drives.length - 1].v === 0);

  // stop: button and Escape; drive disabled afterwards
  await page.click("#stop");
  await page.waitForSelector("#stopped:not([hidden])", { timeout: 2000 });
  await page.keyboard.press("Escape");
  await page.waitForTimeout(300);
  const s2 = await api(page, "api/state");
  check("STOP button and Escape reach the robot", s2.stops >= 2, `${s2.stops} stops`);
  check("drive disabled after a stop", !(await page.isChecked("#drive-enable")) && s2.drive_enabled === false);

  // watch tab renders the uncertainty
  await page.click('.tabs button[data-tab="watch"]');
  const ug = await page.$eval("#u-geo", (m) => m.value);
  check("watch tab shows the uncertainty", Math.abs(ug - 0.6) < 1e-6, String(ug));
  if (shots) { await page.screenshot({ path: `${shots}/gui_desktop.png` }); }

  // phone width: no horizontal scroll, STOP visible
  const phone = await browser.newPage({ viewport: { width: 390, height: 800 } });
  await phone.goto(url);
  await phone.waitForSelector("#chip-conn.ok", { timeout: 5000 });
  const overflow = await phone.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  const stopBox = await phone.locator("#stop").boundingBox();
  check("phone width: no horizontal scroll, STOP on screen", overflow <= 0 && stopBox && stopBox.x + stopBox.width <= 390,
        `overflow ${overflow}px`);
  if (shots) { await phone.screenshot({ path: `${shots}/gui_phone.png` }); }
  check("no page errors", errors.length === 0, errors.slice(0, 3).join(" | "));
} finally {
  await browser.close();
}
const ok = results.every(Boolean);
console.log(ok ? "GUI E2E OK" : "GUI E2E FAILED");
process.exit(ok ? 0 : 1);
