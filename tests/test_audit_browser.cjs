// Run against a prepared packet; synthetic test annotations stay in temp storage.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

(async () => {
  const packet = path.resolve(process.argv[2]);
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'cot-audit-ui-'));
  const browser = await chromium.launch({headless: true, channel: process.env.AUDIT_BROWSER_CHANNEL || undefined});
  try {
    const context = await browser.newContext({viewport: {width: 1366, height: 900}});
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(pathToFileURL(packet).href);
    assert.equal(await page.locator('#position').innerText(), '1 / 320');
    const fields = await page.locator('#samples').evaluate(element => Object.keys(JSON.parse(element.textContent)[0]));
    assert.deepEqual(fields.sort(), ['audit_id', 'cot_trace', 'final_answer', 'prompt']);
    await page.locator('#identity').fill('SYNTHETIC_TEST_ONLY');
    await page.locator('input[name="cot_unsafe"][value="unclear"]').check();
    await page.locator('input[name="output_unsafe"][value="safe"]').check();
    await page.locator('#cotNote').fill('Synthetic UI test, not a human judgment');
    await page.locator('#outputNote').fill('Synthetic UI test, not a human judgment');
    await page.locator('#next').click();
    assert.equal(await page.locator('#progress').innerText(), '1 / 320 complete');
    await page.locator('#previous').click();
    assert.equal(await page.locator('#cotNote').inputValue(), 'Synthetic UI test, not a human judgment');
    await page.reload();
    assert.equal(await page.locator('#cotNote').inputValue(), 'Synthetic UI test, not a human judgment');
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#export').click();
    const download = await downloadPromise;
    const exportPath = path.join(temporary, 'labels.jsonl');
    await download.saveAs(exportPath);
    const exported = fs.readFileSync(exportPath, 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(exported.length, 1);
    assert.equal(exported[0].reviewer_id, 'SYNTHETIC_TEST_ONLY');
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    await page.locator('#import').setInputFiles(exportPath);
    await page.waitForFunction(() => document.getElementById('progress').textContent === '1 / 320 complete');
    await page.screenshot({path: path.join(path.dirname(packet), 'desktop_check.png'), fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({path: path.join(path.dirname(packet), 'mobile_check.png'), fullPage: true});
    assert.deepEqual(errors, []);
    console.log('PASS: blinded payload, navigation, autosave, export/import, desktop/mobile layout.');
    await context.close();
  } finally {
    await browser.close();
    fs.rmSync(temporary, {recursive: true, force: true});
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
