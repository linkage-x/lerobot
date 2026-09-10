// PLAYWRIGHT_MODULE may point at an existing local Playwright installation.
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_BIN||'/usr/bin/google-chrome'});
try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto(process.env.SIM_URL||'http://127.0.0.1:8000/docs/rig0818_vs_hybrid_carrier_v1/simulator/');
  await page.waitForFunction(()=>document.querySelector('[data-metric="R0-p95"]').textContent!=='—',{timeout:30000});
  await page.locator('#play-button').click();
  await page.locator('[data-target="H1"]').click();
  await page.locator('#pose-scrubber').fill('20');
  assert.match(await page.locator('#scene-caption').innerText(),/pose 20/);
  await page.locator('#seed').fill('456');
  await page.locator('#reset-button').click();
  assert.equal(await page.locator('#seed').inputValue(),'20260910');
  await page.locator('#poses').fill('48');
  await page.locator('#run-button').click();
  await page.waitForFunction(()=>!document.querySelector('#run-button').disabled);
  const downloadPromise=page.waitForEvent('download');await page.locator('#export-csv').click();
  const download=await downloadPromise;assert.equal(download.suggestedFilename(),'trials.csv');
  await page.locator('#reload-l2-button').click();
  await page.waitForFunction(()=>document.querySelector('#l2-table').children.length>=3&&document.querySelector('#result-layer').value==='l2');
  assert.equal(await page.locator('#result-layer').inputValue(),'l2');
  assert.match(await page.locator('#scope-note').innerText(),/L2 图像解算/);
  assert.equal(await page.locator('.metric-card').evaluateAll(cards=>cards.every(card=>card.querySelector('footer').getBoundingClientRect().top>=card.querySelector('p').getBoundingClientRect().bottom)),true,'metric card footer overlaps error label');
  await page.screenshot({path:'/tmp/rig-ab-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:'/tmp/rig-ab-mobile.png',fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'mobile horizontal overflow');
  assert.deepEqual(errors,[]);
  console.log('Browser passed: module worker, rerun, tabs, scrubber, reset, CSV export, L2 load, mobile layout.');
}finally{await browser.close();}
