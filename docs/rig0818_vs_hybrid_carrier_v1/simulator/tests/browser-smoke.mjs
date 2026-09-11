// PLAYWRIGHT_MODULE may point at an existing local Playwright or playwright-core installation.
import assert from 'node:assert/strict';
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_BIN||'/usr/bin/google-chrome',
  args:['--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
const shots=process.env.SHOT_DIR||'/tmp';
try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto(process.env.SIM_URL||'http://127.0.0.1:8000/docs/rig0818_vs_hybrid_carrier_v1/simulator/');

  // Real-input report: verdict, cards, one-axis plot, WebGL replay, counterexamples.
  await page.waitForFunction(()=>document.body.dataset.l3==='ready',{timeout:30000});
  assert.match(await page.locator('#decision-label').innerText(),/^(NO-GO|CONDITIONAL GO|INCONCLUSIVE)$/);
  assert.match(await page.locator('#l3-verdict').innerText(),/正式判定/);
  assert.equal(await page.locator('#l3-cards .metric-card').count(),3);
  await page.locator('#l3-level [data-level="stress"]').click();
  assert.match(await page.locator('#l3-scope').innerText(),/2× 压力/);
  await page.locator('#l3-output [data-output="raw"]').click();
  assert.match(await page.locator('#l3-scope').innerText(),/逐帧 BA/);
  await page.locator('#l3-level [data-level="evidence"]').click();
  await page.locator('#l3-output [data-output="smoothed"]').click();
  const ink=selector=>page.evaluate(sel=>{const c=document.querySelector(sel),d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=3;i<d.length;i+=4)if(d[i])n++;return n;},selector);
  assert.ok(await ink('#l3-forest')>2000,'forest plot is drawn');
  assert.ok(await ink('#l3-cdf')>1000,'CDF is drawn');
  assert.equal(await page.locator('#l3-mechanism tbody tr').count(),3,'rigid-body breakdown has one row per arm');
  assert.match(await page.locator('#l3-mechanism').innerText(),/09-09 真机/);
  await page.waitForFunction(()=>/^WebGL/.test(document.querySelector('#l3-scene-status').textContent),{timeout:30000});
  const colours=()=>page.evaluate(()=>{const c=document.querySelector('#l3-scene'),p=document.createElement('canvas');p.width=96;p.height=56;const x=p.getContext('2d');x.drawImage(c,0,0,96,56);const d=x.getImageData(0,0,96,56).data,s=new Set();for(let i=0;i<d.length;i+=4)s.add(`${d[i]>>3},${d[i+1]>>3},${d[i+2]>>3}`);return s.size;});
  assert.ok(await colours()>20,'WebGL orbit view renders geometry');
  await page.locator('#l3-frame').fill('3');
  assert.match(await page.locator('#l3-frame-info h3').innerText(),/第 4 \//);
  await page.locator('#l3-arm [data-arm="R0"]').click();
  await page.selectOption('#l3-view','1');
  assert.match(await page.evaluate(()=>document.querySelector('#l3-view').value),/^1$/);
  if(await page.locator('#l3-examples button[data-example]').count()){
    await page.locator('#l3-examples button[data-example]').first().click();
    assert.notEqual(await page.locator('#l3-view').inputValue(),'orbit');
  }
  await page.screenshot({path:`${shots}/rig-ab-desktop.png`,fullPage:true});

  // Synthetic sandbox: MC-lite worker, rerun, tabs, scrubber, reset, CSV export, L2 pilot.
  await page.waitForFunction(()=>document.querySelector('[data-metric="R0-p95"]').textContent!=='—',{timeout:60000});
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
  assert.equal((await downloadPromise).suggestedFilename(),'trials.csv');
  assert.match(await page.locator('#decision-label').innerText(),/^(NO-GO|CONDITIONAL GO|INCONCLUSIVE)$/,'the sandbox must not overwrite the real-input decision');
  await page.locator('#reload-l2-button').click();
  await page.waitForFunction(()=>document.querySelector('#l2-table').children.length>=3&&document.querySelector('#result-layer').value==='l2');
  assert.match(await page.locator('#scope-note').innerText(),/L2 图像解算/);

  await page.setViewportSize({width:390,height:844});
  await page.waitForTimeout(400);
  await page.screenshot({path:`${shots}/rig-ab-mobile.png`,fullPage:true});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true,'mobile horizontal overflow');
  assert.deepEqual(errors,[]);
  console.log('Browser passed: L3 verdict/cards/forest/CDF/rigid-body breakdown, WebGL replay + fisheye view + counterexample jump, MC-lite sandbox, L2 pilot, mobile layout.');
}finally{await browser.close();}
