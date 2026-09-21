import {
  ARM_COLORS, ARM_ORDER, DEFAULT_CONFIG, buildHybridTargets, buildRigTarget, clamp,
  makeCameras, makeWorldOccluders, percentile, resultForExport, rowsToCsv,
  runSimulation, transformPoint
} from './simulator-core.js';
import {readL2Report, allFrameCdf} from './l2-report.js';

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const fmt = (value, digits=2, suffix='') => Number.isFinite(value) ? `${Number(value).toFixed(digits)}${suffix}` : '—';
const pct = value => Number.isFinite(value) ? `${(100*value).toFixed(1)}%` : '—';
const forceSynchronousRun = new URLSearchParams(location.search).get('sync') === '1';

const state={cad:null,hybrid:null,targets:null,result:null,l2:null,runId:0,selectedArm:'R0',poseIndex:0,playing:true,timer:null,worker:null,orbit:{azimuth:-.75,elevation:.48,distance:1.0},drag:null};

const controls={
  poses:$('#poses'),episodes:$('#episodes'),occlusion:$('#occlusion'),speed:$('#speed'),angularSpeed:$('#angular-speed'),
  exposure:$('#exposure'),cornerSigma:$('#corner-sigma'),edgeSigma:$('#edge-sigma'),edgeRho:$('#edge-rho'),tcp:$('#tcp'),
  cameraDropout:$('#camera-dropout'),seed:$('#seed')
};

function readConfig(){return {
  ...DEFAULT_CONFIG,
  poses:Number(controls.poses.value),episodes:Number(controls.episodes.value),occlusion:Number(controls.occlusion.value)/100,
  linearSpeedMps:Number(controls.speed.value),angularSpeedDegS:Number(controls.angularSpeed.value),exposureMs:Number(controls.exposure.value),
  cornerSigmaPx:Number(controls.cornerSigma.value),edgeSigmaPx:Number(controls.edgeSigma.value),edgeCorrelation:Number(controls.edgeRho.value),
  tcpOffsetMm:Number(controls.tcp.value),cameraDropout:Number(controls.cameraDropout.value),seed:Number(controls.seed.value),
  rigInputStatus:state.targets?.R0?.inputStatus||'proxy',cameraInputStatus:'synthetic',trajectoryInputStatus:'synthetic',l2InputStatus:'model'
};}

function syncOutputs(){
  $('#occlusion-out').textContent=`${controls.occlusion.value}%`;$('#speed-out').textContent=`${Number(controls.speed.value).toFixed(2)} m/s`;
  $('#angular-speed-out').textContent=`${controls.angularSpeed.value}°/s`;$('#exposure-out').textContent=`${Number(controls.exposure.value).toFixed(1)} ms`;
  $('#corner-sigma-out').textContent=`${Number(controls.cornerSigma.value).toFixed(2)} px`;$('#edge-sigma-out').textContent=`${Number(controls.edgeSigma.value).toFixed(2)} px`;
  $('#edge-rho-out').textContent=Number(controls.edgeRho.value).toFixed(2);$('#tcp-out').textContent=`${controls.tcp.value} mm`;
}

const presets={
  task:{poses:240,episodes:12,occlusion:22,speed:.28,angularSpeed:55,exposure:5,cornerSigma:.22,edgeSigma:.42,edgeRho:.55,tcp:120,cameraDropout:0},
  clear:{poses:240,episodes:12,occlusion:0,speed:.06,angularSpeed:12,exposure:3,cornerSigma:.18,edgeSigma:.34,edgeRho:.4,tcp:120,cameraDropout:0},
  stress:{poses:360,episodes:12,occlusion:58,speed:.68,angularSpeed:145,exposure:10,cornerSigma:.45,edgeSigma:.78,edgeRho:.8,tcp:120,cameraDropout:2},
  socket:{poses:240,episodes:12,occlusion:22,speed:.28,angularSpeed:55,exposure:5,cornerSigma:.22,edgeSigma:.42,edgeRho:.55,tcp:0,cameraDropout:0}
};

function markPending(){$('#pending-note').textContent='参数已修改，点击“运行配对仿真”更新 MC-lite 结果。';}
function applyPreset(name){const p=presets[name]||presets.task;for(const [key,value] of Object.entries(p))if(controls[key])controls[key].value=value;syncOutputs();markPending();drawScene();}

async function loadInputs(){
  const base='../../../third_party/opencv_kalibr/metrology/fixtures/cad/';
  const manifestResponse=await fetch('./inputs_manifest.json');
  if(!manifestResponse.ok)throw new Error('Cannot load frozen input manifest');
  const manifest=await manifestResponse.json();
  const [cadResponse,hybridResponse]=await Promise.all([fetch(`${base}marker_rig_20260818_cad.json`),fetch(`${base}hybrid_carrier_v1_20260907.json`)]);
  if(!cadResponse.ok||!hybridResponse.ok)throw new Error('CAD descriptor fetch failed. Serve the repository root with python3 -m http.server; file:// cannot load the frozen JSON inputs.');
  async function checkedJson(response,expected){
    const bytes=await response.arrayBuffer();
    if(!globalThis.crypto?.subtle)throw new Error('Input hashing requires localhost or HTTPS');
    const digest=await crypto.subtle.digest('SHA-256',bytes),hex=[...new Uint8Array(digest)].map(b=>b.toString(16).padStart(2,'0')).join('');
    if(hex!==expected)throw new Error('CAD SHA-256 mismatch: freeze the new input manifest before running');
    return JSON.parse(new TextDecoder().decode(bytes));
  }
  [state.cad,state.hybrid]=await Promise.all([checkedJson(cadResponse,manifest.targets.R0.cad_sha256),checkedJson(hybridResponse,manifest.targets.H0_H1.descriptor_sha256)]);
  const hybridTargets=buildHybridTargets(state.hybrid);state.targets={R0:buildRigTarget(state.cad),...hybridTargets};
  $('#status-hybrid-text').textContent=`3 anchors · ${state.targets.H1.edges.length} edges`;$('#status-hybrid').className='';
  $('#status-rig-text').textContent='CAD + measured-size proxy';$('#status-rig').className='warn';
  $('#run-state').textContent='输入就绪';drawScene();await startRun();
}

function setRunBusy(busy,text){$('#run-button').disabled=busy;$('#run-button').textContent=busy?'仿真运行中…':'运行配对仿真';$('#run-state').textContent=text;}

async function startRun(){
  if(!state.targets)return;
  $('#result-layer').value='mc';
  if(state.worker){state.worker.terminate();state.worker=null;}
  const runId=++state.runId;
  setRunBusy(true,'构造共享轨迹…');
  $('#export-json').disabled=true;$('#export-csv').disabled=true;
  $('#load-layout-button').disabled=true;
  const config=readConfig(),runButton=$('#run-button');
  try{
    if(window.Worker && !forceSynchronousRun){
      const worker=new Worker('./sim-worker.js',{type:'module'});state.worker=worker;
      const result=await new Promise((resolve,reject)=>{
        worker.onmessage=event=>{const message=event.data;if(message.type==='progress'){const p=message.progress;if(p.phase==='fim')setRunBusy(true,`${p.arm} FIM · ${Math.round(100*p.ratio)}%`);}else if(message.type==='result')resolve(message.result);else if(message.type==='error')reject(new Error(message.message));};
        worker.onerror=event=>reject(new Error(event.message));worker.postMessage({targets:state.targets,config});
      });
      worker.terminate();if(runId!==state.runId)return;state.worker=null;acceptResult(result);
    }else{
      await new Promise(resolve=>setTimeout(resolve,30));if(runId!==state.runId)return;acceptResult(runSimulation(state.targets,config));
    }
  }catch(error){
    if(runId!==state.runId)return;
    if(state.worker){state.worker.terminate();state.worker=null;}
    setRunBusy(false,'运行失败');runButton.disabled=false;showError(error);
  }finally{
    if(runId===state.runId)$('#load-layout-button').disabled=false;
  }
}

function acceptResult(result){
  state.result=result;state.poseIndex=clamp(state.poseIndex,0,result.poses.length-1);$('#pose-scrubber').max=result.poses.length-1;$('#pose-scrubber').value=state.poseIndex;
  setRunBusy(false,`${result.config.poses} 帧 · ${result.config.episodes} episodes · seed ${result.config.seed}`);renderResult();drawScene();
  $('#export-json').disabled=false;$('#export-csv').disabled=false;
  $('#pending-note').textContent=JSON.stringify(readConfig())===JSON.stringify(result.config)?'参数对应当前 MC-lite 结果。':'参数已修改，当前显示上一次运行结果。';
}

function showError(error){
  console.error(error);const note=$('#scope-note');note.innerHTML=`<span class="error-box">${escapeHtml(error.message||String(error))}</span>`;
  const g2=$('#g2-result');g2.className='gate-result fail';g2.textContent='沙盒运行失败';
}

function escapeHtml(text){return String(text).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));}

function setMetric(name,text){const node=document.querySelector(`[data-metric="${name}"]`);if(node)node.textContent=text;}

function renderResult(){
  const {gates,validation}=state.result;
  const {metrics,paired}=activeResult();
  for(const arm of ARM_ORDER){setMetric(`${arm}-p95`,fmt(metrics[arm].p95,2,' mm'));setMetric(`${arm}-coverage`,pct(metrics[arm].coverage));}
  setMetric('delta',paired.deltaP95Mm===null?'—':`${paired.deltaP95Mm>=0?'+':''}${fmt(paired.deltaP95Mm,2,' mm')}`);
  setMetric('delta-ci',paired.ci95Mm[0]===null?'95% CI unavailable':`95% CI [${fmt(paired.ci95Mm[0])}, ${fmt(paired.ci95Mm[1])}] mm · n=${paired.count}`);
  $('#delta-bar').style.width=`${clamp((paired.deltaP95Mm||0)/.8*100,0,100)}%`;
  // The hero decision belongs to the real-input report (l3-app.js); the sandbox never writes it.
  const layer=$('#result-layer').value==='l2'?'L2 图像解算':'MC-lite 情景模型';
  if(!$('#run-button').disabled)$('#run-state').textContent=`${layer} · ${metrics.R0.total} 帧 · seed ${activeResult().config.seed}`;
  $('#scope-note').textContent=`${layer} · 各 arm 卡片为各自成功集 p95；配对 Δp95 = ${fmt(paired.deltaP95Mm,2,' mm')}，区间 ${paired.ci95Mm[0]===null?'不可用（共同成功样本不足）':`[${fmt(paired.ci95Mm[0])}, ${fmt(paired.ci95Mm[1])}] mm`}。正值表示 H1 误差更小；结果仅适用于声明的合成情景。`;
  renderMetricsTable();renderGates(validation,gates);drawCdf();drawFim();renderBars();
}

function activeResult(){return $('#result-layer').value==='l2'&&state.l2?state.l2:state.result;}

function acceptL2(doc){
  const report=readL2Report(doc);state.l2=report;
  $('#result-layer option[value="l2"]').disabled=false;$('#result-layer').value='l2';
  $('#l2-status').textContent=`已载入 ${report.metrics.R0.total} 帧 × 7 相机 · ${fmt(report.elapsedSeconds,1,' s')} · ${report.config?.supersample||'?'}× 超采样 · seed ${report.config?.seed??'—'}。本报告参数固定，左侧控件仅更新 MC-lite。`;
  const labels={R0:'R0 · 多相机角点',H0:'H0 · 同彩图角点',H1:'H1 · 同角点 + 原生边缘'};
  $('#l2-table').innerHTML=ARM_ORDER.map(arm=>{const m=report.metrics[arm];return `<tr><td>${labels[arm]}</td><td>${fmt(m.p95,2,' mm')}</td><td>${fmt(m.rotationP95,2,'°')}</td><td>${m.accepted} / ${m.total}</td><td>${pct(m.accurateYield)}</td><td>${pct(1-m.coverage)}</td></tr>`;}).join('');
  const native=report.native?.metrics;if(native)$('#l2-table').insertAdjacentHTML('beforeend',`<tr><td>H1 native · 冷启动独立分支</td><td>${fmt(native.p95,2,' mm')}</td><td>${fmt(native.rotationP95,2,'°')}</td><td>${fmt(native.accepted,0)} / ${fmt(native.total,0)}</td><td>${pct(native.accurateYield)}</td><td>${pct(1-native.coverage)}</td></tr>`);
  const previews=$('#l2-previews');previews.replaceChildren();
  for(const item of (report.previews||[]).slice(0,2)){
    if(typeof item.image!=='string'||!item.image.startsWith('data:image/png;base64,'))continue;
    const figure=document.createElement('figure'),img=document.createElement('img'),caption=document.createElement('figcaption');
    img.src=item.image;img.alt=`${item.arm} 鱼眼渲染输入`;caption.textContent=`${item.arm==='H0'?'H0 / H1 共用图像':item.arm} · ${item.camera} · frame 0`;
    figure.append(img,caption);previews.append(figure);
  }
  if(state.result)renderResult();
}

async function loadL2(optional=false){
  try{const response=await fetch('./l2_report.json',{cache:'no-store'});if(!response.ok){if(optional&&response.status===404)return;throw new Error('尚未找到 l2_report.json，请先运行离线图像试验。');}acceptL2(await response.json());}
  catch(error){$('#l2-status').textContent=error.message;}
}

function renderMetricsTable(){
  const rows=ARM_ORDER.map(arm=>{const m=state.result.metrics[arm];return `<tr><td style="color:${ARM_COLORS[arm]}">${arm}</td><td>${fmt(m.p50,2,' mm')}</td><td><b>${fmt(m.p95,2,' mm')}</b></td><td>${fmt(m.p99,2,' mm')}</td><td>${fmt(m.rotationP95,2,'°')}</td><td>${pct(m.coverage)}</td><td>${pct(m.accurateYield)}</td><td>${pct(m.catastrophicRate)}</td><td>${fmt(m.l1SigmaP95,3,' mm')}</td></tr>`;});
  $('#metrics-table').innerHTML=rows.join('');
}

function renderGates(validation,gates){
  $('#g0-list').innerHTML=validation.checks.map(check=>`<li class="${check.ok?'':check.warning?'warn':'fail'}">${check.ok?'PASS':'WAIT'} · ${escapeHtml(check.label)}</li>`).join('');
  const g1=$('#g1-result'),outcome=gates.g1Outcome||'STOP';
  g1.className=`gate-result ${outcome==='STOP'?'fail':outcome==='LIMITED_L2'?'pending':''}`;
  g1.textContent=outcome==='CONTINUE'?'CONTINUE TO L2':outcome==='LIMITED_L2'?'LIMITED L2 CHECK':'STOP';
  $('#g1-detail').textContent=Number.isFinite(gates.fimImprovement)?`H1 相对 R0 的 L1 σ p95 改善 ${pct(gates.fimImprovement)}；完整扩展筛选线为 30%。H0→H1 的可见性变化只作为有限 L2 假设，不当作 detector 证据。`:'可观测姿态不足，无法形成稳定 FIM 比较。';
  const g2=$('#g2-result');g2.className='gate-result pending';g2.textContent=gates.decision.replace('_',' ');
  const missing=[];if(!validation.g0Pass)missing.push('G0 冻结输入');if(state.result.config.l2InputStatus!=='validated-native-detector')missing.push('原生 detector L2');
  $('#g2-detail').textContent=`数值门槛 ${gates.numericG2?'满足':'未满足'}；证据层 ${gates.evidenceReady?'完整':'未完整'}${missing.length?`（缺 ${missing.join('、')}）`:''}。`;
}

function setupCanvas(canvas){
  const rect=canvas.getBoundingClientRect(),dpr=Math.min(window.devicePixelRatio||1,2),width=Math.max(20,Math.round(rect.width*dpr)),height=Math.max(20,Math.round(rect.height*dpr));
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return {ctx,w:rect.width,h:rect.height};
}

function drawAxes(ctx,w,h,box,xmax,ymax){
  ctx.strokeStyle='#dfe3dc';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(box.l,box.t);ctx.lineTo(box.l,box.b);ctx.lineTo(box.r,box.b);ctx.stroke();
  ctx.fillStyle='#748079';ctx.font='10px system-ui';ctx.textAlign='center';for(let i=0;i<=5;i++){const x=box.l+(box.r-box.l)*i/5,value=xmax*i/5;ctx.fillText(value.toFixed(value<2?1:0),x,box.b+17);ctx.strokeStyle='#eef0eb';ctx.beginPath();ctx.moveTo(x,box.t);ctx.lineTo(x,box.b);ctx.stroke();}
  ctx.textAlign='right';for(let i=0;i<=4;i++){const y=box.b-(box.b-box.t)*i/4;ctx.fillText(`${Math.round(ymax*i/4)}%`,box.l-8,y+3);}
}

function drawCdf(){
  const canvas=$('#cdf-chart'),{ctx,w,h}=setupCanvas(canvas),box={l:52,t:20,r:w-20,b:h-42};ctx.clearRect(0,0,w,h);
  const metrics=activeResult().metrics;
  $('#cdf-layer').textContent=`${$('#result-layer').value==='l2'?'L2 IMAGE':'MC-LITE'} · ALL REQUESTED FRAMES`;
  const all=ARM_ORDER.flatMap(arm=>metrics[arm].errors),xmax=Math.max(4,(percentile(all,.995)||5)*1.15);drawAxes(ctx,w,h,box,xmax,100);
  for(const arm of ARM_ORDER){ctx.strokeStyle=ARM_COLORS[arm];ctx.lineWidth=2.6;ctx.beginPath();ctx.moveTo(box.l,box.b);let lastY=box.b;for(const point of allFrameCdf(metrics[arm])){if(point.error>xmax)break;const x=box.l+point.error/xmax*(box.r-box.l),y=box.b-point.fraction*(box.b-box.t);ctx.lineTo(x,lastY);ctx.lineTo(x,y);lastY=y;}ctx.lineTo(box.r,lastY);ctx.stroke();}
  const thresholdX=box.l+clamp(3/xmax,0,1)*(box.r-box.l);ctx.setLineDash([5,5]);ctx.strokeStyle='#75817b';ctx.beginPath();ctx.moveTo(thresholdX,box.t);ctx.lineTo(thresholdX,box.b);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#5f6b65';ctx.font='10px system-ui';ctx.fillText('3 mm',thresholdX-4,box.t+12);ctx.fillText('translation error / mm',(box.l+box.r)/2,h-8);
}

function drawFim(){
  const canvas=$('#fim-chart'),{ctx,w,h}=setupCanvas(canvas),box={l:45,t:16,r:w-15,b:h-34};ctx.clearRect(0,0,w,h);
  const values=ARM_ORDER.flatMap(arm=>state.result.analysesByArm[arm].map(a=>a.sigmaTcpWorstMm).filter(Number.isFinite));const ymax=clamp((percentile(values,.99)||1)*1.2,.05,8);
  ctx.strokeStyle='#dfe3dc';ctx.beginPath();ctx.moveTo(box.l,box.t);ctx.lineTo(box.l,box.b);ctx.lineTo(box.r,box.b);ctx.stroke();
  for(const arm of ARM_ORDER){const series=state.result.analysesByArm[arm],step=Math.max(1,Math.floor(series.length/180));ctx.strokeStyle=ARM_COLORS[arm];ctx.lineWidth=1.5;ctx.globalAlpha=.9;ctx.beginPath();let started=false;for(let i=0;i<series.length;i+=step){const value=series[i].sigmaTcpWorstMm;if(!Number.isFinite(value))continue;const x=box.l+i/(series.length-1)*(box.r-box.l),y=box.b-clamp(value/ymax,0,1)*(box.b-box.t);if(!started){ctx.moveTo(x,y);started=true;}else ctx.lineTo(x,y);}ctx.stroke();}ctx.globalAlpha=1;
  ctx.fillStyle='#68756e';ctx.font='9px system-ui';ctx.textAlign='right';ctx.fillText(`${ymax.toFixed(2)} mm`,box.l-5,box.t+3);ctx.fillText('0',box.l-5,box.b+3);ctx.textAlign='center';ctx.fillText('shared pose index',(box.l+box.r)/2,h-7);
}

function renderBars(){
  const metrics=activeResult().metrics,maxP95=Math.max(...ARM_ORDER.map(a=>metrics[a].p95||0),1);let html='<div class="bar-subtitle">TCP p95 · lower is better</div>';
  for(const arm of ARM_ORDER)html+=barRow(arm,(metrics[arm].p95||0)/maxP95,fmt(metrics[arm].p95,2,' mm'),ARM_COLORS[arm]);
  html+='<div class="bar-subtitle">all-frame accurate yield · higher is better</div>';for(const arm of ARM_ORDER)html+=barRow(arm,metrics[arm].accurateYield,pct(metrics[arm].accurateYield),ARM_COLORS[arm]);
  html+='<div class="bar-subtitle">failed / all requested frames</div>';for(const arm of ARM_ORDER)html+=barRow(arm,1-metrics[arm].coverage,pct(1-metrics[arm].coverage),ARM_COLORS[arm]);$('#comparison-bars').innerHTML=html;
}
function barRow(label,value,text,color){return `<div class="bar-row"><span>${label}</span><div class="bar-track"><i style="width:${clamp(value,0,1)*100}%;background:${color}"></i></div><b>${text}</b></div>`;}

function sceneView(){
  const {azimuth:a,elevation:e,distance:d}=state.orbit,centre=[0,0,.25],position=[centre[0]+d*Math.cos(e)*Math.cos(a),centre[1]+d*Math.cos(e)*Math.sin(a),centre[2]+d*Math.sin(e)];
  const forward=unit(sub3(centre,position)),upHint=[0,0,1],right=unit(cross3(upHint,forward)),up=unit(cross3(forward,right));return {position,forward,right,up,centre};
}
function sub3(a,b){return a.map((x,i)=>x-b[i]);}function dot3(a,b){return a.reduce((s,x,i)=>s+x*b[i],0);}function cross3(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}function unit(v){const n=Math.hypot(...v)||1;return v.map(x=>x/n);}
function viewProject(view,p,w,h){const d=sub3(p,view.position),z=dot3(d,view.forward);if(z<.02)return null;const f=Math.min(w,h)*1.32;return {x:w/2+f*dot3(d,view.right)/z,y:h/2-f*dot3(d,view.up)/z,z};}
function hexAlpha(hex,alpha){const h=hex.replace('#','');const n=parseInt(h.length===3?h.split('').map(x=>x+x).join(''):h,16);return `rgba(${n>>16},${(n>>8)&255},${n&255},${alpha})`;}

function drawScene(){
  if(!state.targets)return;const canvas=$('#scene'),{ctx,w,h}=setupCanvas(canvas),view=sceneView(),arm=state.selectedArm,target=state.targets[arm];ctx.clearRect(0,0,w,h);
  const pose=state.result?.poses[state.poseIndex]||{R:[[1,0,0],[0,1,0],[0,0,1]],t:[0,0,.25],phase:0};const analysis=state.result?.analysesByArm[arm]?.[state.poseIndex];
  const grid=[];for(let x=-.4;x<=.4+.001;x+=.1)grid.push([[x,-.4,.08],[x,.4,.08]]);for(let y=-.4;y<=.4+.001;y+=.1)grid.push([[-.4,y,.08],[.4,y,.08]]);
  ctx.strokeStyle='rgba(196,224,213,.11)';ctx.lineWidth=1;for(const line of grid){const a=viewProject(view,line[0],w,h),b=viewProject(view,line[1],w,h);if(a&&b){ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}}
  const polygons=target.displayPolygons.map(poly=>{const world=poly.points.map(p=>transformPoint(pose,p)),projected=world.map(p=>viewProject(view,p,w,h));return {...poly,world,projected,depth:projected.reduce((s,p)=>s+(p?.z||0),0)/projected.length};}).filter(p=>p.projected.every(Boolean)).sort((a,b)=>b.depth-a.depth);
  for(const poly of polygons){ctx.beginPath();poly.projected.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.fillStyle=hexAlpha(poly.color||'#83948c',poly.anchor?.94:.62);ctx.fill();ctx.strokeStyle=poly.anchor?'rgba(19,31,26,.85)':'rgba(220,235,228,.27)';ctx.lineWidth=poly.anchor?1.6:.8;ctx.stroke();if(poly.markerPoints){const pts=poly.markerPoints.map(p=>viewProject(view,transformPoint(pose,p),w,h));if(pts.every(Boolean)){ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.fillStyle='#17201d';ctx.fill();}}}
  if(arm==='H1'){ctx.strokeStyle=hexAlpha(ARM_COLORS.H1,.82);ctx.lineWidth=1.4;for(const edge of target.edges){const a=viewProject(view,transformPoint(pose,edge.p0),w,h),b=viewProject(view,transformPoint(pose,edge.p1),w,h);if(a&&b){ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}}}
  const cameras=state.result?.cameras||makeCameras(Number(controls.cameraDropout.value));for(const [index,camera] of cameras.entries()){const p=viewProject(view,camera.position,w,h),aim=viewProject(view,[0,0,.25],w,h);if(!p)continue;ctx.fillStyle='#d6e9e1';ctx.beginPath();ctx.arc(p.x,p.y,4,0,2*Math.PI);ctx.fill();ctx.fillStyle='rgba(215,234,226,.7)';ctx.font='9px system-ui';ctx.fillText(String(index+1),p.x+7,p.y-5);if(aim){ctx.strokeStyle='rgba(196,224,213,.09)';ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(aim.x,aim.y);ctx.stroke();}}
  const occluders=analysis?.occluders||makeWorldOccluders(pose,Number(controls.occlusion.value)/100,pose.phase||0);for(const sphere of occluders){const p=viewProject(view,sphere.c,w,h),p2=viewProject(view,[sphere.c[0]+sphere.r,sphere.c[1],sphere.c[2]],w,h);if(!p||!p2)continue;const radius=Math.max(3,Math.hypot(p2.x-p.x,p2.y-p.y));ctx.fillStyle='rgba(182,111,77,.35)';ctx.strokeStyle='rgba(236,161,119,.65)';ctx.beginPath();ctx.arc(p.x,p.y,radius,0,2*Math.PI);ctx.fill();ctx.stroke();}
  if(analysis){ctx.strokeStyle='rgba(111,223,178,.23)';ctx.lineWidth=1;for(const viewItem of analysis.visibleAnchorViews.slice(0,14)){const camera=cameras.find(c=>c.name===viewItem.camera),a=camera&&viewProject(view,camera.position,w,h),b=viewProject(view,viewItem.centre,w,h);if(a&&b){ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}}}
  const pivot=viewProject(view,pose.t,w,h);if(pivot){ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(pivot.x,pivot.y,4,0,2*Math.PI);ctx.fill();ctx.strokeStyle='rgba(255,255,255,.4)';ctx.beginPath();ctx.arc(pivot.x,pivot.y,9,0,2*Math.PI);ctx.stroke();}
  $('#scene-caption').textContent=analysis?`pose ${state.poseIndex} · ${analysis.cameraCount} cams · ${analysis.anchorCount} anchors · ${analysis.edgeRows} edge rows · σ ${fmt(analysis.sigmaTcpWorstMm,3,' mm')}`:'拖动画面旋转视角';
}

function animate(){if(state.playing&&state.result){state.poseIndex=(state.poseIndex+1)%state.result.poses.length;$('#pose-scrubber').value=state.poseIndex;drawScene();}state.timer=requestAnimationFrame(animate);}

function download(name,text,mime){const blob=new Blob([text],{type:mime}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}

function bind(){
  Object.values(controls).forEach(control=>control.addEventListener('input',()=>{syncOutputs();markPending();if(control===controls.cameraDropout)drawScene();}));
  $('#result-layer').addEventListener('change',()=>{if(state.result)renderResult();});
  $('#load-l2-button').addEventListener('click',()=>$('#l2-file').click());
  $('#reload-l2-button').addEventListener('click',()=>loadL2());
  $('#l2-file').addEventListener('change',async event=>{const file=event.target.files?.[0];if(!file)return;try{acceptL2(JSON.parse(await file.text()));}catch(error){$('#l2-status').textContent=error.message;}event.target.value='';});
  $('#preset').addEventListener('change',event=>applyPreset(event.target.value));$('#reset-button').addEventListener('click',()=>{$('#preset').value='task';controls.seed.value=DEFAULT_CONFIG.seed;applyPreset('task');});$('#run-button').addEventListener('click',startRun);
  $('#load-layout-button').addEventListener('click',()=>$('#layout-file').click());$('#layout-file').addEventListener('change',async event=>{const file=event.target.files?.[0];if(!file)return;try{const doc=JSON.parse(await file.text()),rig=buildRigTarget(state.cad,doc);state.targets={...state.targets,R0:rig};$('#status-rig-text').textContent=`measured · ${file.name}`;$('#status-rig').className='';await startRun();}catch(error){showError(error);}event.target.value='';});
  $('#target-tabs').addEventListener('click',event=>{const button=event.target.closest('button[data-target]');if(!button)return;state.selectedArm=button.dataset.target;$$('#target-tabs button').forEach(b=>b.classList.toggle('active',b===button));drawScene();});
  $('#pose-scrubber').addEventListener('input',event=>{state.poseIndex=Number(event.target.value);state.playing=false;$('#play-button').textContent='播放';drawScene();});$('#play-button').addEventListener('click',()=>{state.playing=!state.playing;$('#play-button').textContent=state.playing?'暂停':'播放';});
  const scene=$('#scene');scene.addEventListener('pointerdown',event=>{state.drag={x:event.clientX,y:event.clientY,a:state.orbit.azimuth,e:state.orbit.elevation};scene.setPointerCapture(event.pointerId);});scene.addEventListener('pointermove',event=>{if(!state.drag)return;state.orbit.azimuth=state.drag.a-(event.clientX-state.drag.x)*.006;state.orbit.elevation=clamp(state.drag.e+(event.clientY-state.drag.y)*.005,-.05,1.35);drawScene();});scene.addEventListener('pointerup',()=>state.drag=null);scene.addEventListener('wheel',event=>{event.preventDefault();state.orbit.distance=clamp(state.orbit.distance*(1+Math.sign(event.deltaY)*.08),.45,2);drawScene();},{passive:false});
  $('#export-json').addEventListener('click',()=>download('decision.json',JSON.stringify(resultForExport(state.result),null,2),'application/json'));$('#export-csv').addEventListener('click',()=>download('trials.csv',rowsToCsv(state.result),'text/csv'));
  window.addEventListener('resize',()=>{drawScene();if(state.result){drawCdf();drawFim();}});
}

syncOutputs();bind();animate();loadInputs().then(()=>loadL2(true)).catch(showError);
