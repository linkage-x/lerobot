/* Real-input result page: verdict, one-axis comparison, WebGL replay, evidence. */
import {buildHybridTargets, buildRigTarget} from './simulator-core.js';
import {ARMS, GATE_MM, LEVEL_LABELS, OUTPUT_LABELS, attribution, forestRows, readL3Report} from './l3-report.js';
import {SceneView} from './webgl-scene.js';

const $ = selector => document.querySelector(selector);
const fmt = (value, digits = 2, suffix = '') => Number.isFinite(value) ? `${Number(value).toFixed(digits)}${suffix}` : '—';
const pct = value => Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : '—';
const signed = (value, digits = 2) => Number.isFinite(value) ? `${value > 0 ? '+' : value < 0 ? '−' : '±'}${Math.abs(value).toFixed(digits)}` : '—';
const interval = ci => Array.isArray(ci) && Number.isFinite(ci[0]) && Number.isFinite(ci[1]) ? `[${signed(ci[0])}, ${signed(ci[1])}]` : '[—]';
const escapeHtml = text => String(text).replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
const COLORS = {R0: '#e15b3d', H0: '#e2a72e', H1: '#1d826b'};
const ARM_LABEL = {R0: 'R0 · rig0818', H0: 'H0 · 0907 anchors', H1: 'H1 · 0907 full'};
const ARM_SUB = {R0: '5 anchors · corner BA', H0: '3 anchors · corner BA', H1: '3 anchors + 31 edges'};
const DECISION = {NO_GO: 'NO-GO', CONDITIONAL_GO: 'CONDITIONAL GO', INCONCLUSIVE: 'INCONCLUSIVE'};
const CAD = '../../../third_party/opencv_kalibr/metrology/fixtures/cad/';
const HAND = {left: '左手', right: '右手'};

const state = {
  doc: null, level: 'evidence', output: 'smoothed', view: null, row: 0, arm: 'H1', mode: 'orbit', cameraIndex: 0,
  playing: false, lastStep: 0, orbit: {azimuth: -2.35, elevation: 0.5, distance: 1.9, centre: [0.5, 0.1, 0.5]}, drag: null, centroids: null
};

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(20, Math.round(rect.width * dpr)), height = Math.max(20, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, w: rect.width, h: rect.height};
}

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function currentBlock() { return state.doc.groups.gripper.levels[state.level][state.output]; }

function renderHero() {
  const d = state.doc.decision, card = $('#decision-card');
  card.className = `decision-card ${d.formal === 'CONDITIONAL_GO' ? 'ready' : d.numeric === 'NO_GO' ? 'fail' : ''}`;
  $('#decision-kicker').textContent = '正式判定 · 真实输入 L2 + L3';
  $('#decision-label').textContent = DECISION[d.formal];
  $('#decision-reason').innerHTML = `数值方向 <b>${DECISION[d.numeric]}</b>：Δp95(R0−H1) ${signed(d.primary.delta)} mm，95% 区间 ${interval(d.primary.ci)} mm；立项门槛为区间下界 ≥ ${GATE_MM} mm。`
    + (d.g0Pass ? '' : `<br>未关闭的阻塞输入：${escapeHtml(d.missing.join('；'))}。`);
}

function renderVerdict() {
  const d = state.doc.decision, [lo, hi] = d.primary.ci, box = $('#l3-verdict');
  const kind = d.numeric === 'CONDITIONAL_GO' ? 'go' : d.numeric === 'NO_GO' ? 'nogo' : 'open';
  let headline, detail;
  if (d.numeric === 'NO_GO') {
    headline = hi < 0 ? `H1 比 R0 更差：p95 多 ${fmt(Math.abs(d.primary.delta))} mm`
      : lo > 0 ? `H1 更准，但只好 ${fmt(d.primary.delta)} mm，够不到 ${GATE_MM} mm 门槛`
        : `H1 与 R0 没有可分辨的差别，收益上界也够不到 ${GATE_MM} mm`;
    const strata = (state.doc.groups.gripper.levels.evidence.smoothed.buckets || [])
      .filter(b => ['speed', 'occlusion'].includes(b.family) && Number.isFinite(b.R0_H1?.ci?.[1]));
    const crossing = forestRows(state.doc).filter(r => r.kind !== 'bucket' && Number.isFinite(r.ci?.[1]) && r.ci[1] >= GATE_MM);
    detail = `判定口径：证据级扰动 + 离线平滑（数据集标签），95% 区间 ${interval(d.primary.ci)} mm，上界低于 ${GATE_MM} mm。`
      + (d.inversions.length && d.inversions.length === strata.length ? ' 每个速度与遮挡分层里 H1 都显著更差。'
        : d.inversions.length ? ` H1 显著更差的分层：${escapeHtml(d.inversions.join('、'))}。` : '')
      + (crossing.length
        ? ` 区间跨过门槛的只有${crossing.map(r => `「${escapeHtml(r.label)}」${signed(r.delta)} ${interval(r.ci)}`).join('、')}：区间宽，点估计${crossing.every(r => r.delta < 0) ? '仍是 H1 更差' : '方向不一'}。`
        : ' 所有扰动级别、两种输出与对照组的区间上界都低于门槛。');
  } else if (d.numeric === 'CONDITIONAL_GO') {
    headline = `H1 比 R0 好 ${fmt(d.primary.delta)} mm，区间下界过门槛`;
    detail = `区间下界 ${signed(lo)} mm ≥ ${GATE_MM} mm，H1 p95、旋转、覆盖率与快速/遮挡分层均满足。仍需真实最小 A/B 与独立 GT。`;
  } else {
    headline = '区间跨过门槛，数值上无法判定';
    detail = `95% 区间 ${interval(d.primary.ci)} mm 同时覆盖不足与超过 ${GATE_MM} mm 的收益${d.inversions.length ? `；分层反转：${escapeHtml(d.inversions.join('、'))}` : ''}。`;
  }
  box.className = `verdict ${kind}`;
  box.innerHTML = `<div class="verdict-main"><span class="verdict-chip">${DECISION[d.numeric]}</span><div><strong>${headline}</strong><p>${detail}</p></div></div>`
    + `<div class="verdict-side"><span>正式判定</span><b>${DECISION[d.formal]}</b><small>${d.g0Pass ? 'G0 阻塞项全部关闭' : `缺：${escapeHtml(d.missing.join('；'))}`}</small></div>`;
}

function gauge(p) {
  const lo = -3, hi = 3, x = v => (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * 100;
  const [a, b] = p.ci, ok = Number.isFinite(a) && Number.isFinite(b);
  return `<div class="gauge" aria-hidden="true"><i class="gauge-zone" style="left:${x(GATE_MM)}%"></i><i class="gauge-zero" style="left:${x(0)}%"></i>`
    + `<i class="gauge-gate" style="left:${x(GATE_MM)}%"></i>${ok ? `<i class="gauge-ci" style="left:${x(a)}%;width:${Math.max(0.8, x(b) - x(a))}%"></i>` : ''}`
    + `${Number.isFinite(p.delta) ? `<i class="gauge-dot" style="left:${x(p.delta)}%"></i>` : ''}</div>`
    + `<div class="gauge-scale"><span style="left:0">−3</span><span style="left:${x(0)}%">0</span><span style="left:${x(GATE_MM)}%">+${GATE_MM} 门槛</span><span style="left:100%">+3 mm</span></div>`;
}

function renderCards() {
  const doc = state.doc, level = doc.groups.gripper.levels[state.level], block = level[state.output];
  const max = Math.max(6, ...ARMS.map(arm => block.arms[arm].p95 || 0)) * 1.08;
  const cards = ARMS.map(arm => {
    const m = block.arms[arm], mount = doc.groups.gripper.mounts[arm === 'R0' ? 'R0' : 'H'];
    return `<article class="metric-card ${arm.toLowerCase()}"><div><span>${ARM_LABEL[arm]}</span><small>${ARM_SUB[arm]}</small></div>`
      + `<strong data-l3-metric="${arm}-p95">${fmt(m.p95)}<em> mm</em></strong><p>TCP 平移 p95 · 旋转 p95 ${fmt(m.rotation_p95, 2, '°')}</p>`
      + `<div class="target-bar" title="竖虚线 = 3 mm 目标"><i style="width:${Math.min(100, (m.p95 || 0) / max * 100)}%"></i><b style="left:${3 / max * 100}%"></b></div>`
      + `<dl><dt>覆盖率</dt><dd>${pct(m.coverage)}</dd><dt>≤3 mm 且 ≤0.5°</dt><dd>${pct(m.accurate_yield)}</dd><dt>粗差 &gt;20 mm/5°</dt><dd>${pct(m.catastrophic_rate)}</dd></dl>`
      + `<footer>安装 yaw ${mount.yaw_deg}° · 臂仰 ${-mount.pitch_deg}° · marker→TCP ${fmt(mount.feature_centroid_to_tcp_mm, 0)} mm</footer></article>`;
  });
  const p = block.paired.R0_H1;
  cards.push(`<article class="delta-card"><span>R0 → H1 配对 Δp95</span><strong data-l3-metric="delta">${signed(p.delta)} mm</strong>`
    + `<p>95% 区间 ${interval(p.ci)} mm · ${p.n} 个配对样本（${level.sessions} session × ${level.frames} 帧）</p>${gauge(p)}`
    + `<p class="gauge-note">正值 = H1 更准；立项需整个区间落在门槛右侧</p></article>`);
  $('#l3-cards').innerHTML = cards.join('');
  $('#l3-scope').textContent = `卡片与 CDF：${LEVEL_LABELS[state.level]} · ${OUTPUT_LABELS[state.output]}。覆盖率、达标率与 CDF 的分母是全部请求帧（失败帧不删除）；Δ 只在双方都出位姿的帧上配对，按 session × 2 s 时间块 bootstrap（inverted-CDF 分位数）。`
    + `留出 episode ${doc.groups.gripper.episodes.join(' / ')}，每 ${doc.groups.gripper.stride} 帧取 1 帧；真值帧中 ${pct(doc.groups.gripper.interpolated_fraction)} 为插值。`;
}

function drawForest() {
  const rows = forestRows(state.doc), canvas = $('#l3-forest');
  const narrow = canvas.getBoundingClientRect().width < 620, rowH = narrow ? 44 : 28, top = 30, bottom = 44;
  canvas.style.height = `${top + rows.length * rowH + bottom}px`;
  const {ctx, w, h} = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const labelW = narrow ? 0 : Math.min(290, w * 0.36), valueW = narrow ? 0 : 138;
  const x0 = labelW + 14, x1 = w - valueW - 14;
  const values = rows.flatMap(r => [r.ci?.[0], r.ci?.[1], r.delta]).filter(Number.isFinite);
  const lo = Math.floor(Math.min(-1, ...values) * 2 - 0.4) / 2, hi = Math.ceil(Math.max(GATE_MM + 0.7, ...values) * 2 + 0.4) / 2;
  const X = v => x0 + (v - lo) / (hi - lo) * (x1 - x0), bottomY = top + rows.length * rowH;
  ctx.fillStyle = 'rgba(29,130,107,.09)';
  ctx.fillRect(X(GATE_MM), top - 8, x1 - X(GATE_MM), bottomY - top + 8);
  ctx.font = '10px system-ui, sans-serif';
  ctx.textAlign = 'center';
  const step = hi - lo > 6 ? 1 : 0.5;
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    ctx.strokeStyle = Math.abs(v) < 1e-9 ? 'rgba(18,34,29,.45)' : 'rgba(18,34,29,.08)';
    ctx.beginPath(); ctx.moveTo(X(v), top - 8); ctx.lineTo(X(v), bottomY); ctx.stroke();
    ctx.fillStyle = '#6b7771'; ctx.fillText(`${v > 0 ? '+' : ''}${v.toFixed(step < 1 ? 1 : 0)}`, X(v), bottomY + 14);
  }
  ctx.setLineDash([5, 4]); ctx.strokeStyle = '#1d826b'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(X(GATE_MM), top - 14); ctx.lineTo(X(GATE_MM), bottomY); ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
  ctx.fillStyle = '#1d826b'; ctx.textAlign = 'left'; ctx.fillText(`立项门槛 ${GATE_MM} mm →`, X(GATE_MM) + 5, top - 16);
  ctx.fillStyle = '#58645e'; ctx.textAlign = 'center';
  ctx.fillText('Δp95 = p95(R0) − p95(H1)，mm（正值 = H1 更准）', (x0 + x1) / 2, bottomY + 32);
  rows.forEach((row, i) => {
    const yMid = top + i * rowH + rowH / 2, y = narrow ? yMid + 8 : yMid, [a, b] = row.ci;
    if (row.primary) { ctx.fillStyle = 'rgba(223,184,76,.2)'; ctx.fillRect(0, top + i * rowH + 1, w, rowH - 2); }
    if (i && rows[i - 1].kind !== row.kind) { ctx.strokeStyle = 'rgba(18,34,29,.18)'; ctx.beginPath(); ctx.moveTo(0, top + i * rowH); ctx.lineTo(w, top + i * rowH); ctx.stroke(); }
    const colour = Number.isFinite(a) && a >= GATE_MM ? '#1d826b' : Number.isFinite(b) && b < GATE_MM ? '#c2553c' : '#c49222';
    ctx.textAlign = 'left'; ctx.fillStyle = row.primary ? '#12221d' : '#3f4d47';
    ctx.font = `${row.primary ? 700 : 500} ${narrow ? 10.5 : 11.5}px system-ui, sans-serif`;
    if (narrow) ctx.fillText(`${row.label}  ${signed(row.delta)} ${interval(row.ci)}`, 4, yMid - 9);
    else ctx.fillText(row.label, 4, y + 4);
    if (Number.isFinite(a) && Number.isFinite(b)) {
      ctx.strokeStyle = colour; ctx.lineWidth = row.primary ? 4 : 2.6; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(X(Math.max(lo, a)), y); ctx.lineTo(X(Math.min(hi, b)), y); ctx.stroke(); ctx.lineWidth = 1; ctx.lineCap = 'butt';
    }
    if (Number.isFinite(row.delta)) {
      ctx.fillStyle = '#fff'; ctx.strokeStyle = colour; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(Math.max(lo, Math.min(hi, row.delta))), y, row.primary ? 5.5 : 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); ctx.lineWidth = 1;
    }
    if (!narrow) {
      ctx.textAlign = 'right'; ctx.fillStyle = '#3f4d47'; ctx.font = `${row.primary ? 700 : 500} 11px ui-monospace, monospace`;
      ctx.fillText(`${signed(row.delta)} ${interval(row.ci)}`, w - 4, y + 4);
    }
  });
}

function drawCdf() {
  const block = currentBlock(), canvas = $('#l3-cdf'), {ctx, w, h} = setupCanvas(canvas), box = {l: 44, t: 16, r: w - 14, b: h - 36};
  ctx.clearRect(0, 0, w, h);
  const xs = block.cdf.x_mm, xmax = xs[xs.length - 1];
  ctx.font = '10px system-ui, sans-serif'; ctx.fillStyle = '#6b7771';
  for (let i = 0; i <= 4; i++) {
    const y = box.b - (box.b - box.t) * i / 4;
    ctx.strokeStyle = 'rgba(18,34,29,.07)'; ctx.beginPath(); ctx.moveTo(box.l, y); ctx.lineTo(box.r, y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(`${25 * i}%`, box.l - 6, y + 3);
  }
  for (let v = 0; v <= xmax + 1e-9; v += 2) {
    const x = box.l + v / xmax * (box.r - box.l);
    ctx.textAlign = 'center'; ctx.fillText(String(v), x, box.b + 14);
  }
  const x3 = box.l + 3 / xmax * (box.r - box.l);
  ctx.setLineDash([4, 4]); ctx.strokeStyle = '#75817b'; ctx.beginPath(); ctx.moveTo(x3, box.t); ctx.lineTo(x3, box.b); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#5f6b65'; ctx.textAlign = 'left'; ctx.fillText('3 mm', x3 + 4, box.t + 10);
  for (const arm of ARMS) {
    ctx.strokeStyle = COLORS[arm]; ctx.lineWidth = 2.4; ctx.beginPath();
    block.cdf.arms[arm].forEach((f, i) => { const x = box.l + xs[i] / xmax * (box.r - box.l), y = box.b - f * (box.b - box.t); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke(); ctx.lineWidth = 1;
  }
  ctx.fillStyle = '#5f6b65'; ctx.textAlign = 'center'; ctx.fillText('TCP 平移误差 / mm', (box.l + box.r) / 2, h - 6);
  $('#l3-cdf-layer').textContent = `${LEVEL_LABELS[state.level]} · ${OUTPUT_LABELS[state.output]}`;
}

function renderAttribution() {
  const att = attribution(state.doc, state.level, state.output);
  const row = (label, sub, p) => {
    const colour = Number.isFinite(p.ci?.[0]) && p.ci[0] > 0 ? 'gain' : Number.isFinite(p.ci?.[1]) && p.ci[1] < 0 ? 'loss' : 'flat';
    return `<div class="attr-row ${colour}"><div><b>${label}</b><small>${sub}</small></div><strong>${signed(p.delta)} mm</strong><span>${interval(p.ci)}</span></div>`;
  };
  $('#l3-attribution').innerHTML = row('R0 → H0', '换成 0907 的 3 个 anchor：角点分布更紧，旋转约束更弱', att.geometry)
    + row('H0 → H1', '同一图像、同一角点，再加彩色边缘精修', att.edges)
    + row('R0 → H1', '合计，即立项比较', att.total)
    + '<p class="attr-note">正值表示后者更准；每行单独配对并 bootstrap，三行不必严格相加。</p>';
  renderMechanism();
}

function renderMechanism() {
  const box = $('#l3-mechanism'), mech = state.doc.groups.gripper.mechanism;
  if (!box) return;
  if (!mech?.levels) { box.innerHTML = '<p class="small-note">这份报告没有刚体分解。</p>'; return; }
  const name = mech.levels[state.level] ? state.level : 'evidence', level = mech.levels[name], g = mech.geometry;
  const cols = [['centroid', '特征质心'], ['origin', 'rig 原点（球窝）'], ['common', `同一力臂 ${fmt(mech.common_arm_mm, 0)} mm`], ['tcp', 'TCP 标签（判定口径）']];
  let html = `<div class="table-wrap"><table class="mechanism-table"><thead><tr><th>分支</th>${cols.map(([, label]) => `<th>${label}</th>`).join('')}<th>旋转</th></tr></thead><tbody>`
    + ARMS.map(arm => { const a = level.arms[arm] || {}; return `<tr><td style="color:${COLORS[arm]}">${arm}</td>${cols.map(([key]) => `<td>${fmt(a[key]?.p95, 2)}</td>`).join('')}<td>${fmt(a.rotation_deg?.p95, 2, '°')}</td></tr>`; }).join('')
    + '</tbody></table></div>'
    + `<p class="small-note">${LEVEL_LABELS[name]}${name === state.level ? '' : '（压力级没有做这项分解，显示证据级）'} · 逐帧 BA 位姿 · 平移误差 p95（mm）。同一帧、同一位姿，只换读出误差的点：任一点的误差 = 质心误差 + 旋转误差 × 力臂。`
    + `角点分布 rms 半径 R0 ${fmt(g.R0.corner_rms_radius_mm, 0)} mm、0907 ${fmt(g.H.corner_rms_radius_mm, 0)} mm；质心→TCP R0 ${fmt(g.R0.centroid_to_tcp_mm, 0)} mm、0907 ${fmt(g.H.centroid_to_tcp_mm, 0)} mm。</p>`;
  const bench = mech.bench_crosscheck;
  if (bench) {
    const nf = bench.noise_floor, b = bench.bench;
    html += `<p class="small-note">与 09-09 真机同口径对照（真机为手持、开阔处七台相机、数值全部在 rig 原点）：慢速 1 s 局部三次曲线噪声底，仿真 0907 ${fmt(nf.H1?.translation_rms_mm, 2)} mm / ${fmt(nf.H1?.rotation_rms_deg, 2)}°、R0 ${fmt(nf.R0?.translation_rms_mm, 2)} mm / ${fmt(nf.R0?.rotation_rms_deg, 2)}°，真机 0907 ${fmt(b.noise_translation_rms_mm, 2)} mm / ${fmt(b.noise_rotation_rms_deg, 2)}°；`
      + `facet 精修相对 BA 的位移，仿真 p50 ${fmt(bench.refinement_step_mm?.p50, 2)} mm，真机 &lt; ${fmt(b.refinement_step_below_mm, 2)} mm；每帧解出 marker 的相机数 p50，仿真 R0 ${bench.cameras_decoding?.R0?.p50 ?? '—'} 台、0907 ${bench.cameras_decoding?.H?.p50 ?? '—'} 台，真机手持 ${b.cameras_per_frame_p50} 台。</p>`;
  }
  box.innerHTML = html;
}

function renderExamples() {
  const doc = state.doc, list = doc.counterexamples || [];
  $('#l3-examples').innerHTML = list.map((c, k) => `<article class="example"><header><b>${escapeHtml(c.label)}</b><span>${HAND[c.hand] || c.hand} #${c.index} · TCP ${fmt(c.speed_mps, 2)} m/s · 遮挡 ${pct(c.occlusion)}</span></header>`
    + `<div class="example-arms">${ARMS.map(arm => `<span class="${c.arms[arm].ok ? 'ok' : 'miss'}" style="--arm:${COLORS[arm]}">${arm} ${c.arms[arm].ok ? fmt(c.arms[arm].error_mm, 2, ' mm') : '未出位姿'}</span>`).join('')}</div>`
    + `<div class="example-views">${(c.views || []).filter(v => typeof v.image === 'string' && v.image.startsWith('data:image/png;base64,')).map(v => `<figure><img src="${v.image}" alt="${v.target === 'R0' ? 'R0' : '0907'} ${escapeHtml(v.camera)} 渲染图" loading="lazy"><figcaption>${v.target === 'R0' ? 'R0' : '0907'} · ${escapeHtml(v.camera)} · 解码 ${v.decoded.length ? v.decoded.join('/') : '无'}</figcaption></figure>`).join('')}</div>`
    + `<button class="quiet" type="button" data-example="${k}">在三维回放中打开</button></article>`).join('') || '<p>报告中没有反例帧。</p>';
}

function renderEvidence() {
  const doc = state.doc;
  $('#l3-checks').innerHTML = doc.checks.map(c => `<li class="${c.ok ? '' : c.blocking ? 'fail' : 'warn'}">${c.ok ? 'PASS' : c.blocking ? 'WAIT' : 'NOTE'} · ${escapeHtml(c.label)}</li>`).join('');
  const conv = doc.convergence, bias = conv.aruco_corner_bias_noiseless || {};
  $('#l3-convergence').innerHTML = `<p class="kv"><span>超采样 ${conv.supersample}× vs ${conv.reference_supersample}×</span><b class="${conv.pass ? 'good' : 'bad'}">${fmt(conv.edge_shift_p95_px, 3)} px</b></p>`
    + `<p class="kv"><span>边缘位移阈值</span><b>${fmt(conv.threshold_px, 2)} px</b></p><p class="kv"><span>两档解码不一致</span><b>${conv.decode_mismatch} / ${conv.decodes}</b></p>`
    + `<p class="kv"><span>理想图像 ArUco 角点偏差 R0</span><b>p50 ${fmt(bias.R0?.p50_px, 2)} · p95 ${fmt(bias.R0?.p95_px, 2)} px</b></p>`
    + `<p class="kv"><span>理想图像 ArUco 角点偏差 0907</span><b>p50 ${fmt(bias.H?.p50_px, 2)} · p95 ${fmt(bias.H?.p95_px, 2)} px</b></p>`
    + '<p class="small-note">角点偏差是检测器在无噪声渲染上相对解析投影的偏差，属于检测层性质，照实计入 L2；R0 的 4.4 mm 白边紧贴黑色支架是主要来源。</p>';
  const tracking = doc.groups.gripper.levels.nominal.tracking || {};
  const lags = Object.keys(tracking.R0?.rpe || {});
  $('#l3-tracking').innerHTML = `<div class="table-wrap"><table><thead><tr><th>分支</th>${lags.map(l => `<th>RPE ${l} p95</th>`).join('')}<th>缺测 &gt;100 ms</th><th>σ 95% 椭球覆盖</th></tr></thead><tbody>`
    + ARMS.map(arm => `<tr><td style="color:${COLORS[arm]}">${arm}</td>${lags.map(l => `<td>${fmt(tracking[arm]?.rpe?.[l]?.p95_mm, 2)}</td>`).join('')}<td>${tracking[arm]?.gaps_over_100ms ?? '—'}</td><td>${pct(tracking[arm]?.sigma_ellipsoid95_coverage)}</td></tr>`).join('')
    + `</tbody></table></div><p class="small-note">RPE 为平滑后相对位移误差（毫米）。σ 覆盖远低于 95% 说明 BA 自报协方差偏乐观，与主路线图「σ 被低估」一致。H1 精修采纳率 ${pct(tracking.H1?.h1_refined_rate)}，被 12 mm 限幅丢弃 ${pct(tracking.H1?.h1_discarded_rate)}。</p>`;
  const selection = doc.selection;
  $('#l3-mounts').innerHTML = `<thead><tr><th>靶标</th><th>yaw</th><th>臂仰</th><th>覆盖率</th><th>TCP p95</th><th>选中</th></tr></thead><tbody>`
    + selection.table.map(r => { const chosen = selection.mounts[r.target][0] === r.yaw_deg && selection.mounts[r.target][1] === r.pitch_deg; return `<tr class="${chosen ? 'chosen' : ''}"><td>${r.target === 'R0' ? 'R0' : '0907 (H1)'}</td><td>${r.yaw_deg}°</td><td>${-r.pitch_deg}°</td><td>${pct(r.coverage)}</td><td>${fmt(r.p95_mm, 2, ' mm')}</td><td>${chosen ? '✓' : ''}</td></tr>`; }).join('')
    + `</tbody><caption>${escapeHtml(selection.rule)} · dev 帧数 ${selection.dev_frames}</caption>`;
  $('#l3-perturbations').innerHTML = '<thead><tr><th>来源</th><th>证据级 σ</th><th>压力 σ</th><th>作用范围</th><th>依据</th></tr></thead><tbody>'
    + Object.entries(doc.perturbations).map(([name, p]) => `<tr><td>${escapeHtml(name)}</td><td>${p.sigma_evidence}</td><td>${p.sigma_stress}</td><td>${escapeHtml(p.scope)}</td><td class="${p.evidence.startsWith('measured') ? 'measured' : 'scenario'}">${escapeHtml(p.evidence)}</td></tr>`).join('') + '</tbody>';
  const prov = doc.provenance, short = hash => String(hash).slice(0, 12);
  $('#l3-provenance').innerHTML = `<p class="kv"><span>生成时间</span><b>${escapeHtml(doc.generated_utc)}</b></p><p class="kv"><span>仓库版本</span><b>${short(prov.git?.revision)}${prov.git?.simulator_dir_dirty ? '（仿真器目录有未提交改动）' : ''}</b></p>`
    + `<p class="kv"><span>软件</span><b>Python ${prov.versions.python} · NumPy ${prov.versions.numpy} · SciPy ${prov.versions.scipy} · OpenCV ${prov.versions.opencv}</b></p>`
    + `<div class="table-wrap"><table><thead><tr><th>输入 / 代码</th><th>SHA-256</th></tr></thead><tbody>${Object.entries({...prov.inputs, ...prov.code}).map(([path, hash]) => `<tr><td>${escapeHtml(path)}</td><td><code>${short(hash)}</code></td></tr>`).join('')}</tbody></table></div>`
    + `<ul class="limits">${doc.limitations.map(l => `<li>${escapeHtml(l)}</li>`).join('')}</ul>`;
}

function errorColour(i) {
  const e = state.doc.replay.arms[state.arm].smoothed_error_mm[i];
  if (!Number.isFinite(e)) return [0.45, 0.48, 0.47];
  const t = Math.min(1, e / 6), lerp = (a, b, u) => a.map((v, k) => v + (b[k] - v) * u);
  return t < 0.5 ? lerp([0.18, 0.71, 0.49], [0.95, 0.76, 0.31], t / 0.5) : lerp([0.95, 0.76, 0.31], [0.88, 0.32, 0.23], (t - 0.5) / 0.5);
}

function renderFrameInfo() {
  const r = state.doc.replay, i = state.row;
  $('#l3-frame').value = i;
  $('#l3-frame-caption').textContent = `${HAND[r.hand[i]]} · ep ${r.episode[i]} · #${r.index[i]} · t ${fmt(r.t_s[i], 2)} s · TCP ${fmt(r.speed_mps[i], 2)} m/s${r.interpolated[i] ? ' · 插值真值' : ''}`;
  const cams = ['cam_06', 'cam_07', 'cam_08', 'cam_09', 'cam_12', 'cam_13', 'cam_14'];
  const decoded = target => r.decoded[target][i].map((n, k) => n ? `${cams[k].slice(4)}:${n}` : null).filter(Boolean).join(' ') || '无';
  $('#l3-frame-info').innerHTML = `<h3>第 ${i + 1} / ${r.hand.length} 帧</h3><table><thead><tr><th></th><th>逐帧</th><th>旋转</th><th>平滑</th></tr></thead><tbody>`
    + ARMS.map(arm => { const a = r.arms[arm]; return `<tr class="${arm === state.arm ? 'active' : ''}"><td style="color:${COLORS[arm]}">${arm}</td><td>${a.ok[i] ? fmt(a.error_mm[i], 2) : '失败'}</td><td>${a.ok[i] ? fmt(a.rotation_deg[i], 2, '°') : '—'}</td><td>${fmt(a.smoothed_error_mm[i], 2)}</td></tr>`; }).join('')
    + `</tbody></table><p class="kv"><span>遮挡 R0 / 0907</span><b>${pct(r.occlusion.R0[i])} / ${pct(r.occlusion.H[i])}</b></p>`
    + `<p class="kv"><span>R0 解码（相机:个数）</span><b>${decoded('R0')}</b></p><p class="kv"><span>0907 解码</span><b>${decoded('H')}</b></p>`;
}

function drawScene() {
  if (!state.view) return;
  state.view.render({row: state.row, arm: state.arm, mode: state.mode, cameraIndex: state.cameraIndex, orbit: state.orbit, targetCentroid: state.centroids});
  renderFrameInfo();
}

function step(time) {
  if (state.playing && state.view && time - state.lastStep > 33) {
    state.row = (state.row + 1) % state.doc.replay.hand.length;
    state.lastStep = time;
    drawScene();
  }
  requestAnimationFrame(step);
}

async function initScene() {
  const doc = state.doc, status = $('#l3-scene-status');
  try {
    const load = async path => { const response = await fetch(path); if (!response.ok) throw new Error(`无法读取 ${path}`); return response.json(); };
    const [cameras, gripper, cad, hybrid] = await Promise.all(['./inputs/cameras_0804_fisheye.json', './inputs/gripper_v2.json', `${CAD}marker_rig_20260818_cad.json`, `${CAD}hybrid_carrier_v1_20260907.json`].map(load));
    const R0 = doc.r0_geometry?.status === 'measured' ? buildRigTarget(cad, await load('./inputs/r0_layout_measured.json')) : buildRigTarget(cad);
    const H1 = buildHybridTargets(hybrid).H1;
    const centroid = target => target.anchors.map(a => a.points.reduce((s, p) => s.map((v, k) => v + p[k] / 4), [0, 0, 0]))
      .reduce((s, c, _, all) => s.map((v, k) => v + c[k] / all.length), [0, 0, 0]);
    state.centroids = {R0: centroid(R0), H: centroid(H1)};
    const view = new SceneView($('#l3-scene'), $('#l3-scene-overlay'));
    view.setStatic({cameras: cameras.cameras, gripper, targets: {R0, H1}, tcpInBox: gripper.T_box_tcp.slice(0, 3).map(row => row[3]),
      mounts: {R0: doc.groups.gripper.mounts.R0.T_box_rig, H: doc.groups.gripper.mounts.H.T_box_rig}});
    view.setReplay(doc.replay, errorColour);
    state.view = view;
    state.orbit.centre = [0, 1, 2].map(k => view.tcp.reduce((s, p) => s + p[k], 0) / view.tcp.length);
    state.cameraIndex = 1;
    $('#l3-view').innerHTML = '<option value="orbit">自由视角（拖动旋转 / 滚轮缩放）</option>' + cameras.cameras.map((c, k) => `<option value="${k}">${c.name} 鱼眼画面</option>`).join('');
    $('#l3-frame').max = String(doc.replay.hand.length - 1);
    status.textContent = `WebGL · ${doc.replay.hand.length} 帧留出轨迹 · 轨迹颜色 = 所选分支平滑后 TCP 误差（绿 0 → 红 ≥6 mm，灰 = 未出位姿）；射线绿 = 相机解码 ≥2 个 marker，琥珀 = 1 个。`;
    drawScene();
  } catch (error) {
    console.error(error);
    status.textContent = `三维回放不可用：${error.message}`;
    $('#l3-replay').classList.add('unavailable');
  }
}

function bind() {
  $('#l3-level').addEventListener('click', event => {
    const button = event.target.closest('button[data-level]'); if (!button || !state.doc?.groups.gripper.levels[button.dataset.level]) return;
    state.level = button.dataset.level;
    [...$('#l3-level').children].forEach(b => b.classList.toggle('active', b === button));
    renderCards(); drawCdf(); renderAttribution();
  });
  $('#l3-output').addEventListener('click', event => {
    const button = event.target.closest('button[data-output]'); if (!button || !state.doc) return;
    state.output = button.dataset.output;
    [...$('#l3-output').children].forEach(b => b.classList.toggle('active', b === button));
    renderCards(); drawCdf(); renderAttribution();
  });
  $('#l3-arm').addEventListener('click', event => {
    const button = event.target.closest('button[data-arm]'); if (!button || !state.view) return;
    state.arm = button.dataset.arm;
    [...$('#l3-arm').children].forEach(b => b.classList.toggle('active', b === button));
    state.view.recolour(errorColour); drawScene();
  });
  $('#l3-view').addEventListener('change', event => {
    state.mode = event.target.value === 'orbit' ? 'orbit' : 'camera';
    if (state.mode === 'camera') state.cameraIndex = Number(event.target.value);
    drawScene();
  });
  $('#l3-frame').addEventListener('input', event => { state.row = Number(event.target.value); state.playing = false; $('#l3-play').textContent = '播放'; drawScene(); });
  $('#l3-play').addEventListener('click', () => { state.playing = !state.playing; $('#l3-play').textContent = state.playing ? '暂停' : '播放'; });
  const scene = $('#l3-scene');
  scene.addEventListener('pointerdown', event => { if (state.mode !== 'orbit') return; state.drag = {x: event.clientX, y: event.clientY, a: state.orbit.azimuth, e: state.orbit.elevation}; scene.setPointerCapture(event.pointerId); });
  scene.addEventListener('pointermove', event => {
    if (!state.drag) return;
    state.orbit.azimuth = state.drag.a - (event.clientX - state.drag.x) * 0.006;
    state.orbit.elevation = Math.max(-0.2, Math.min(1.45, state.drag.e + (event.clientY - state.drag.y) * 0.005));
    drawScene();
  });
  scene.addEventListener('pointerup', () => { state.drag = null; });
  scene.addEventListener('wheel', event => { if (state.mode !== 'orbit') return; event.preventDefault(); state.orbit.distance = Math.max(0.4, Math.min(5, state.orbit.distance * (1 + Math.sign(event.deltaY) * 0.08))); drawScene(); }, {passive: false});
  $('#l3-examples').addEventListener('click', event => {
    const button = event.target.closest('button[data-example]'); if (!button || !state.view) return;
    const example = state.doc.counterexamples[Number(button.dataset.example)];
    const row = state.view.rowOf.get(`${example.hand}:${example.index}`);
    if (row === undefined) return;
    state.row = row; state.playing = false; $('#l3-play').textContent = '播放';
    const best = example.views?.find(v => v.target === (state.arm === 'R0' ? 'R0' : 'H'));
    const cameraIndex = best ? state.view.cameras.findIndex(c => c.name === best.camera) : -1;
    if (cameraIndex >= 0) { state.mode = 'camera'; state.cameraIndex = cameraIndex; $('#l3-view').value = String(cameraIndex); }
    $('#l3-replay').scrollIntoView({behavior: 'smooth', block: 'start'});
    drawScene();
  });
  let resizeTimer = null;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (!state.doc) return; drawForest(); drawCdf(); drawScene(); }, 80); });
}

async function init() {
  bind();
  requestAnimationFrame(step);
  try {
    const response = await fetch('./l3_report.json', {cache: 'no-store'});
    if (!response.ok) throw new Error('尚未生成 l3_report.json：在仓库根目录运行 python3 docs/rig0818_vs_hybrid_carrier_v1/simulator/l3_runner.py');
    state.doc = readL3Report(await response.json());
  } catch (error) {
    $('#l3-verdict').className = 'verdict open';
    $('#l3-verdict').textContent = error.message;
    $('#decision-label').textContent = '无真实输入结果';
    $('#decision-reason').textContent = error.message;
    return;
  }
  document.body.dataset.l3 = 'ready';
  const r0 = state.doc.r0_geometry;
  $('#l3-r0-text').textContent = r0?.status === 'measured' ? `实测 layout · ${r0.layout_id}` : 'CAD + 实测边长代理';
  $('#l3-r0-dot').className = r0?.status === 'measured' ? '' : 'warn';
  renderHero(); renderVerdict(); renderCards(); drawForest(); drawCdf(); renderAttribution(); renderExamples(); renderEvidence();
  await initScene();
}

init();
