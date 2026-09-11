/* Reader for the real-input L2+L3 report. The decision is recomputed here from
 * the numbers and the checklist; a decision string in the file is ignored. */

export const ARMS = ['R0', 'H0', 'H1'];
export const GATE_MM = 0.8;
export const LEVEL_LABELS = {nominal: '名义（仅图像噪声）', evidence: '证据级扰动', stress: '2× 压力扰动'};
export const OUTPUT_LABELS = {raw: '逐帧 BA', smoothed: '离线平滑（数据集标签）'};

const finite = value => typeof value === 'number' && Number.isFinite(value);
const nullableFinite = value => value === null || finite(value);

function checkArm(where, arm, requested) {
  if (!arm || typeof arm !== 'object') throw new Error(`${where}: missing arm summary`);
  if (arm.requested !== requested) throw new Error(`${where}: requested frames must equal frames × sessions`);
  if (!Number.isInteger(arm.accepted) || arm.accepted < 0 || arm.accepted > requested) throw new Error(`${where}: invalid accepted count`);
  if (requested && !(finite(arm.coverage) && Math.abs(arm.coverage - arm.accepted / requested) < 1e-9)) throw new Error(`${where}: coverage must be accepted / requested`);
  for (const key of ['p50', 'p95', 'p99', 'max', 'rotation_p95']) {
    if (!nullableFinite(arm[key]) || (finite(arm[key]) && arm[key] < 0)) throw new Error(`${where}: ${key} must be a nonnegative number or null`);
    if (arm.accepted === 0 && arm[key] !== null) throw new Error(`${where}: an arm without accepted frames cannot report ${key}`);
  }
}

function checkPaired(where, paired) {
  if (!paired || !Array.isArray(paired.ci) || paired.ci.length !== 2) throw new Error(`${where}: paired delta needs a two-sided interval`);
  if (!paired.ci.every(nullableFinite) || !nullableFinite(paired.delta)) throw new Error(`${where}: paired delta must be numeric or null`);
  if (finite(paired.ci[0]) && finite(paired.ci[1]) && paired.ci[0] > paired.ci[1]) throw new Error(`${where}: interval bounds are reversed`);
}

export function readL3Report(doc) {
  if (doc?.schema !== 'rig_target_ab_l3/v1') throw new Error('Expected rig_target_ab_l3/v1 report');
  if (!doc.groups?.gripper?.levels?.evidence) throw new Error('L3 report needs the gripper group with an evidence level');
  for (const [groupName, group] of Object.entries(doc.groups)) {
    for (const [levelName, level] of Object.entries(group.levels || {})) {
      if (!Number.isInteger(level.frames) || !Number.isInteger(level.sessions) || level.frames < 1 || level.sessions < 1) throw new Error(`${groupName}/${levelName}: invalid frame or session count`);
      for (const output of ['raw', 'smoothed']) {
        const block = level[output], where = `${groupName}/${levelName}/${output}`;
        if (!block) throw new Error(`${where}: missing`);
        for (const arm of ARMS) checkArm(`${where}/${arm}`, block.arms?.[arm], level.frames * level.sessions);
        for (const pair of ['R0_H1', 'R0_H0', 'H0_H1']) checkPaired(`${where}/${pair}`, block.paired?.[pair]);
        for (const bucket of block.buckets || []) checkPaired(`${where}/${bucket.name}`, bucket.R0_H1);
      }
    }
    for (const [levelName, level] of Object.entries(group.mechanism?.levels || {})) {
      for (const arm of ARMS) {
        for (const key of ['centroid', 'origin', 'common', 'tcp', 'rotation_deg']) {
          const value = level.arms?.[arm]?.[key]?.p95;
          if (!nullableFinite(value) || (finite(value) && value < 0)) throw new Error(`${groupName}/mechanism/${levelName}/${arm}: ${key} p95 must be a nonnegative number or null`);
        }
      }
    }
  }
  if (!Array.isArray(doc.checks) || !doc.checks.some(check => check.blocking)) throw new Error('L3 report needs a G0 checklist with blocking items');
  if (doc.replay) {
    const n = doc.replay.hand?.length;
    if (!Number.isInteger(n) || n < 1) throw new Error('replay table is empty');
    for (const arm of ARMS) if (doc.replay.arms?.[arm]?.ok?.length !== n) throw new Error('replay rows must align across arms');
    if (doc.replay.box_pose7?.length !== n) throw new Error('replay poses must align with rows');
  }
  return {...doc, decision: decideL3(doc)};
}

export function decideL3(doc) {
  const levels = doc.groups.gripper.levels, primary = levels.evidence.smoothed;
  const delta = primary.paired.R0_H1, [lo, hi] = delta.ci, r0 = primary.arms.R0, h1 = primary.arms.H1;
  const inversions = (primary.buckets || []).filter(b => ['speed', 'occlusion'].includes(b.family) && finite(b.R0_H1?.ci?.[1]) && b.R0_H1.ci[1] < 0).map(b => b.label);
  const uppers = Object.values(levels).flatMap(level => ['raw', 'smoothed'].map(output => level[output].paired.R0_H1.ci[1]));
  const numericGo = finite(lo) && lo >= GATE_MM && finite(h1.p95) && h1.p95 <= 3 && finite(h1.rotation_p95) && h1.rotation_p95 <= 0.5
    && h1.coverage > 0.99 && h1.coverage >= r0.coverage && h1.catastrophic_rate <= r0.catastrophic_rate && !inversions.length;
  const numericNoGo = finite(hi) && hi < GATE_MM;
  const numeric = numericGo ? 'CONDITIONAL_GO' : numericNoGo ? 'NO_GO' : 'INCONCLUSIVE';
  const blocking = doc.checks.filter(check => check.blocking);
  const g0Pass = blocking.every(check => check.ok === true);
  return {
    formal: g0Pass ? numeric : 'INCONCLUSIVE', numeric, g0Pass,
    missing: blocking.filter(check => check.ok !== true).map(check => check.label),
    warnings: doc.checks.filter(check => !check.blocking && check.ok !== true).map(check => check.label),
    robustNoGo: uppers.length > 0 && uppers.every(u => finite(u) && u < GATE_MM),
    inversions, primary: {delta: delta.delta, ci: delta.ci, n: delta.n, r0, h1}
  };
}

export function forestRows(doc) {
  const rows = [];
  const gripper = doc.groups.gripper.levels;
  for (const level of ['nominal', 'evidence', 'stress']) {
    if (!gripper[level]) continue;
    for (const output of ['raw', 'smoothed']) {
      const p = gripper[level][output].paired.R0_H1;
      rows.push({kind: 'scenario', primary: level === 'evidence' && output === 'smoothed', label: `${LEVEL_LABELS[level]} · ${OUTPUT_LABELS[output]}`, delta: p.delta, ci: p.ci, n: p.n});
    }
  }
  const socket = doc.groups.socket_tcp?.levels?.evidence?.smoothed?.paired?.R0_H1;
  if (socket) rows.push({kind: 'control', label: '对照：球窝放在 TCP（朝向同主实验，无本夹爪）· 证据级 · 平滑', delta: socket.delta, ci: socket.ci, n: socket.n});
  for (const bucket of gripper.evidence.smoothed.buckets || []) {
    if (bucket.family === 'hand') continue;
    rows.push({kind: 'bucket', family: bucket.family, label: `分层 · ${bucket.label}`, delta: bucket.R0_H1.delta, ci: bucket.R0_H1.ci, n: bucket.R0_H1.n});
  }
  return rows;
}

export function attribution(doc, level = 'evidence', output = 'smoothed') {
  const paired = doc.groups.gripper.levels[level][output].paired;
  return {geometry: paired.R0_H0, edges: paired.H0_H1, total: paired.R0_H1};
}
