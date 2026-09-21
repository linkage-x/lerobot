import test from 'node:test';
import assert from 'node:assert/strict';
import {readL3Report, decideL3, forestRows, attribution} from '../l3-report.js';

function arm(accepted, requested, p95, extra = {}) {
  return {requested, accepted, coverage: accepted / requested, p50: p95 / 2, p95, p99: p95 * 1.2, max: p95 * 1.5,
    rotation_p95: 0.3, accurate_yield: 0.9, catastrophic_rate: 0, ...extra};
}
function block(delta, ci, arms = {}) {
  const pair = {n: 100, delta, ci, a_p95: 3, b_p95: 3 - delta};
  return {arms: {R0: arm(100, 100, 3.2), H0: arm(100, 100, 3.3), H1: arm(100, 100, 2.4), ...arms},
    paired: {R0_H1: pair, R0_H0: {...pair}, H0_H1: {...pair}},
    buckets: [{family: 'speed', name: 'fast', label: '≥0.45 m/s', frames: 10, R0_H1: {...pair}}]};
}
function report({ci = [0.1, 0.5], delta = 0.3, checksOk = true} = {}) {
  const level = (sessions = 1) => ({frames: 100 / sessions, sessions, raw: block(delta, ci), smoothed: block(delta, ci)});
  return {
    schema: 'rig_target_ab_l3/v1',
    groups: {gripper: {levels: {nominal: level(), evidence: level(), stress: level()}}, socket_tcp: {levels: {evidence: level()}}},
    checks: [{id: 'a', ok: true, blocking: true, label: 'a'}, {id: 'b', ok: checksOk, blocking: true, label: 'measured R0'},
      {id: 'c', ok: false, blocking: false, label: 'assumed hand'}],
    decision: {formal: 'CONDITIONAL_GO'}
  };
}

test('a decision string in the file cannot promote the result', () => {
  const doc = readL3Report(report({ci: [0.1, 0.5]}));
  assert.equal(doc.decision.numeric, 'NO_GO');
  assert.equal(doc.decision.formal, 'NO_GO');
  assert.equal(doc.decision.robustNoGo, true);
  assert.deepEqual(doc.decision.warnings, ['assumed hand']);
});

test('an open blocking input keeps the formal decision inconclusive', () => {
  const decision = decideL3(report({ci: [0.1, 0.5], checksOk: false}));
  assert.equal(decision.numeric, 'NO_GO');
  assert.equal(decision.formal, 'INCONCLUSIVE');
  assert.deepEqual(decision.missing, ['measured R0']);
});

test('conditional go needs the lower bound at the gate and every H1 criterion', () => {
  assert.equal(decideL3(report({ci: [0.85, 1.4], delta: 1.1})).numeric, 'CONDITIONAL_GO');
  assert.equal(decideL3(report({ci: [0.5, 1.4], delta: 1.1})).numeric, 'INCONCLUSIVE');
  const low = report({ci: [0.85, 1.4], delta: 1.1});
  low.groups.gripper.levels.evidence.smoothed.arms.H1.coverage = 0.98;
  low.groups.gripper.levels.evidence.smoothed.arms.H1.accepted = 98;
  assert.equal(decideL3(low).numeric, 'INCONCLUSIVE');
  const inverted = report({ci: [0.85, 1.4], delta: 1.1});
  inverted.groups.gripper.levels.evidence.smoothed.buckets[0].R0_H1.ci = [-2, -0.2];
  assert.equal(decideL3(inverted).numeric, 'INCONCLUSIVE');
});

test('the reader rejects inconsistent denominators and reversed intervals', () => {
  const a = report();
  a.groups.gripper.levels.evidence.raw.arms.R0.requested = 99;
  assert.throws(() => readL3Report(a), /frames × sessions/);
  const b = report();
  b.groups.gripper.levels.evidence.raw.arms.H1.coverage = 0.5;
  assert.throws(() => readL3Report(b), /accepted \/ requested/);
  const c = report();
  c.groups.gripper.levels.stress.smoothed.paired.R0_H1.ci = [1, 0];
  assert.throws(() => readL3Report(c), /reversed/);
  const d = report();
  d.checks = [];
  assert.throws(() => readL3Report(d), /checklist/);
});

test('forest rows put the primary scenario, the control and the buckets on one axis', () => {
  const rows = forestRows(report());
  assert.equal(rows.filter(r => r.primary).length, 1);
  assert.equal(rows.filter(r => r.kind === 'control').length, 1);
  assert.ok(rows.some(r => r.kind === 'bucket'));
  assert.deepEqual(Object.keys(attribution(report())), ['geometry', 'edges', 'total']);
});

test('the rigid-body breakdown must carry nonnegative numbers', () => {
  const doc = report();
  const cell = {p50: 0.2, p95: 0.5};
  const arms = Object.fromEntries(['R0', 'H0', 'H1'].map(name => [name, {centroid: cell, origin: cell, common: cell, tcp: cell, rotation_deg: cell}]));
  doc.groups.gripper.mechanism = {levels: {nominal: {arms}}};
  assert.doesNotThrow(() => readL3Report(doc));
  doc.groups.gripper.mechanism.levels.nominal.arms.H1 = {...arms.H1, common: {p50: 0.2, p95: -1}};
  assert.throws(() => readL3Report(doc), /common p95/);
});
