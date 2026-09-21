import test from 'node:test';
import assert from 'node:assert/strict';
import {readL2Report,allFrameCdf} from '../l2-report.js';

const report=()=>({schema:'rig_target_ab_l2/v1',config:{seed:9},rowsByArm:Object.fromEntries(['R0','H0','H1'].map(a=>[a,Array.from({length:12},(_,i)=>({index:i,episode:i%4,accepted:i<6,translationErrorMm:i<6?i:null,rotationErrorDeg:i<6?.1:null}))]))});

test('L2 report recomputes metrics and CDF keeps missing frames in denominator',()=>{
  const doc=report();doc.metrics={R0:{p95:0}};doc.decision='GO';
  const result=readL2Report(doc);
  assert.equal(result.metrics.R0.coverage,.5);
  assert.equal(result.metrics.R0.p95,4.75);
  assert.equal(allFrameCdf(result.metrics.R0).at(-1).fraction,.5);
  assert.equal(result.decision,'INCONCLUSIVE');
});

test('L2 rejects unpaired rows, invalid errors, and finite failure penalties',()=>{
  const a=report();a.rowsByArm.H1.pop();assert.throws(()=>readL2Report(a),/same requested/);
  const b=report();b.rowsByArm.H0[0].episode=3;assert.throws(()=>readL2Report(b),/pairing/);
  const c=report();c.rowsByArm.H1[0].translationErrorMm=NaN;assert.throws(()=>readL2Report(c),/finite/);
  const d=report();d.rowsByArm.R0[7].translationErrorMm=100;assert.throws(()=>readL2Report(d),/penalty/);
});
