import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {fileURLToPath} from 'node:url';
import {
  buildHybridTargets, buildRigTarget, makeCameras, percentile, projectFisheye,
  resultForExport, rowsToCsv, runSimulation, tcpCovariance, matMul, transpose,
  makeWorldOccluders, DEFAULT_CONFIG, cross, sub, dot
} from '../simulator-core.js';

const here=fileURLToPath(new URL('.',import.meta.url));
const root=fileURLToPath(new URL('../../../../',import.meta.url));
const cad=JSON.parse(fs.readFileSync(`${root}third_party/opencv_kalibr/metrology/fixtures/cad/marker_rig_20260818_cad.json`,'utf8'));
const hybridDoc=JSON.parse(fs.readFileSync(`${root}third_party/opencv_kalibr/metrology/fixtures/cad/hybrid_carrier_v1_20260907.json`,'utf8'));

function targets(){const hybrid=buildHybridTargets(hybridDoc);return {R0:buildRigTarget(cad),...hybrid};}
function tinyConfig(extra={}){return {poses:48,episodes:4,bootstrapReplicates:32,occlusion:0.12,...extra};}

test('frozen target identities and physical sticker rotations are preserved',()=>{
  const t=targets();
  assert.deepEqual(t.R0.anchors.map(a=>a.id).sort((a,b)=>a-b),[7,12,14,16,17]);
  assert.deepEqual(t.H1.anchors.map(a=>a.id).sort((a,b)=>a-b),[30,31,32]);
  assert.equal(t.H1.edges.length,31);
  assert.deepEqual(t.H1.pasteQuadrants,{30:3,31:3,32:2});
  for(const arm of ['R0','H1'])for(const a of t[arm].anchors)assert.ok(dot(cross(sub(a.points[1],a.points[0]),sub(a.points[2],a.points[0])),a.normal)<0,'ArUco corner order must be clockwise about the outward normal');
});

test('fisheye optical axis maps to the declared principal point',()=>{
  const camera=makeCameras(0)[0];
  const point=camera.position.map((x,i)=>x+camera.forward[i]*0.7);
  const uv=projectFisheye(camera,point);
  assert.ok(uv);
  assert.ok(Math.abs(uv.u-camera.cx)<1e-9);
  assert.ok(Math.abs(uv.v-camera.cy)<1e-9);
});

test('same seed and inputs reproduce every reported metric',()=>{
  const first=runSimulation(targets(),tinyConfig({seed:701}));
  const second=runSimulation(targets(),tinyConfig({seed:701}));
  const {generated_at:firstGenerated,...firstExport}=resultForExport(first);
  const {generated_at:secondGenerated,...secondExport}=resultForExport(second);
  assert.ok(firstGenerated);
  assert.ok(secondGenerated);
  assert.deepEqual(firstExport,secondExport);
  assert.equal(first.rowsByArm.R0.length,48);
  assert.equal(first.paired.count,second.paired.count);
});

test('correlating samples on one edge cannot improve H1 FIM sigma',()=>{
  const independent=runSimulation(targets(),tinyConfig({seed:33,edgeCorrelation:0}));
  const correlated=runSimulation(targets(),tinyConfig({seed:33,edgeCorrelation:.9}));
  assert.ok(correlated.metrics.H1.l1SigmaP95>=independent.metrics.H1.l1SigmaP95);
  assert.equal(correlated.metrics.R0.l1SigmaP95,independent.metrics.R0.l1SigmaP95);
  assert.equal(correlated.metrics.H0.l1SigmaP95,independent.metrics.H0.l1SigmaP95);
});

test('proxy inputs keep G0 and G2 inconclusive even if a numerical scenario looks favorable',()=>{
  const result=runSimulation(targets(),tinyConfig({seed:91}));
  assert.equal(result.validation.simulatorChecksPass,true);
  assert.equal(result.validation.g0Pass,false);
  assert.equal(result.gates.evidenceReady,false);
  assert.equal(result.gates.decision,'INCONCLUSIVE');
});

test('exports retain the all-frame denominator and one row per arm per frame',()=>{
  const result=runSimulation(targets(),tinyConfig({seed:18}));
  const csv=rowsToCsv(result).trim().split('\n');
  assert.equal(csv.length,1+48*3);
  assert.match(csv[0],/translation_error_mm/);
  for(const arm of ['R0','H0','H1'])assert.equal(result.metrics[arm].total,48);
  assert.ok(percentile(result.metrics.R0.errors,.95)>0);
});

test('a marker_layout measured file replaces the proxy and keeps its corners',()=>{
  const proxy=buildRigTarget(cad);
  const layout={schema:'marker_layout/measured_v1',units:'m',markers:proxy.anchors.map(anchor=>({id:anchor.id,corners_rig:anchor.points}))};
  const measured=buildRigTarget(cad,layout);
  assert.equal(measured.inputStatus,'measured');
  assert.deepEqual(measured.anchors[0].points,proxy.anchors[0].points);
});

test('TCP covariance is invariant under a change of the target origin',()=>{
  const pose={R:[[1,0,0],[0,1,0],[0,0,1]],t:[0,0,0]};
  const C=Array.from({length:6},(_,i)=>Array.from({length:6},(_,j)=>i===j?(i<3?1e-6:1e-4):0));
  const shift=[.04,-.07,.02],r=[.12,0,0],S=[[0,-shift[2],shift[1]],[shift[2],0,-shift[0]],[-shift[1],shift[0],0]];
  const A=Array.from({length:6},(_,i)=>Array.from({length:6},(_,j)=>i===j?1:i<3&&j>=3?-S[i][j-3]:0));
  const shifted=matMul(matMul(A,C),transpose(A));
  const a=tcpCovariance(C,pose,r),b=tcpCovariance(shifted,pose,r.map((v,i)=>v-shift[i]));
  a.forEach((row,i)=>row.forEach((v,j)=>assert.ok(Math.abs(v-b[i][j])<1e-16)));
});

test('invalid input cannot silently produce a measured or successful run',()=>{
  const proxy=buildRigTarget(cad);
  const layout={schema:'marker_layout/measured_v1',units:'m',markers:proxy.anchors.map(a=>({id:a.id,corners_rig:a.points}))};
  assert.throws(()=>buildRigTarget(cad,{...layout,units:'inches'}),/units/);
  assert.throws(()=>buildRigTarget(cad,{...layout,markers:[layout.markers[0],...layout.markers.slice(0,4)]}),/exactly once/);
  assert.throws(()=>runSimulation(targets(),{cornerSigmaPx:0}),/positive/);
  assert.throws(()=>runSimulation(targets(),{seed:NaN}),/numeric/);
  assert.deepEqual(makeWorldOccluders({R:[[1,0,0],[0,1,0],[0,0,1]],t:[0,0,0]},0),[]);
  assert.deepEqual(makeCameras().map(c=>c.name),['cam_06','cam_07','cam_08','cam_09','cam_12','cam_13','cam_14']);
});

test('angular timing perturbation changes total error, not the static FIM',()=>{
  const a=runSimulation(targets(),tinyConfig({seed:32,angularSpeedDegS:0}));
  const b=runSimulation(targets(),tinyConfig({seed:32,angularSpeedDegS:160}));
  assert.equal(a.metrics.R0.l1SigmaP95,b.metrics.R0.l1SigmaP95);
  assert.notEqual(a.metrics.R0.rotationP95,b.metrics.R0.rotationP95);
  assert.equal(DEFAULT_CONFIG.installTranslationSigmaMm,.6);
});
