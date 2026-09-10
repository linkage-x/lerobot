// Shared numerical scene contract for the independent OpenCV image runner.
import fs from 'node:fs';
import {buildRigTarget, buildHybridTargets, DEFAULT_CONFIG, generateTrajectory,
  makeCameras, makeWorldOccluders} from './simulator-core.js';

const read = path => JSON.parse(fs.readFileSync(new URL(path, import.meta.url), 'utf8'));
const base = '../../../third_party/opencv_kalibr/metrology/fixtures/cad/';
const config = {...DEFAULT_CONFIG, ...JSON.parse(process.argv[2] || '{}')};
const measured = process.argv[3] ? JSON.parse(fs.readFileSync(process.argv[3], 'utf8')) : null;
const rig = buildRigTarget(read(`${base}marker_rig_20260818_cad.json`), measured);
const hybrid = read(`${base}hybrid_carrier_v1_20260907.json`);
const poses = generateTrajectory(config);
process.stdout.write(JSON.stringify({schema:'rig_target_ab_world/v1', config,
  rig, hybrid, cameras:makeCameras(config.cameraDropout),
  poses:poses.map(p=>({...p,occluders:makeWorldOccluders(p,config.occlusion,p.phase)}))}));
