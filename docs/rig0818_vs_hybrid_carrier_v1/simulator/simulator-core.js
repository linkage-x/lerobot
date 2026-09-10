/* Deterministic M0/M1/MC-lite kernel. No DOM and no external dependencies. */

export const ARM_ORDER = ['R0', 'H0', 'H1'];
export const ARM_COLORS = {R0: '#e15b3d', H0: '#e2a72e', H1: '#1d826b'};
export const DEFAULT_CONFIG = Object.freeze({
  seed: 20260910,
  poses: 240,
  episodes: 12,
  occlusion: 0.22,
  linearSpeedMps: 0.28,
  angularSpeedDegS: 55,
  exposureMs: 5,
  syncSigmaMs: 2.2,
  cornerSigmaPx: 0.22,
  edgeSigmaPx: 0.42,
  edgeCorrelation: 0.55,
  tcpOffsetMm: 120,
  cameraDropout: 0,
  minMarkerPixels: 18,
  cameraTranslationSigmaMm: 1.1,
  cameraRotationSigmaDeg: 0.10,
  installTranslationSigmaMm: 0.60,
  installRotationSigmaDeg: 0.18,
  bootstrapReplicates: 320,
  rigInputStatus: 'proxy',
  cameraInputStatus: 'synthetic',
  trajectoryInputStatus: 'synthetic',
  l2InputStatus: 'model'
});

const DEG = Math.PI / 180;
const EPS = 1e-12;

export function clamp(value, lo, hi) { return Math.max(lo, Math.min(hi, value)); }
export function add(a, b) { return a.map((x, i) => x + b[i]); }
export function sub(a, b) { return a.map((x, i) => x - b[i]); }
export function scale(a, s) { return a.map(x => x * s); }
export function dot(a, b) { return a.reduce((sum, x, i) => sum + x * b[i], 0); }
export function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
export function norm(a) { return Math.sqrt(dot(a, a)); }
export function normalize(a) { const n = norm(a); return n < EPS ? a.map(() => 0) : scale(a, 1/n); }
export function mat3Vec(M, v) { return M.map(row => dot(row, v)); }
export function transpose(M) { return M[0].map((_, j) => M.map(row => row[j])); }
export function matMul(A, B) {
  const BT = transpose(B);
  return A.map(row => BT.map(col => dot(row, col)));
}

function identity(n) { return Array.from({length:n}, (_, i) => Array.from({length:n}, (_, j) => i === j ? 1 : 0)); }
function zeros(rows, cols) { return Array.from({length:rows}, () => Array(cols).fill(0)); }

export function rotationFromEuler(rx, ry, rz) {
  const [sx,cx,sy,cy,sz,cz] = [Math.sin(rx),Math.cos(rx),Math.sin(ry),Math.cos(ry),Math.sin(rz),Math.cos(rz)];
  return [
    [cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
    [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
    [-sy, cy*sx, cy*cx]
  ];
}

function axisRotation(axis, angle) {
  const [x,y,z] = axis;
  const c = Math.cos(angle), s = Math.sin(angle), q = 1-c;
  return [[c+x*x*q,x*y*q-z*s,x*z*q+y*s],[y*x*q+z*s,c+y*y*q,y*z*q-x*s],[z*x*q-y*s,z*y*q+x*s,c+z*z*q]];
}

export function transformPoint(pose, point) { return add(mat3Vec(pose.R, point), pose.t); }
export function rotateVector(pose, vector) { return mat3Vec(pose.R, vector); }

function hashSeed(seed, salt) {
  let h = (Number(seed) >>> 0) ^ 0x9e3779b9;
  for (let i=0; i<String(salt).length; i++) {
    h ^= String(salt).charCodeAt(i); h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export class Rng {
  constructor(seed, salt='') { this.state = hashSeed(seed, salt) || 1; this.spare = null; }
  uniform() {
    let t = this.state += 0x6D2B79F5;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
  normal() {
    if (this.spare !== null) { const z=this.spare; this.spare=null; return z; }
    const u = Math.max(this.uniform(), 1e-12), v = this.uniform();
    const mag = Math.sqrt(-2*Math.log(u));
    this.spare = mag*Math.sin(2*Math.PI*v);
    return mag*Math.cos(2*Math.PI*v);
  }
  normals(n) { return Array.from({length:n}, () => this.normal()); }
}

export function percentile(values, q) {
  const sorted = values.filter(Number.isFinite).slice().sort((a,b)=>a-b);
  if (!sorted.length) return null;
  const index = clamp(q,0,1)*(sorted.length-1), lo=Math.floor(index), hi=Math.ceil(index);
  return sorted[lo] + (sorted[hi]-sorted[lo])*(index-lo);
}

function symmetricEigenvalues(matrix, maxSweeps=80) {
  const A = matrix.map(row => row.slice()), n=A.length;
  for (let sweep=0; sweep<maxSweeps; sweep++) {
    let p=0,q=1,best=0;
    for (let i=0;i<n;i++) for (let j=i+1;j<n;j++) if (Math.abs(A[i][j])>best) {best=Math.abs(A[i][j]);p=i;q=j;}
    if (best < 1e-13) break;
    const angle=0.5*Math.atan2(2*A[p][q], A[q][q]-A[p][p]), c=Math.cos(angle), s=Math.sin(angle);
    for (let k=0;k<n;k++) if (k!==p && k!==q) {
      const akp=A[k][p], akq=A[k][q];
      A[k][p]=A[p][k]=c*akp-s*akq;
      A[k][q]=A[q][k]=s*akp+c*akq;
    }
    const app=A[p][p], aqq=A[q][q], apq=A[p][q];
    A[p][p]=c*c*app-2*s*c*apq+s*s*aqq;
    A[q][q]=s*s*app+2*s*c*apq+c*c*aqq;
    A[p][q]=A[q][p]=0;
  }
  return A.map((row,i)=>row[i]).sort((a,b)=>a-b);
}

function invertMatrix(matrix) {
  const n=matrix.length, aug=matrix.map((row,i)=>[...row,...identity(n)[i]]);
  for (let col=0;col<n;col++) {
    let pivot=col;
    for (let row=col+1;row<n;row++) if (Math.abs(aug[row][col])>Math.abs(aug[pivot][col])) pivot=row;
    if (Math.abs(aug[pivot][col])<1e-16) return null;
    [aug[col],aug[pivot]]=[aug[pivot],aug[col]];
    const d=aug[col][col]; for (let j=0;j<2*n;j++) aug[col][j]/=d;
    for (let row=0;row<n;row++) if (row!==col) {
      const factor=aug[row][col];
      for (let j=0;j<2*n;j++) aug[row][j]-=factor*aug[col][j];
    }
  }
  return aug.map(row=>row.slice(n));
}

function cholesky(matrix) {
  const n=matrix.length, L=zeros(n,n);
  for (let i=0;i<n;i++) for (let j=0;j<=i;j++) {
    let sum=matrix[i][j]; for (let k=0;k<j;k++) sum-=L[i][k]*L[j][k];
    if (i===j) { if (sum<=1e-18 || !Number.isFinite(sum)) return null; L[i][j]=Math.sqrt(sum); }
    else L[i][j]=sum/L[j][j];
  }
  return L;
}

function addOuter(F, row, weight) {
  for (let i=0;i<row.length;i++) for (let j=0;j<row.length;j++) F[i][j]+=weight*row[i]*row[j];
}

function scalePointRing(points, centre, targetEdgeMm, sourceEdgeMm) {
  const factor=targetEdgeMm/sourceEdgeMm;
  return points.map(p=>p.map((x,i)=>(centre[i]+(x-centre[i])*factor)*1e-3));
}

function triangleFan(polygon) {
  const out=[]; for (let i=1;i+1<polygon.length;i++) out.push([polygon[0],polygon[i],polygon[i+1]]); return out;
}

export function buildRigTarget(cad, measuredLayout=null) {
  const map={7:4,12:2,14:1,16:0,17:3};
  const measuredEdges={7:60.88,12:60.63,14:60.55,16:60.56,17:61.12};
  let anchors=[];
  if (measuredLayout) {
    if(measuredLayout.schema!=='marker_layout/measured_v1')throw new Error('Expected marker_layout/measured_v1 schema');
    const unit=String(measuredLayout.units||'m').toLowerCase();
    if(!['m','mm','cm'].includes(unit))throw new Error('Layout units must be m, mm or cm');
    const factor=unit==='mm'?1e-3:unit==='cm'?1e-2:1;
    const ids=(measuredLayout.markers||[]).filter(m=>Object.hasOwn(map,Number(m.id))).map(m=>Number(m.id));
    if(new Set(ids).size!==5||ids.length!==5)throw new Error('Layout requires each R0 ID exactly once');
    anchors=(measuredLayout.markers||[]).filter(m=>Object.hasOwn(map,Number(m.id))).map(marker=>{
      if(!Array.isArray(marker.corners_rig)||marker.corners_rig.length!==4||!marker.corners_rig.every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite)))throw new Error(`Invalid corners for marker ${marker.id}`);
      const points=marker.corners_rig.map(p=>p.map(x=>x*factor));
      const raw=cross(sub(points[1],points[0]),sub(points[2],points[0]));
      if(norm(raw)<1e-8)throw new Error(`Degenerate marker ${marker.id}`);
      return {id:Number(marker.id),name:`marker_${marker.id}`,points,normal:scale(normalize(raw),-1)};
    });
    if (anchors.length!==5) throw new Error(`measured R0 layout must contain ids 7/12/14/16/17; found ${anchors.length}`);
  } else {
    anchors=Object.entries(map).map(([id,index])=>{
      const plate=cad.plates[index], centre=plate.centre_mm;
      // CAD plate loops are CCW about the outward normal; ArUco corners are CW.
      return {id:Number(id),name:`marker_${id}`,points:scalePointRing(plate.corners_mm,centre,measuredEdges[id],plate.edge_mm).reverse(),normal:plate.normal.slice()};
    });
  }
  const plates=measuredLayout?anchors.map(anchor=>{
    const centre=scale(anchor.points.reduce((a,p)=>add(a,p),[0,0,0]),.25),edge=norm(sub(anchor.points[1],anchor.points[0]));
    return {name:`plate_${anchor.id}`,points:anchor.points.map(p=>add(centre,scale(sub(p,centre),.0696/edge))),normal:anchor.normal,color:'#9ea9a2'};
  }):cad.plates.map((plate,index)=>({name:`plate_${index}`,points:plate.corners_mm.map(p=>p.map(x=>x*1e-3)),normal:plate.normal.slice(),color:'#9ea9a2'}));
  const displayPolygons=[...plates,...anchors.map(a=>({name:a.name,points:a.points,normal:a.normal,color:'#f2f0e4',anchor:true}))];
  const triangles=plates.flatMap(p=>triangleFan(p.points));
  return {id:'R0',family:'rig',label:'rig0818',anchors,edges:[],faceNormals:new Map(),triangles,displayPolygons,inputStatus:measuredLayout?'measured':'proxy'};
}

export function buildHybridTargets(doc) {
  const mm=p=>p.map(x=>x*1e-3);
  const anchors=doc.anchors.map(anchor=>({id:Number(anchor.marker_id),name:anchor.name,points:anchor.marker_corners.map(mm),pad:anchor.pad_corners.map(mm),normal:anchor.normal.slice(),pasteQuadrant:Number(anchor.paste_quadrant)}));
  const colorMap={red:'#d85246',green:'#3b9b72',blue:'#4e75b9',yellow:'#e1ad3d',white:'#f5f0df'};
  const facets=doc.facets.map(f=>({name:f.name,points:f.polygon.map(mm),normal:f.normal.slice(),color:colorMap[f.colour]||'#71827a'}));
  const faceNormals=new Map(); anchors.forEach(a=>faceNormals.set(a.name,a.normal)); facets.forEach(f=>faceNormals.set(f.name,f.normal));
  const edges=doc.edges.filter(e=>e.use_for_pose!==false).map(e=>({name:`${e.face_a}:${e.face_b}:${e.step_edge_id}`,p0:mm(e.p0),p1:mm(e.p1),faceA:e.face_a,faceB:e.face_b}));
  const triangles=doc.occluders?.triangles?doc.occluders.triangles.map(tri=>tri.map(mm)):facets.flatMap(f=>triangleFan(f.points));
  const displayPolygons=[...facets,...anchors.map(a=>({name:a.name,points:a.pad,normal:a.normal,color:'#f4f0df',anchor:true,markerPoints:a.points}))];
  const common={family:'hybrid',label:'Hybrid Carrier V1',anchors,edges,faceNormals,triangles,displayPolygons,inputStatus:'descriptor',pasteQuadrants:Object.fromEntries(anchors.map(a=>[a.id,a.pasteQuadrant]))};
  return {
    H0:{...common,id:'H0',label:'0907 anchors-only',edges:[]},
    H1:{...common,id:'H1',label:'0907 full',edges}
  };
}

function lookAt(position, target) {
  const forward=normalize(sub(target,position));
  const upHint=Math.abs(forward[2])>.92?[0,1,0]:[0,0,1];
  const right=normalize(cross(upHint,forward));
  const down=normalize(cross(forward,right));
  return {right,down,forward};
}

export function makeCameras(dropout=0) {
  const centre=[0,0,0.27];
  const positions=[[.72,-.54,.56],[.78,.05,.48],[.65,.62,.60],[.02,.78,.55],[-.58,.54,.62],[-.70,-.08,.53],[-.36,-.66,.70]];
  return positions.slice(0,Math.max(1,7-Number(dropout))).map((position,index)=>({
    name:['cam_06','cam_07','cam_08','cam_09','cam_12','cam_13','cam_14'][index],position,...lookAt(position,centre),width:1280,height:1024,
    fx:438,fy:438,cx:640,cy:512,D:[-0.012,0.0018,-0.00022,0.00001],near:0.05,maxTheta:1.48
  }));
}

export function projectFisheye(camera, pointWorld) {
  const d=sub(pointWorld,camera.position), x=dot(d,camera.right), y=dot(d,camera.down), z=dot(d,camera.forward);
  if (z<=camera.near) return null;
  const r=Math.hypot(x,y), theta=Math.atan2(r,z);
  if (theta>camera.maxTheta) return null;
  const t2=theta*theta, [k1,k2,k3,k4]=camera.D;
  const td=theta*(1+k1*t2+k2*t2*t2+k3*t2*t2*t2+k4*t2*t2*t2*t2);
  const radial=r<EPS?1:td/r;
  return {u:camera.fx*x*radial+camera.cx,v:camera.fy*y*radial+camera.cy,z,theta};
}

function inFrame(camera, uv, margin=3) { return uv && uv.u>=margin && uv.v>=margin && uv.u<camera.width-margin && uv.v<camera.height-margin; }

function segmentSphereHit(origin, end, sphere) {
  const d=sub(end,origin), f=sub(origin,sphere.c), a=dot(d,d), b=2*dot(f,d), c=dot(f,f)-sphere.r*sphere.r;
  const disc=b*b-4*a*c; if (disc<0) return false;
  const root=Math.sqrt(disc), t1=(-b-root)/(2*a), t2=(-b+root)/(2*a);
  return (t1>1e-4&&t1<.985)||(t2>1e-4&&t2<.985);
}

function segmentTriangleHit(origin, end, tri) {
  const dir=sub(end,origin), e1=sub(tri[1],tri[0]), e2=sub(tri[2],tri[0]), h=cross(dir,e2), a=dot(e1,h);
  if (Math.abs(a)<1e-12) return false;
  const f=1/a,s=sub(origin,tri[0]),u=f*dot(s,h); if (u<0||u>1) return false;
  const q=cross(s,e1),v=f*dot(dir,q); if (v<0||u+v>1) return false;
  const t=f*dot(e2,q); return t>1e-4&&t<.985;
}

export function makeWorldOccluders(pose, severity, phase=0) {
  if(severity<=0)return [];
  const s=clamp(severity,0,.9), wave=Math.sin(phase*2*Math.PI);
  const local=[
    {c:[.105,.105-.085*s+.018*wave,.045],r:.011+.052*s},
    {c:[.15,-.11+.105*s,.018],r:.010+.038*s},
    {c:[-.025,.045*wave,-.045],r:.013+.018*s}
  ];
  return local.map(item=>({c:transformPoint(pose,item.c),r:item.r}));
}

function visiblePoint(camera, worldPoint, worldTriangles, worldOccluders) {
  if (!inFrame(camera,projectFisheye(camera,worldPoint))) return false;
  for (const sphere of worldOccluders) if (segmentSphereHit(camera.position,worldPoint,sphere)) return false;
  for (const tri of worldTriangles) if (segmentTriangleHit(camera.position,worldPoint,tri)) return false;
  return true;
}

function projectedJacobian(camera, pose, pointLocal) {
  const base=projectFisheye(camera,transformPoint(pose,pointLocal)); if (!base) return null;
  const epsT=1e-5,epsR=1e-5,Ju=[],Jv=[];
  for (let k=0;k<6;k++) {
    let perturbed;
    if (k<3) { perturbed={R:pose.R,t:pose.t.slice()}; perturbed.t[k]+=epsT; }
    else { const axis=[0,0,0];axis[k-3]=1;perturbed={R:matMul(axisRotation(axis,epsR),pose.R),t:pose.t}; }
    const uv=projectFisheye(camera,transformPoint(perturbed,pointLocal)); if (!uv) return null;
    const eps=k<3?epsT:epsR;Ju.push((uv.u-base.u)/eps);Jv.push((uv.v-base.v)/eps);
  }
  return {uv:base,Ju,Jv};
}

function targetWorldTriangles(target, pose) { return target.triangles.map(tri=>tri.map(p=>transformPoint(pose,p))); }

export function tcpCovariance(poseCovariance, pose, tcpOffsetMm) {
  const r=rotateVector(pose,Array.isArray(tcpOffsetMm)?tcpOffsetMm:[Number(tcpOffsetMm)*1e-3,0,0]);
  const S=[[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]];
  const H=[
    [1,0,0,-S[0][0],-S[0][1],-S[0][2]],
    [0,1,0,-S[1][0],-S[1][1],-S[1][2]],
    [0,0,1,-S[2][0],-S[2][1],-S[2][2]]
  ];
  return matMul(matMul(H,poseCovariance),transpose(H));
}

export function analyzePose(target, pose, cameras, config, phase=0) {
  const F=zeros(6,6), worldTriangles=targetWorldTriangles(target,pose), occluders=makeWorldOccluders(pose,config.occlusion,phase);
  const anchorIds=new Set(), cameraNames=new Set(), visibleAnchorViews=[], visibleEdges=[];
  let cornerRows=0,edgeRows=0,minProjectedEdge=Infinity;
  for (const camera of cameras) {
    for (const anchor of target.anchors) {
      const centre=scale(anchor.points.reduce((acc,p)=>add(acc,p),[0,0,0]),1/anchor.points.length);
      const centreWorld=transformPoint(pose,centre), normalWorld=rotateVector(pose,anchor.normal), toCamera=normalize(sub(camera.position,centreWorld));
      if (dot(normalWorld,toCamera)<0.08) continue;
      const projections=anchor.points.map(p=>projectFisheye(camera,transformPoint(pose,p)));
      if (!projections.every(uv=>inFrame(camera,uv,4))) continue;
      const edgePixels=[];for(let i=0;i<4;i++)edgePixels.push(Math.hypot(projections[(i+1)%4].u-projections[i].u,projections[(i+1)%4].v-projections[i].v));
      const apparent=Math.min(...edgePixels); if (apparent<config.minMarkerPixels) continue;
      if (!anchor.points.every(p=>visiblePoint(camera,transformPoint(pose,p),worldTriangles,occluders))) continue;
      minProjectedEdge=Math.min(minProjectedEdge,apparent); anchorIds.add(anchor.id);cameraNames.add(camera.name);
      visibleAnchorViews.push({camera:camera.name,id:anchor.id,apparentPx:apparent,centre:centreWorld});
      for (const point of anchor.points) {
        const J=projectedJacobian(camera,pose,point); if (!J) continue;
        addOuter(F,J.Ju,1/(config.cornerSigmaPx**2));addOuter(F,J.Jv,1/(config.cornerSigmaPx**2));cornerRows+=2;
      }
    }
    for (const edge of target.edges) {
      const normals=[target.faceNormals.get(edge.faceA),target.faceNormals.get(edge.faceB)].filter(Boolean).map(n=>rotateVector(pose,n));
      const mid=scale(add(edge.p0,edge.p1),.5),midWorld=transformPoint(pose,mid),toCamera=normalize(sub(camera.position,midWorld));
      if (!normals.some(n=>dot(n,toCamera)>0.05)) continue;
      const p0w=transformPoint(pose,edge.p0),p1w=transformPoint(pose,edge.p1),uv0=projectFisheye(camera,p0w),uv1=projectFisheye(camera,p1w);
      if (!inFrame(camera,uv0,3)||!inFrame(camera,uv1,3)||!visiblePoint(camera,midWorld,worldTriangles,occluders)) continue;
      const du=uv1.u-uv0.u,dv=uv1.v-uv0.v,length=Math.hypot(du,dv);if(length<10)continue;
      const sampleT=[.2,.5,.8],n=sampleT.length;
      const nEff=n/(1+(n-1)*config.edgeCorrelation),weightEach=(nEff/n)/(config.edgeSigmaPx**2);
      let used=0;
      for (const alpha of sampleT) {
        const point=add(scale(edge.p0,1-alpha),scale(edge.p1,alpha)),J=projectedJacobian(camera,pose,point);if(!J)continue;
        if(!visiblePoint(camera,transformPoint(pose,point),worldTriangles,occluders))continue;
        // Fisheye edges need the local image tangent, not the endpoint chord.
        const near=projectFisheye(camera,transformPoint(pose,add(point,scale(sub(edge.p1,edge.p0),1e-4))));
        if(!near)continue;
        const tangent=[near.u-J.uv.u,near.v-J.uv.v],tangentLength=Math.hypot(...tangent);
        if(tangentLength<1e-12)continue;
        const imageNormal=[-tangent[1]/tangentLength,tangent[0]/tangentLength];
        const row=J.Ju.map((x,i)=>imageNormal[0]*x+imageNormal[1]*J.Jv[i]);addOuter(F,row,weightEach);edgeRows++;used++;
      }
      if(used){cameraNames.add(camera.name);visibleEdges.push({camera:camera.name,name:edge.name,lengthPx:length,mid:midWorld});}
    }
  }
  // Rank and condition are evaluated in dimensionless local coordinates:
  // 0.1 m translation and 1 rad rotation. The raw mixed-unit FIM condition
  // changes if metres are merely rewritten as millimetres.
  const parameterScale=[.1,.1,.1,1,1,1];
  const scaledF=F.map((row,i)=>row.map((value,j)=>value*parameterScale[i]*parameterScale[j]));
  const eigen=symmetricEigenvalues(scaledF),largest=eigen[eigen.length-1]||0,smallest=eigen[0]||0,condition=smallest>0?largest/smallest:Infinity;
  const enoughAnchors=anchorIds.size>=2&&cameraNames.size>=2&&cornerRows>=16;
  const observable=enoughAnchors&&smallest>largest*1e-12&&Number.isFinite(condition);
  const covariance=observable?invertMatrix(F):null;
  const tcpCov=covariance?tcpCovariance(covariance,pose,config.tcpOffsetMm):null;
  const tcpEig=tcpCov?symmetricEigenvalues(tcpCov):[];
  return {
    observable,covariance,condition,anchorCount:anchorIds.size,cameraCount:cameraNames.size,cornerRows,edgeRows,
    minProjectedEdgePx:Number.isFinite(minProjectedEdge)?minProjectedEdge:null,
    sigmaTcpWorstMm:tcpEig.length?Math.sqrt(Math.max(...tcpEig))*1000:null,
    visibleAnchorViews,visibleEdges,occluders
  };
}

export function generateTrajectory(config) {
  const count=Math.max(1,Math.round(config.poses)),episodes=clamp(Math.round(config.episodes),1,count),poses=[];
  for(let i=0;i<count;i++){
    const episode=Math.min(episodes-1,Math.floor(i*episodes/count)),local=(i-episode*count/episodes)/(count/episodes),u=clamp(local,0,1),phase=(episode+u)/episodes;
    const a=2*Math.PI*phase,b=2*Math.PI*u;
    const t=[.075*Math.sin(a*2.3)+.025*Math.sin(b),.105*Math.sin(a*1.7+.6),.25+.065*Math.sin(a*2.0-.4)+.018*Math.cos(b*2)];
    const R=rotationFromEuler(.25*Math.sin(a*2.1),-.30+.48*Math.sin(a*1.35+.8),.62*Math.sin(a*1.8-.5));
    const direction=normalize([Math.cos(a*2.3)+.25*Math.cos(b),.75*Math.cos(a*1.7+.6),.55*Math.cos(a*2.0-.4)]);
    const angularDirection=normalize([.4*Math.cos(a*2.1),.7*Math.cos(a*1.35+.8),Math.cos(a*1.8-.5)]);
    poses.push({R,t,episode,index:i,phase,velocity:scale(direction,config.linearSpeedMps),angularVelocity:scale(angularDirection,config.angularSpeedDegS*DEG)});
  }
  return poses;
}

function configWithDefaults(input) {
  const c={...DEFAULT_CONFIG,...input};
  for(const [key,value] of Object.entries(DEFAULT_CONFIG))if(typeof value==='number'&&!Number.isFinite(Number(c[key])))throw new Error(`Invalid numeric parameter: ${key}`);
  for(const key of ['cornerSigmaPx','edgeSigmaPx'])if(Number(c[key])<=0)throw new Error(`${key} must be positive`);
  for(const key of ['linearSpeedMps','angularSpeedDegS','exposureMs','syncSigmaMs','cameraTranslationSigmaMm','cameraRotationSigmaDeg','installTranslationSigmaMm','installRotationSigmaDeg'])if(Number(c[key])<0)throw new Error(`${key} must be nonnegative`);
  c.poses=clamp(Math.round(Number(c.poses)||240),48,1200);c.episodes=clamp(Math.round(Number(c.episodes)||12),4,Math.min(30,c.poses));
  c.occlusion=clamp(Number(c.occlusion),0,.75);c.edgeCorrelation=clamp(Number(c.edgeCorrelation),0,.95);c.cameraDropout=clamp(Math.round(Number(c.cameraDropout)),0,6);
  return c;
}

function matVec(M,v){return M.map(row=>dot(row,v));}
function vecSum(...items){return items.reduce((acc,item)=>add(acc,item),[0,0,0]);}
function randomVector(rng,sigma){return rng.normals(3).map(x=>x*sigma);}

function detectorProbability(arm, analysis, config, pose) {
  if(!analysis.observable)return 0;
  const distance=norm(sub(pose.t,[0,0,.27]))+.62;
  const blurPx=(config.linearSpeedMps/distance+config.angularSpeedDegS*DEG*.1/distance)*(config.exposureMs/1000)*438;
  const anchorSupport=clamp(analysis.cornerRows/56,0,1);
  let miss=.0015+.006*Math.max(0,blurPx-.6)+.018*config.occlusion*config.occlusion;
  miss+=.003*(1-anchorSupport);
  if(analysis.condition>1e9)miss+=.02;
  return clamp(1-miss,.72,.9998);
}

export function computeArmMetrics(rows, analyses) {
  const accepted=rows.filter(r=>r.accepted),errors=accepted.map(r=>r.translationErrorMm),rot=accepted.map(r=>r.rotationErrorDeg),total=rows.length;
  return {
    total,accepted:accepted.length,p50:percentile(errors,.5),p95:percentile(errors,.95),p99:percentile(errors,.99),max:percentile(errors,1),
    rotationP95:percentile(rot,.95),coverage:accepted.length/total,geometricCoverage:analyses.filter(a=>a.observable).length/analyses.length,
    accurateYield:accepted.filter(r=>r.translationErrorMm<=3&&r.rotationErrorDeg<=.5).length/total,
    catastrophicRate:accepted.filter(r=>r.translationErrorMm>20||r.rotationErrorDeg>5).length/total,
    wrongAcceptRate:accepted.filter(r=>r.wrongAccept).length/total,
    failed:total-accepted.length,
    failureReasons:rows.filter(r=>!r.accepted).reduce((out,r)=>{const reason=r.failureReason||'no_pose';out[reason]=(out[reason]||0)+1;return out;},{}),
    l1SigmaP95:percentile(analyses.map(a=>a.sigmaTcpWorstMm),.95),conditionP95:percentile(analyses.map(a=>a.condition),.95),
    errors
  };
}

export function commonSuccessDelta(rowsByArm) {
  const paired=[];
  for(let i=0;i<rowsByArm.R0.length;i++)if(rowsByArm.R0[i].accepted&&rowsByArm.H1[i].accepted)paired.push({episode:rowsByArm.R0[i].episode,r0:rowsByArm.R0[i].translationErrorMm,h1:rowsByArm.H1[i].translationErrorMm});
  if(paired.length<10)return {count:paired.length,delta:null,r0P95:null,h1P95:null,paired};
  const r0=percentile(paired.map(x=>x.r0),.95),h1=percentile(paired.map(x=>x.h1),.95);
  return {count:paired.length,delta:r0-h1,r0P95:r0,h1P95:h1,paired};
}

export function bootstrapDelta(paired,episodes,seed,replicates) {
  if(paired.length<10)return [null,null];
  const blocks=Array.from({length:episodes},(_,episode)=>paired.filter(row=>row.episode===episode));
  if(blocks.filter(block=>block.length).length<2)return [null,null];
  const rng=new Rng(seed,'block-bootstrap'),values=[];
  for(let b=0;b<replicates;b++){
    const sample=[];for(let j=0;j<blocks.length;j++)sample.push(...blocks[Math.floor(rng.uniform()*blocks.length)]);
    const r0=percentile(sample.map(x=>x.r0),.95),h1=percentile(sample.map(x=>x.h1),.95);if(r0!==null&&h1!==null)values.push(r0-h1);
  }
  return [percentile(values,.025),percentile(values,.975)];
}

function finiteMatrix(matrix){return matrix&&matrix.every(row=>row.every(Number.isFinite));}

export function validateInputs(targets,config,analysesByArm) {
  const quadrants=targets.H1?.pasteQuadrants||{};
  const checks=[
    {id:'paste-quadrant',ok:quadrants[30]===3&&quadrants[31]===3&&quadrants[32]===2,label:'0907 paste quadrant 固定为 30→3 / 31→3 / 32→2'},
    {id:'descriptor-counts',ok:targets.H1?.anchors.length===3&&targets.H1?.edges.length===31,label:'0907 使用 3 anchors / 31 pose edges'},
    {id:'rig-ids',ok:(targets.R0?.anchors||[]).map(a=>a.id).sort((a,b)=>a-b).join(',')==='7,12,14,16,17',label:'R0 仅使用 7 / 12 / 14 / 16 / 17'},
    {id:'fisheye',ok:makeCameras(0).every(c=>c.D.length===4&&c.maxTheta>1.4),label:'七相机统一走声明式 fisheye 投影'},
    {id:'fim',ok:ARM_ORDER.every(arm=>(analysesByArm[arm]||[]).some(a=>a.observable&&finiteMatrix(a.covariance))),label:'三 arm 均产生 full-rank whitened FIM；秩在无量纲局部坐标检查'},
    {id:'same-tcp',ok:Number.isFinite(config.tcpOffsetMm),label:`三个 arm 使用同一 TCP 杠杆臂 ${config.tcpOffsetMm} mm`},
    {id:'measured-r0',ok:config.rigInputStatus==='measured',warning:true,label:'冻结 measured R0 layout（当前代理不可用于 G0 决策）'},
    {id:'production-cameras',ok:config.cameraInputStatus==='frozen-production',warning:true,label:'冻结生产 K/D/extrinsic 快照'},
    {id:'task-trajectory',ok:config.trajectoryInputStatus==='frozen-task',warning:true,label:'冻结真实任务分布轨迹'}
  ];
  return {checks,simulatorChecksPass:checks.filter(c=>!c.warning).every(c=>c.ok),g0Pass:checks.every(c=>c.ok)};
}

export function runSimulation(targets,inputConfig={},progress=null) {
  const config=configWithDefaults(inputConfig),cameras=makeCameras(config.cameraDropout),poses=generateTrajectory(config),analysesByArm={};
  for(const [armIndex,arm] of ARM_ORDER.entries()){
    analysesByArm[arm]=poses.map((pose,index)=>{
      if(progress&&index%24===0)progress({phase:'fim',arm,index,total:poses.length,ratio:(armIndex+index/poses.length)/ARM_ORDER.length});
      return analyzePose(targets[arm],pose,cameras,config,pose.phase);
    });
  }
  const commonRng=new Rng(config.seed,'common-frames'),detectionUniform=[],wrongUniform=[],measurementZ=[],timingZ=[];
  for(let i=0;i<poses.length;i++){detectionUniform.push(commonRng.uniform());wrongUniform.push(commonRng.uniform());measurementZ.push(commonRng.normals(6));timingZ.push(commonRng.normal());}
  const commonByEpisode=Array.from({length:config.episodes},(_,episode)=>{
    const rng=new Rng(config.seed,`common-episode-${episode}`);
    return {t:randomVector(rng,config.cameraTranslationSigmaMm*1e-3),r:randomVector(rng,config.cameraRotationSigmaDeg*DEG),sync:rng.normal()*config.syncSigmaMs*1e-3};
  });
  const installByArm={};
  for(const arm of ARM_ORDER){
    // Equal manufacturing assumptions and paired draws: no built-in advantage.
    installByArm[arm]=Array.from({length:config.episodes},(_,episode)=>{const rng=new Rng(config.seed,'install-'+episode);return {t:randomVector(rng,config.installTranslationSigmaMm*1e-3),r:randomVector(rng,config.installRotationSigmaDeg*DEG)};});
  }
  const rowsByArm={R0:[],H0:[],H1:[]};
  for(const arm of ARM_ORDER){
    for(let i=0;i<poses.length;i++){
      const pose=poses[i],analysis=analysesByArm[arm][i],common=commonByEpisode[pose.episode],install=installByArm[arm][pose.episode],acceptProb=detectorProbability(arm,analysis,config,pose),accepted=analysis.observable&&detectionUniform[i]<acceptProb;
      let translationErrorMm=null,rotationErrorDeg=null,wrongAccept=false;
      if(accepted){
        const L=cholesky(analysis.covariance),measurement=L?matVec(L,measurementZ[i]):Array(6).fill(0);
        const distance=norm(sub(pose.t,[0,0,.27]))+.62,blurPx=config.linearSpeedMps*(config.exposureMs/1000)*438/distance,blurScale=1+.12*Math.max(0,blurPx-.4);
        for(let k=0;k<6;k++)measurement[k]*=blurScale;
        const dt=common.sync+timingZ[i]*config.syncSigmaMs*.25e-3,timing=scale(pose.velocity,dt);
        let tErr=vecSum(measurement.slice(0,3),common.t,install.t,timing),rErr=vecSum(measurement.slice(3,6),common.r,install.r,scale(pose.angularVelocity,dt));
        const tcpWorld=rotateVector(pose,[config.tcpOffsetMm*1e-3,0,0]);tErr=add(tErr,cross(rErr,tcpWorld));
        const wrongProb=clamp((analysis.condition>1e10?.002:.00015)+.0002*config.occlusion,0,.01);
        wrongAccept=wrongUniform[i]<wrongProb;
        if(wrongAccept){const rng=new Rng(config.seed,`wrong-${arm}-${i}`);tErr=add(tErr,scale(normalize(rng.normals(3)),.022+.018*rng.uniform()));rErr=add(rErr,scale(normalize(rng.normals(3)),(5+4*rng.uniform())*DEG));}
        translationErrorMm=norm(tErr)*1000;rotationErrorDeg=norm(rErr)/DEG;
      }
      rowsByArm[arm].push({index:i,episode:pose.episode,accepted,failureReason:accepted?null:analysis.observable?'detector_proxy_miss':'unobservable',observable:analysis.observable,acceptProbability:acceptProb,translationErrorMm,rotationErrorDeg,wrongAccept,anchorCount:analysis.anchorCount,cameraCount:analysis.cameraCount,cornerRows:analysis.cornerRows,edgeRows:analysis.edgeRows,sigmaTcpWorstMm:analysis.sigmaTcpWorstMm});
    }
  }
  const metrics=Object.fromEntries(ARM_ORDER.map(arm=>[arm,computeArmMetrics(rowsByArm[arm],analysesByArm[arm])]));
  const paired=commonSuccessDelta(rowsByArm),ci=bootstrapDelta(paired.paired,config.episodes,config.seed,config.bootstrapReplicates),validation=validateInputs(targets,config,analysesByArm);
  const fimImprovement=metrics.R0.l1SigmaP95&&metrics.H1.l1SigmaP95?1-metrics.H1.l1SigmaP95/metrics.R0.l1SigmaP95:null;
  const strongG1=Number.isFinite(fimImprovement)&&(fimImprovement>=.30||metrics.H1.geometricCoverage>metrics.R0.geometricCoverage+.01);
  const limitedG1=Number.isFinite(fimImprovement)&&(fimImprovement>0||metrics.H1.geometricCoverage>metrics.H0.geometricCoverage+.03);
  const g1Outcome=strongG1?'CONTINUE':limitedG1?'LIMITED_L2':'STOP';
  const g1Continue=g1Outcome!=='STOP';
  const numericG2=paired.delta!==null&&ci[0]!==null&&ci[0]>=.8&&metrics.H1.p95<=3&&metrics.H1.rotationP95<=.5&&metrics.H1.coverage>.99&&metrics.H1.coverage>=metrics.R0.coverage&&metrics.H1.catastrophicRate<=metrics.R0.catastrophicRate;
  // Status strings cannot promote this proxy into an executed L2/L3 experiment.
  const evidenceReady=false;
  const decision=evidenceReady&&numericG2?'CONDITIONAL_GO':evidenceReady?'NO_GO':'INCONCLUSIVE';
  if(progress)progress({phase:'done',ratio:1});
  return {schema:'rig_target_ab_result/v1',generatedAt:new Date().toISOString(),config,cameras,poses,analysesByArm,rowsByArm,metrics,paired:{count:paired.count,deltaP95Mm:paired.delta,r0P95Mm:paired.r0P95,h1P95Mm:paired.h1P95,ci95Mm:ci},validation,gates:{g1Continue,g1Outcome,fimImprovement,numericG2,evidenceReady,decision}};
}

export function resultForExport(result) {
  return {
    schema:result.schema,generated_at:result.generatedAt,config:result.config,
    inputs:{rig:result.config.rigInputStatus,cameras:result.config.cameraInputStatus,trajectory:result.config.trajectoryInputStatus,l2:result.config.l2InputStatus},
    metrics:Object.fromEntries(ARM_ORDER.map(arm=>[arm,Object.fromEntries(Object.entries(result.metrics[arm]).filter(([key])=>key!=='errors'))])),
    paired:result.paired,validation:result.validation,gates:result.gates,
    claims_scope:'M0/M1/MC-lite sensitivity result; not production accuracy certification'
  };
}

export function rowsToCsv(result) {
  const header=['frame','episode','arm','observable','accepted','translation_error_mm','rotation_error_deg','wrong_accept','cameras','anchors','corner_rows','edge_rows','sigma_tcp_worst_mm','failure_reason'];
  const lines=[header.join(',')];
  for(const arm of ARM_ORDER)for(const row of result.rowsByArm[arm])lines.push([row.index,row.episode,arm,Number(row.observable),Number(row.accepted),row.translationErrorMm??'',row.rotationErrorDeg??'',Number(row.wrongAccept),row.cameraCount,row.anchorCount,row.cornerRows,row.edgeRows,row.sigmaTcpWorstMm??'',row.failureReason||''].join(','));
  return lines.join('\n');
}
