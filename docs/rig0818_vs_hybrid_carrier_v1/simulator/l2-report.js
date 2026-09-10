import {ARM_ORDER, percentile, commonSuccessDelta, bootstrapDelta} from './simulator-core.js';

export function readL2Report(doc) {
  if(doc?.schema!=='rig_target_ab_l2/v1')throw new Error('Expected rig_target_ab_l2/v1 report');
  const total=doc.rowsByArm?.R0?.length;
  if(!Number.isInteger(total)||total<1||total>1200)throw new Error('L2 report must contain 1–1200 requested frames');
  for(const arm of ARM_ORDER){
    const rows=doc.rowsByArm?.[arm];
    if(!Array.isArray(rows)||rows.length!==total)throw new Error('All L2 arms must retain the same requested frames');
    rows.forEach((r,i)=>{
      if(r.index!==i||!Number.isInteger(r.episode)||r.episode<0||r.episode>=total||r.episode!==doc.rowsByArm.R0[i].episode||typeof r.accepted!=='boolean')throw new Error('Invalid L2 frame pairing');
      if(r.accepted&&![r.translationErrorMm,r.rotationErrorDeg].every(v=>Number.isFinite(v)&&v>=0))throw new Error('Accepted L2 frames need finite nonnegative errors');
      if(!r.accepted&&(r.translationErrorMm!==null||r.rotationErrorDeg!==null))throw new Error('Failed L2 frames must not carry finite penalty errors');
    });
  }
  const metrics=Object.fromEntries(ARM_ORDER.map(arm=>{
    const rows=doc.rowsByArm[arm],ok=rows.filter(r=>r.accepted),errors=ok.map(r=>r.translationErrorMm);
    return [arm,{total,accepted:ok.length,failed:total-ok.length,errors,p50:percentile(errors,.5),p95:percentile(errors,.95),p99:percentile(errors,.99),
      rotationP95:percentile(ok.map(r=>r.rotationErrorDeg),.95),coverage:ok.length/total,
      accurateYield:ok.filter(r=>r.translationErrorMm<=3&&r.rotationErrorDeg<=.5).length/total,
      catastrophicRate:ok.filter(r=>r.translationErrorMm>20||r.rotationErrorDeg>5).length/total}];
  }));
  const p=commonSuccessDelta(doc.rowsByArm),episodes=Math.max(...doc.rowsByArm.R0.map(r=>r.episode))+1;
  return {...doc,metrics,paired:{count:p.count,deltaP95Mm:p.delta,ci95Mm:bootstrapDelta(p.paired,episodes,doc.config?.seed||0,320)},decision:'INCONCLUSIVE'};
}

export function allFrameCdf(metric) {
  return metric.errors.slice().sort((a,b)=>a-b).map((error,i)=>({error,fraction:(i+1)/metric.total}));
}
