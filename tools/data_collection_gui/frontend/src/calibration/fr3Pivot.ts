// Presentation logic for the FR3 FK pivot and the ball check of the world ->
// fr3_base edge, kept out of the component so the wording can be pinned by a test.
//
// One ball fixed to the table, pivoted about twice. The FR3 in teaching mode
// (fr3_tcp_pivot.py) gives the socket centre in its EE frame -- which settles
// whether corenetic_socket_ee is the socket (until 10-14 it was EE + (0, 0, -10.3)
// mm, fitted end to end *through the live edge*, and had absorbed ~5 mm of the
// edge; since then it is the pivot's own value) -- and the ball in fr3_base. The handheld UMI pivoted about the same ball, recorded as an ordinary
// episode, gives the ball in the camera world. Each edge maps one onto the other;
// what is left over is that edge's error at the ball. Neither side uses a TCP
// definition, a station or the laser.
import type {
  BallCheckReport,
  BallEdge,
  Fr3PivotAnchor,
  Fr3PivotCommanded,
  Fr3PivotJob,
  Fr3PivotJointFit,
  Fr3PivotPlan,
  Fr3PivotResponse,
  PivotFitSummary,
} from "../types";
import { fmtNum } from "./fr3BaseEdge";

type Run = Fr3PivotResponse["runs"][number];
type Episode = Fr3PivotResponse["episodes"][number];

/** Below these the centre and the lever arm trade off (same gates as the tool's warning). */
export const MIN_SPAN_DEG = 20;
export const MIN_WEAK_SINGULAR = 0.1;
/** FK pivot residual above this: the insert was not seated the whole time. */
export const MAX_FR3_RESIDUAL_RMS_MM = 0.5;
/** |s| in corenetic_socket_ee within this: the socket frame is the socket. */
export const SOCKET_FRAME_TOLERANCE_MM = 1.0;

export function fmtVec(v: number[] | null | undefined, digits = 1): string {
  if (!v || v.length !== 3 || v.some((x) => x == null || !Number.isFinite(x))) return "—";
  return `(${v.map((x) => x.toFixed(digits)).join(", ")})`;
}

export function norm(v: number[] | null | undefined): number {
  return v ? Math.sqrt(v.reduce((acc, x) => acc + x * x, 0)) : Number.NaN;
}

export function isJobActive(job: Fr3PivotJob | null | undefined): boolean {
  return job?.state === "collecting" || job?.state === "fitting";
}

export function jobStatusText(job: Fr3PivotJob | null | undefined): string {
  switch (job?.state) {
    case "collecting":
      if (job.mode === "guided") {
        return job.finishRequested
          ? "引导 pivot：正在松开按压、收尾"
          : `引导 pivot 中 ${(job.elapsedS ?? 0).toFixed(0)} / ${(job.durationS ?? 0).toFixed(0)} s：扶着夹爪倾斜、旋转，机器人把球窝压在球上（偏置 ${fmtVec(job.biasMm, 0)} mm${isAnchored(job.anchorMm) ? `，弹簧挂在 ${fmtVec(job.anchorMm)} mm` : ""}）`;
      }
      if (job.mode === "auto") {
        return `自动 pivot 执行中 ${(job.elapsedS ?? 0).toFixed(0)} / ${(job.durationS ?? 0).toFixed(0)} s（柔顺阻抗，偏置 ${fmtVec(job.biasMm, 0)} mm）`;
      }
      return `采集中 ${(job.elapsedS ?? 0).toFixed(0)} / ${(job.durationS ?? 0).toFixed(0)} s（示教模式，机械臂是软的）`;
    case "fitting":
      return "采集完成，正在拟合";
    case "done":
      return `完成：${job.run ?? ""}`;
    case "stopped":
      return job.mode === "auto" || job.mode === "guided"
        ? "已停止：机械臂停在当前位置，这次不拟合"
        : "已停止，没有写入数据（机械臂已退出示教模式）";
    case "error":
      return `失败${job.returncode != null ? `（exit ${job.returncode}）` : ""}：${job.error || lastLine(job.log) || "见日志"}`;
    default:
      return "空闲";
  }
}

/** A run's own ball centre this far from the others': the ball moved (fr3_tcp_pivot.JOINT_CENTRE_MOVED_MM). */
export const JOINT_CENTRE_MOVED_MM = 1.0;

/** What a joint fit says, line by line, and whether it can be trusted. */
export function jointFitLines(joint: Fr3PivotJointFit): { ok: boolean; text: string }[] {
  const sock = joint.socketFrame;
  if (!sock) return [{ ok: false, text: "报告里没有 corenetic_socket_ee 的拟合" }];
  const out: { ok: boolean; text: string }[] = [];
  out.push({
    ok: true,
    text: `${joint.runs.length} 次采集，${joint.sharedCentre ? "共用一个球心" : "每次一个球心"}：残差 rms ${(sock.residual_rms_mm ?? Number.NaN).toFixed(2)} mm，总转角 ${(sock.span_deg ?? Number.NaN).toFixed(0)}°`,
  });
  const loo = sock.s_leave_one_run_out_mm ?? [];
  if (loo.length) {
    const spread = [0, 1, 2].map((i) => Math.max(...loo.map((v) => v[i])) - Math.min(...loo.map((v) => v[i])));
    out.push({
      ok: Math.max(...spread) <= 1,
      text: `去掉任意一次后 s 变化范围 ${fmtVec(spread, 2)} mm${Math.max(...spread) > 1 ? "：还没定住，多录大转角（工具轴方向靠绕竖直轴摆）" : ""}`,
    });
  }
  const moved = (sock.run_centre_offset_mm ?? [])
    .map((v, i) => ({ run: joint.runs[i], d: norm(v) }))
    .filter((x) => x.d > JOINT_CENTRE_MOVED_MM);
  if (moved.length) {
    out.push({
      ok: !joint.sharedCentre,
      text: `各次自己的球心差超过 ${JOINT_CENTRE_MOVED_MM} mm（${moved.map((x) => `${x.run} ${x.d.toFixed(1)} mm`).join("；")}）：球动过` +
        (joint.sharedCentre ? "，勾「每次一个球心」重拟合或分开拟合" : ""),
    });
  }
  if ((sock.span_deg ?? 0) < MIN_SPAN_DEG || (sock.weak_singular ?? 0) < MIN_WEAK_SINGULAR) {
    out.push({ ok: false, text: `转得不够（总转角 ${(sock.span_deg ?? 0).toFixed(0)}°、弱方向 ${(sock.weak_singular ?? 0).toFixed(2)}）` });
  }
  return out;
}

export function isAnchored(anchorMm: number[] | null | undefined): boolean {
  return !!anchorMm && anchorMm.some((x) => x !== 0);
}

/** The checkbox's lines: the offset it would hold and where it came from, or why it is off; and
 * the re-clamp warning, which does not stop it being used. */
export function anchorText(anchor: Fr3PivotAnchor | null | undefined): { ok: boolean; text: string; warning: string } {
  if (!anchor) return { ok: false, text: "网关没有给出拟合的球心", warning: "" };
  const warning = anchor.warning ?? "";
  if (!anchor.available) return { ok: false, text: anchor.reason, warning };
  const rms = anchor.residualRmsMm == null ? "" : `，残差 ${anchor.residualRmsMm.toFixed(2)} mm`;
  return { ok: true, text: `球心在 corenetic_socket_ee 里 ${fmtVec(anchor.sMm)} mm，来自 ${anchor.run}${rms}`, warning };
}

function lastLine(lines: string[] | undefined): string {
  return lines && lines.length ? lines[lines.length - 1] : "";
}

/** Why a pivot fit's centre may not be trusted. */
export function pivotFitWarnings(fit: PivotFitSummary | null | undefined): string[] {
  if (!fit) return ["还没拟合"];
  const out: string[] = [];
  if (fit.span_deg < MIN_SPAN_DEG || fit.weak_singular < MIN_WEAK_SINGULAR) {
    out.push(
      `转得不够（总转角 ${fit.span_deg.toFixed(0)}°、弱方向 ${fit.weak_singular.toFixed(2)}）：` +
        `yaw 和绕工具轴的 roll 都要转（各 ±${MIN_SPAN_DEG}° 以上），只绕一个轴转时球心和杠杆臂分不开`
    );
  }
  if (fit.residual_rms_mm > MAX_FR3_RESIDUAL_RMS_MM) {
    out.push(`残差 rms ${fit.residual_rms_mm.toFixed(2)} mm：插件中途离开过球，或球没固定牢`);
  }
  return out;
}

/** What the FK pivot says about corenetic_socket_ee, the frame the replay drives. */
export function socketFrameVerdict(run: Run): { text: string; ok: boolean | null } {
  if (!run.fk) return { text: "还没拟合", ok: null };
  if (!run.socketFrame) {
    return { text: `球窝中心在 corenetic_gripper_ee 系 ${fmtVec(run.fk.s_ee_mm)} mm（这份 URDF 没有 corenetic_socket_ee）`, ok: null };
  }
  const s = run.socketFrame.s_ee_mm;
  const off = norm(s);
  if (off <= SOCKET_FRAME_TOLERANCE_MM) {
    return {
      text: `corenetic_socket_ee 就是球窝中心（差 ${off.toFixed(2)} mm）：URDF 里的偏距是真实的 TCP 偏距`,
      ok: true,
    };
  }
  return {
    text:
      `球窝中心在 corenetic_socket_ee 系 ${fmtVec(s)} mm（差 ${off.toFixed(1)} mm）：` +
      `拟合时 URDF 里的偏距和实测不符（10-14 前的 −10.3 mm 混进了注册边的误差），` +
      `replay 目标应是实测值 ${fmtVec(run.fk.s_ee_mm)} mm（相对 corenetic_gripper_ee）`,
    ok: false,
  };
}

export function edgeDisplayName(name: string, fromBalls = false): string {
  if (name === "live") return "现役边";
  // scored on balls it was fitted to (or some of them): small by construction
  return name.replace(/\.json$/, "") + (fromBalls ? "（球位拟合，自评）" : "");
}

/** The handheld lever arm shared over a world's positions, next to how far the per-position ones wandered. */
export function handheldSharedSummary(report: BallCheckReport): string[] {
  return Object.entries(report.handheld_shared ?? {}).map(([world, sh]) => {
    if (sh.error || !sh.s_mm) return `${world}：${sh.error ?? "共用杠杆臂没有拟合"}`;
    const each = sh.s_per_position_mm ?? [];
    const spread = each.length
      ? Math.max(...[0, 1, 2].map((k) => Math.max(...each.map((v) => v[k])) - Math.min(...each.map((v) => v[k]))))
      : Number.NaN;
    return (
      `手持 ${sh.positions.length} 个位置共用一个杠杆臂 ${fmtVec(sh.s_mm)} mm（插件没换，就该是一个常量）` +
      (Number.isFinite(spread) ? `；各位置单独拟合时同一分量来回差 ${spread.toFixed(1)} mm，那是球心和杠杆臂互相耦合的拟合误差` : "")
    );
  });
}

/** One line on the edge the balls alone imply, and whether it may become a candidate. */
export function ballEdgeSummary(be: BallEdge): string {
  if (be.error) return be.error;
  const loo = (be.leave_one_out_mm ?? []).map((v) => (v == null ? "—" : v.toFixed(1))).join(" / ");
  const head =
    `${be.positions} 个位置（跨度 ${fmtVec(be.extent_mm, 0)} mm）刚体拟合残差 rms ${fmtNum(be.residual_rms_mm, 1, " mm")}` +
    (loo ? `；每个位置由其余预测：${loo} mm（rms ${fmtNum(be.leave_one_out_rms_mm, 1)}）` : "") +
    `；相机 world 尺度 ${fmtNum(be.scale_pct, 2, "%")}（刚体边不吸收）`;
  return be.verdict === "ok" ? `${head}。可以存为候选边` : `${head}。不能作候选：${(be.reasons ?? []).join("；")}`;
}

/** Why a check's ball edge cannot be saved as a candidate, or "" if it can. */
export function ballEdgeSaveBlocker(report: BallCheckReport, world: string): string {
  const be = report.ball_edges[world];
  if (!be || be.error) return be?.error ?? "没有球位边";
  if (be.verdict !== "ok") return "判定未通过";
  if (!report.world_reference_sha256) return "这次检验没有记下 world_reference.json（旧版检验），重新跑一次";
  if (report.reference_world && report.reference_world !== world) return `检验时的基准世界系是 ${report.reference_world}`;
  return "";
}

/** Edges that could be scored, best (smallest rms) first. */
export function rankedEdges(report: BallCheckReport): BallCheckReport["edges"] {
  return report.edges
    .filter((e) => !e.error && e.rms_mm != null && (e.positions?.length ?? 0) > 0)
    .sort((a, b) => (a.rms_mm ?? Infinity) - (b.rms_mm ?? Infinity));
}

/** One line on what the check can and cannot say, given how many positions it had. */
export function ballCheckVerdict(report: BallCheckReport): string {
  const usable = report.positions.filter((p) => p.usable).length;
  if (usable === 0) return "没有可用的球位置，见下表的原因";
  const ranked = rankedEdges(report);
  if (!ranked.length) return "没有能和这些录制比较的边（世界系不一致）";
  const best = ranked[0];
  const scope =
    usable === 1
      ? "只有 1 个球位置：只能看出该处的平移误差，看不出旋转（换 2–3 个相隔约 30 cm 的位置再测）"
      : `${usable} 个球位置`;
  const next = ranked[1];
  const margin = next
    ? `，比次好的 ${edgeDisplayName(next.name, next.from_balls)} 小 ${((next.rms_mm ?? 0) - (best.rms_mm ?? 0)).toFixed(1)} mm`
    : "";
  return `${scope}；最贴合的是 ${edgeDisplayName(best.name, best.from_balls)}（rms ${best.rms_mm?.toFixed(1)} mm${margin}）`;
}

export type PositionDraft = { pivot: string; episodeKey: string };

export function episodeKey(e: Pick<Episode, "dataset" | "episode">): string {
  return `${e.dataset}#${e.episode}`;
}

/** Problems with the chosen pairs that the check would not catch by itself. */
export function positionDraftWarnings(drafts: PositionDraft[], runs: Run[], episodes: Episode[]): string[] {
  const out: string[] = [];
  const complete = drafts.filter((d) => d.pivot && d.episodeKey);
  if (!complete.length) out.push("每个位置要选一次 FR3 pivot 和一个手持 episode");
  const pivots = complete.map((d) => d.pivot);
  if (new Set(pivots).size < pivots.length) out.push("同一次 FR3 pivot 用在了两个位置");
  const keys = complete.map((d) => d.episodeKey);
  if (new Set(keys).size < keys.length) out.push("同一个手持 episode 用在了两个位置");
  for (const d of complete) {
    const run = runs.find((r) => r.name === d.pivot);
    if (run && !run.fk) out.push(`${d.pivot} 还没拟合`);
    const ep = episodes.find((e) => episodeKey(e) === d.episodeKey);
    if (ep && !ep.world) out.push(`${ep.datasetName} ep${ep.episode} 没有 world 戳，检验会拒绝它`);
  }
  return out;
}

// --- planned pivot (fr3_pivot_auto.py) ---------------------------------------------------------

/** The +-3 mm pair is the proof the ball, not the command, decided the fit. */
export const BIAS_PRESETS: { label: string; mm: number[] }[] = [
  { label: "无偏置", mm: [0, 0, 0] },
  { label: "偏置 +3 mm（x）", mm: [3, 0, 0] },
  { label: "偏置 −3 mm（x）", mm: [-3, 0, 0] },
];
/** |Δ fitted centre| / |Δ commanded bias| below this: the socket stayed on the ball. */
export const SEATED_MAX_RATIO = 0.15;
export const FOLLOWED_MIN_RATIO = 0.5;
export const MIN_CONTACT_FRACTION = 0.95;

const minOf = (v: number[] | null | undefined) => (v && v.length ? Math.min(...v) : Number.NaN);
const maxOf = (v: number[] | null | undefined) => (v && v.length ? Math.max(...v) : Number.NaN);

export function planSummaryText(plan: Fr3PivotPlan): string {
  if (!plan.passed) return `没通过：${plan.problems.join("；")}`;
  const clearance =
    plan.tableClearanceMm == null
      ? "没做 MuJoCo 检查"
      : `离桌面${plan.tableZMm == null ? "" : `（z=${plan.tableZMm.toFixed(0)} mm）`}最近 ${plan.tableClearanceMm.toFixed(0)} mm（${plan.tableClearanceBody}）`;
  return (
    `通过：${(plan.durationS ?? 0).toFixed(0)} s，最快关节 ${maxOf(plan.maxJointSpeedDegS).toFixed(0)}°/s，` +
    `离 panda_py 的墙最近 ${minOf(plan.minMarginWallDeg).toFixed(0)}°，j6 转 ${(plan.jointTravelDeg?.[5] ?? 0).toFixed(0)}°、` +
    `j7 转 ${(plan.jointTravelDeg?.[6] ?? 0).toFixed(0)}°，${clearance}`
  );
}

export function biasLabel(mm: number[] | null | undefined): string {
  const preset = BIAS_PRESETS.find((b) => mm && b.mm.every((x, i) => Math.abs(x - mm[i]) < 1e-6));
  return preset ? preset.label : `偏置 ${fmtVec(mm, 1)} mm`;
}

export function contactText(c: Fr3PivotCommanded | null | undefined): { text: string; ok: boolean } | null {
  if (!c) return null;
  if (!c.contact) return { text: "没有接触力记录", ok: false };
  const k = c.contact;
  const ok = k.in_contact_fraction >= MIN_CONTACT_FRACTION;
  return {
    text:
      `按压 ${k.press_median_n.toFixed(1)} N（5% 分位 ${k.press_p05_n.toFixed(1)} N），接触 ${(k.in_contact_fraction * 100).toFixed(0)}%` +
      (ok ? "" : "：有一段没压在球上，这次可能跟着指令走了"),
    ok,
  };
}

export type BiasPairVerdict = {
  centre: string;
  runs: [string, string];
  dBiasMm: number;
  dCentreMm: number;
  dSocketMm: number;
  seated: boolean | null;
  text: string;
};

/** Per captured centre, the two completed runs with the most different bias: did the fit follow the command? */
export function biasConsistency(runs: Run[]): BiasPairVerdict[] {
  const byCentre = new Map<string, Run[]>();
  for (const r of runs) {
    const c = r.commanded;
    if (!c?.completed || !r.fk || !c.bias_mm) continue;
    byCentre.set(c.centre, [...(byCentre.get(c.centre) ?? []), r]);
  }
  const out: BiasPairVerdict[] = [];
  for (const [centre, group] of byCentre) {
    let best: [Run, Run, number] | null = null;
    for (let i = 0; i < group.length; i++) {
      for (let j = i + 1; j < group.length; j++) {
        const d = norm(group[i].commanded!.bias_mm!.map((x, k) => x - group[j].commanded!.bias_mm![k]));
        if (d > 1 && (!best || d > best[2])) best = [group[i], group[j], d];
      }
    }
    if (!best) continue;
    const [a, b, dBias] = best;
    const dCentre = norm(a.fk!.c_base_mm.map((x, k) => x - b.fk!.c_base_mm[k]));
    const sa = a.socketFrame?.s_ee_mm ?? a.fk!.s_ee_mm;
    const sb = b.socketFrame?.s_ee_mm ?? b.fk!.s_ee_mm;
    const dSocket = norm(sa.map((x, k) => x - sb[k]));
    const ratio = dCentre / dBias;
    const head = `球心差 ${dCentre.toFixed(2)} mm、球窝差 ${dSocket.toFixed(2)} mm（指令差 ${dBias.toFixed(1)} mm）`;
    let seated: boolean | null;
    let text: string;
    if (ratio < SEATED_MAX_RATIO) {
      seated = true;
      text = `${head}：结果由球决定，测量成立`;
    } else if (ratio > FOLLOWED_MIN_RATIO) {
      seated = false;
      text = `${head}：结果跟着指令走，插件没坐住，这组不能用`;
    } else {
      seated = null;
      text = `${head}：部分跟着指令走（插件被拖动或摩擦大），结果要打折扣`;
    }
    out.push({ centre, runs: [a.name, b.name], dBiasMm: dBias, dCentreMm: dCentre, dSocketMm: dSocket, seated, text });
  }
  return out;
}
