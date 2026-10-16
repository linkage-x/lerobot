// Presentation logic for the world -> fr3_base registration edge, kept out of
// the component so the wording can be pinned by a test.
//
// The rule this file exists to enforce: a candidate's own verdict is not enough
// to swap the edge. Each solve checks itself out of sample against the same
// run's captures, so two runs can both pass and still disagree -- on 2026-10-13
// the two 10-10 runs (held-out p95 5.2 / 5.6 mm) each moved the live edge by
// 2.6 mm where they were captured, about 4 mm at the 10-13 replay trajectory, in
// different directions. How far a candidate moves the live edge, and how far
// the candidates are from each other, are shown next to the verdict.
//
// Two methods feed the same table. The laser edge is a tracker station
// (world <- tracker) times the FR3's FK fitted to the laser on replays
// (tracker <- fr3_base); it is one edge only if the tracker did not move in
// between, which nothing in either half can show except two replays on either
// side of the station agreeing -- so the selection is judged on that.
//
// The ball edge is the rigid fit of the ball check's positions (a ball both the
// FR3 and the handheld pivoted about, at 4+ places): judged by predicting each
// position from the others. Rigid on purpose -- the camera world's scale is
// shown, not absorbed; it is fixed at its source and the edge refitted.
import type { Fr3BaseEdgeMove, Fr3BaseEdgeQuality, Fr3BaseEdgeResponse } from "../types";

type Current = NonNullable<Fr3BaseEdgeResponse["current"]>;
type Candidate = Fr3BaseEdgeResponse["candidates"][number];
type Station = Fr3BaseEdgeResponse["stations"][number];
type LaserReplay = Fr3BaseEdgeResponse["laserReplays"][number];

export function fmtNum(value: number | null | undefined, digits = 2, unit = ""): string {
  return value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(digits)}${unit}`;
}

export function runLabel(path: string): string {
  return path.split("/").filter(Boolean).pop() ?? path;
}

export function methodLabel(q: Fr3BaseEdgeQuality): string {
  return q.method === "laser" ? "激光" : q.method === "ball" ? "同球检验" : "单 tag";
}

/** The numbers that judge an edge, by how it was measured. */
export function qualitySummary(q: Fr3BaseEdgeQuality): string {
  if (q.method === "ball") {
    const scale = q.scalePct == null ? "" : ` · 相机 world 尺度 ${q.scalePct >= 0 ? "+" : ""}${fmtNum(q.scalePct, 2, "%")}（未吸收）`;
    const arm = q.handheldS === "per-position" ? " · 手持各自杠杆臂" : "";
    return (
      `${q.positions ?? "—"} 个球位 · 留一预测 rms ${fmtNum(q.leaveOneOutRmsMm, 1, " mm")} / 最差 ${fmtNum(q.leaveOneOutMaxMm, 1, " mm")} · ` +
      `拟合残差 rms ${fmtNum(q.residualRmsMm, 1, " mm")}${scale}${arm}`
    );
  }
  if (q.method === "laser") {
    const spread = q.replaySpreadMm == null ? "" : ` · 回放之间相差 ${fmtNum(q.replaySpreadMm, 2, " mm")}`;
    return (
      `站位 ${q.stationPoses ?? "—"} 点 rms ${fmtNum(q.stationRmsMm, 2, " mm")}${q.stationCertifies ? "" : "（未认证）"} · ` +
      `${q.replays ?? 0} 次回放 激光-FK p95 ${fmtNum(q.laserVsFkP95Mm, 2, " mm")} · 姿态跨度 ${fmtNum(q.orientationSpanDeg, 1, "°")}` +
      spread
    );
  }
  return (
    `${q.captures ?? "—"} 次捕获 · 样本外 tag 中心 p95 ${fmtNum(q.outOfSampleP95Mm, 2, " mm")} · ` +
    `tag 尺度 ${fmtNum(q.tagScale, 4)} · RMSE ${fmtNum(q.rmsePx, 2, " px")}`
  );
}

export function liveEdgeSummary(current: Current | null): string {
  if (!current) return "没有注册边";
  return (
    `${current.world} → fr3_base · ${methodLabel(current.quality)} · 来源 ${current.label || runLabel(current.p0Run) || "—"} · ` +
    qualitySummary(current.quality)
  );
}

/** What is wrong with the live edge itself, worded as what it does to the FR3. */
export function liveEdgeWarnings(current: Current | null, referenceWorld: string): string[] {
  if (!current) {
    return ["world_graph.json 里没有到 fr3_base 的边：MuJoCo 预览、IK QC、真机 replay 和导出到 fr3_base 都会被拒绝。"];
  }
  const warnings: string[] = [];
  if (!current.fromReferenceWorld) {
    warnings.push(
      `现役边是从 ${current.world} 出发的，当前基准世界系是 ${referenceWorld || "—"}：新录的数据走不到 fr3_base，需要在当前基准下重新解算一条边。`
    );
  } else if (!current.referenceUnchanged) {
    warnings.push("world_reference.json 在解出现役边之后改过，这条边对应的相机位姿可能已经不是现在的。");
  }
  if (current.edgeCount > 1) {
    warnings.push(`有 ${current.edgeCount} 条边接到 fr3_base，replay 只会走其中一条。`);
  }
  return warnings;
}

export function moveSummary(move: Fr3BaseEdgeMove | null): string {
  if (!move) return "—";
  const [x, y, z] = move.meanMm.map((v) => v.toFixed(1));
  return `平均 ${move.meanNormMm.toFixed(1)} mm (${x}, ${y}, ${z})，最大 ${move.maxMm.toFixed(1)} mm，转 ${move.rotationDeg.toFixed(2)}°`;
}

/**
 * The largest disagreement between two passing candidates, in the workspace.
 *
 * Each candidate's move is measured at its own run's tool positions, so this is
 * the difference of two averages over slightly different places -- an estimate
 * of how repeatable the method is, not an exact edge-to-edge distance.
 */
export function candidateSpread(candidates: Candidate[]): { mm: number; a: string; b: string } | null {
  const usable = candidates.filter((c) => c.verdict === "ok" && c.moveVsLive);
  // the live edge counts too: it sits at zero move
  const points = usable.map((c) => ({ name: c.label, mean: c.moveVsLive!.meanMm }));
  if (!usable.some((c) => c.sameAsLive)) points.push({ name: "现役边", mean: [0, 0, 0] });
  let best: { mm: number; a: string; b: string } | null = null;
  for (let i = 0; i < points.length; i += 1) {
    for (let j = i + 1; j < points.length; j += 1) {
      if (points[i].name === points[j].name) continue;
      const d = Math.hypot(...points[i].mean.map((v, k) => v - points[j].mean[k]));
      if (!best || d > best.mm) best = { mm: d, a: points[i].name, b: points[j].name };
    }
  }
  return best && points.length >= 2 ? best : null;
}

export function verdictLabel(candidate: Candidate): string {
  if (candidate.sameAsLive) return "= 现役边";
  return candidate.verdict === "ok" ? "通过" : "未通过";
}

export function applyConfirmation(candidate: Candidate, current: Current | null): string {
  const move = candidate.moveVsLive;
  return (
    `把 world → fr3_base 换成 ${candidate.label}（${methodLabel(candidate.quality)}）解出的边` +
    (move && current ? `：在这一轮的工作区里，它把 fr3_base 下的每个位姿平均挪 ${move.meanNormMm.toFixed(1)} mm（最大 ${move.maxMm.toFixed(1)} mm）。` : "。") +
    "之后的 MuJoCo 预览、IK QC、真机 replay 和导出到 fr3_base 都改走新边；已经导出的数据集和做过的 replay 分析仍是旧边的结果。" +
    "旧边保存在新边的 supersedes 里。改动只发生在这台机器的 world_graph.json：请拷回仓库并提交。"
  );
}

function hoursBetween(a: string, b: string): number | null {
  const ta = Date.parse(a);
  const tb = Date.parse(b);
  return Number.isFinite(ta) && Number.isFinite(tb) ? (tb - ta) / 3.6e6 : null;
}

/** Where a replay sits relative to the station, which is what decides whether they share a tracker placement. */
export function replayRelation(replay: LaserReplay, station: Station | null): string {
  if (!station?.sessionUtc || !replay.sessionUtc) return "—";
  const h = hoursBetween(station.sessionUtc, replay.sessionUtc);
  if (h == null) return "—";
  if (replay.sessionUtc.slice(0, 10) !== station.sessionUtc.slice(0, 10)) return "另一天";
  const span = Math.abs(h) < 1 ? `${Math.round(Math.abs(h) * 60)} 分钟` : `${Math.abs(h).toFixed(1)} 小时`;
  return h < 0 ? `驻点前 ${span}` : `驻点后 ${span}`;
}

/** What is wrong with a station + replays selection before it is solved; the solve gates the rest. */
export function laserSelectionWarnings(station: Station | null, replays: LaserReplay[]): string[] {
  if (!station) return ["先选一个站位拟合。"];
  const warnings: string[] = [];
  if (!station.matchesReference) warnings.push(`站位拟合在 ${station.world || "—"} 下，不是当前基准世界系。`);
  if (!station.certifies) warnings.push("站位拟合没有通过认证，解出来也会被拒。");
  // register_fr3_base GATE_STATION_RMS_MM: certification judges the geometry only
  if (station.rmsMm != null && station.rmsMm > 1.5) {
    warnings.push(`站位残差 ${station.rmsMm.toFixed(2)} mm > 1.5 mm（相机和激光不自洽，常见是尺度差），解出来会被拒。`);
  }
  if (!replays.length) return [...warnings, "至少选一次带激光的 FR3 回放。"];
  const relations = replays.map((r) => replayRelation(r, station));
  if (relations.includes("另一天")) {
    warnings.push("有回放和驻点不是同一天：跟踪仪很可能挪过，两段接不成一条边。");
  }
  const before = relations.some((r) => r.startsWith("驻点前"));
  const after = relations.some((r) => r.startsWith("驻点后"));
  if (replays.length === 1) {
    warnings.push("只有一次回放：证明不了跟踪仪在驻点和回放之间没动。最好驻点前后各回放一次。");
  } else if (!(before && after)) {
    warnings.push("回放没有分在驻点两侧：它们彼此一致，也说明不了驻点期间跟踪仪没动。");
  }
  return warnings;
}
