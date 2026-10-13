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
import type { Fr3BaseEdgeMove, Fr3BaseEdgeResponse } from "../types";

type Current = NonNullable<Fr3BaseEdgeResponse["current"]>;
type Candidate = Fr3BaseEdgeResponse["candidates"][number];

export function fmtNum(value: number | null | undefined, digits = 2, unit = ""): string {
  return value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(digits)}${unit}`;
}

export function runLabel(path: string): string {
  return path.split("/").filter(Boolean).pop() ?? path;
}

export function liveEdgeSummary(current: Current | null): string {
  if (!current) return "没有注册边";
  const q = current.quality;
  return (
    `${current.world} → fr3_base · 来源 ${runLabel(current.p0Run) || "—"} · ${q.captures ?? "—"} 次捕获 · ` +
    `样本外 tag 中心 p95 ${fmtNum(q.outOfSampleP95Mm, 2, " mm")} · tag 尺度 ${fmtNum(q.tagScale, 4)}`
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
  const points = usable.map((c) => ({ name: runLabel(c.p0Run), mean: c.moveVsLive!.meanMm }));
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
    `把 world → fr3_base 换成 ${runLabel(candidate.p0Run)} 解出的边` +
    (move && current ? `：在这一轮的工作区里，它把 fr3_base 下的每个位姿平均挪 ${move.meanNormMm.toFixed(1)} mm（最大 ${move.maxMm.toFixed(1)} mm）。` : "。") +
    "之后的 MuJoCo 预览、IK QC、真机 replay 和导出到 fr3_base 都改走新边；已经导出的数据集和做过的 replay 分析仍是旧边的结果。" +
    "旧边保存在新边的 supersedes 里。改动只发生在这台机器的 world_graph.json：请拷回仓库并提交。"
  );
}
