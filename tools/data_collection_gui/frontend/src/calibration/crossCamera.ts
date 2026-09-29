// Presentation logic for the cross-camera consistency check.
//
// Two things the wording must not do. It must not present a split of the
// disagreement as the only one: offsets are defined up to a shift every camera
// shares, so the least-squares split is shown beside the robust one. And it
// must not let a report about replaced extrinsics read as a verdict on the
// current ones.
import type {
  CrossCameraCamera,
  CrossCameraOverall,
  CrossCameraReport,
  CrossCameraSetChanges,
  CrossCameraVerdict,
} from "../types";

export const verdictLabel: Record<CrossCameraVerdict, string> = {
  ok: "一致",
  warn: "接近门槛",
  fail: "对不上",
  unknown: "未判定",
};

export const verdictDot: Record<CrossCameraVerdict, string> = {
  ok: "running",
  warn: "warning",
  fail: "error",
  unknown: "idle",
};

export const overallLabel: Record<CrossCameraOverall, string> = {
  ok: "各相机一致",
  partial: "部分未判定",
  warn: "有相机接近门槛",
  fail: "相机之间对不上",
  unknown: "无法判定",
};

export const overallDot: Record<CrossCameraOverall, string> = {
  ok: "running",
  partial: "warning",
  warn: "warning",
  fail: "error",
  unknown: "idle",
};

export type CrossCameraRow = {
  key: string;
  camera: string;
  serial: string;
  verdict: CrossCameraVerdict;
  offset: string;
  detail: string;
};

const mm = (value: number | null | undefined) => (value == null ? "—" : `${value.toFixed(1)} mm`);

export function formatOffset(camera: CrossCameraCamera): string {
  if (camera.lateral_offset_mm == null) return "—";
  const robust = mm(camera.lateral_offset_mm);
  const ls = camera.lateral_offset_least_squares_mm;
  // Only worth a second number when the two splits actually differ.
  return ls != null && Math.abs(ls - camera.lateral_offset_mm) >= 0.5 ? `${robust}（最小二乘 ${mm(ls)}）` : robust;
}

export function formatDetail(camera: CrossCameraCamera): string {
  if (camera.verdict === "unknown") return camera.reason ?? `${camera.frames} 帧`;
  const parts = [`${camera.frames} 帧`];
  if (camera.scatter_p95_mm != null) parts.push(`逐帧散布 p95 ${mm(camera.scatter_p95_mm)}`);
  // Depth is where a marker size error lives; shown so it is not mistaken for
  // something the verdict missed.
  if (camera.along_p50_mm != null) parts.push(`深度 ${camera.along_p50_mm >= 0 ? "+" : ""}${mm(camera.along_p50_mm)}（不参与判定）`);
  if (camera.rotation_p50_deg != null) parts.push(`姿态分歧 ${camera.rotation_p50_deg.toFixed(2)}°`);
  return parts.join(" · ");
}

export function crossCameraRows(report: CrossCameraReport | null): CrossCameraRow[] {
  if (!report) return [];
  const cubes = Object.keys(report.cubes).sort();
  const serials = report.camera_serials ?? {};
  return cubes.flatMap((cube) =>
    Object.keys(report.cubes[cube].cameras)
      .sort()
      .map((camera) => {
        const entry = report.cubes[cube].cameras[camera];
        return {
          key: `${cube}/${camera}`,
          camera: cubes.length > 1 ? `${camera}（${cube}）` : camera,
          serial: serials[camera] ?? "",
          verdict: entry.verdict,
          offset: formatOffset(entry),
          detail: formatDetail(entry),
        };
      }),
  );
}

export function setChangeSummary(steps: CrossCameraSetChanges): string {
  if (steps.verdict === "unknown" && steps.reason) return steps.reason;
  const flicker = steps.flicker_changes ? `，另有 ${steps.flicker_changes} 次逐帧闪烁未计入` : "";
  if (!steps.changes) return `录制期间相机集没有干净的切换${flicker}`;
  return (
    `${steps.changes} 次切换，跳变 p50 ${mm(steps.step_p50_mm)} / p95 ${mm(steps.step_p95_mm)}` +
    `（不切换时 ${mm(steps.steady_p50_mm)} / ${mm(steps.steady_p95_mm)}，预算 ${mm(steps.budget_mm)}）${flicker}`
  );
}

/** Why this report may not describe the calibration in production, or "". */
export function staleReason(report: CrossCameraReport | null, currentExtrinsics: string | undefined): string {
  if (!report) return "";
  if (report.extrinsics_run && currentExtrinsics && report.extrinsics_run !== currentExtrinsics) {
    return `这份结果针对的是 ${report.extrinsics_run}，当前外参是 ${currentExtrinsics}——换外参后重新生成轨迹再跑一次。`;
  }
  return "";
}
