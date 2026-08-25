// Presentation logic for the camera self-check, kept out of the component so
// the wording of a verdict can be pinned by a test.
//
// The one rule worth stating: a verdict must never sound more certain than the
// measurement. "unknown" is not "ok", and "every camera I could check is fine"
// is not "every camera is fine" -- sending someone to redo a calibration is
// expensive, and telling them the rig is fine when it was never checked is
// worse.
import type { RigCheckCamera, RigCheckOverall, RigCheckReport, RigCheckVerdict } from "../types";

// A camera with no baseline was never in the comparison, which is a third
// thing: not a verdict, not a failed measurement. It gets its own row label so
// the table cannot imply it was checked.
export type RigCheckRowVerdict = RigCheckVerdict | "no_baseline";

export const verdictLabel: Record<RigCheckRowVerdict, string> = {
  ok: "未移动",
  suspect: "接近阈值",
  moved: "疑似移动",
  unknown: "无法判定",
  no_baseline: "无基线",
};

export const verdictDot: Record<RigCheckRowVerdict, string> = {
  ok: "running",
  suspect: "warning",
  moved: "error",
  unknown: "idle",
  no_baseline: "idle",
};

export const overallLabel: Record<RigCheckOverall, string> = {
  ok: "全部未移动",
  partial: "部分未检出",
  suspect: "有相机接近阈值",
  moved: "检测到相机移动",
  inconclusive: "无法判定（场景变化过大）",
  unknown: "无法判定",
};

export const overallDot: Record<RigCheckOverall, string> = {
  ok: "running",
  partial: "warning",
  suspect: "warning",
  moved: "error",
  inconclusive: "warning",
  unknown: "idle",
};

/** Only a confirmed move should push someone into a recalibration. */
export function shouldOfferRecalibration(report: RigCheckReport | null): boolean {
  return report?.overall === "moved";
}

/** Re-baselining after a real move would erase the very evidence of it. */
export function canUpdateBaseline(report: RigCheckReport | null): boolean {
  return report == null || report.overall !== "moved";
}

export function formatShift(camera: RigCheckCamera): string {
  if (camera.status !== "measured" || camera.shift_px_median == null) {
    return "—";
  }
  const px = `${camera.shift_px_median.toFixed(2)} px`;
  if (camera.equivalent_rotation_deg == null) {
    return px;
  }
  // The angle is what makes the number actionable; pixels alone mean nothing
  // without knowing the focal length.
  return `${px} · ≈${camera.equivalent_rotation_deg.toFixed(3)}°`;
}

export function formatImpact(camera: RigCheckCamera): string {
  const mm = camera.equivalent_error_mm_at_working_distance;
  if (camera.status !== "measured" || mm == null) {
    return "";
  }
  return `1 m 处约 ${mm.toFixed(1)} mm`;
}

export type RigCheckRow = {
  camera: string;
  verdict: RigCheckRowVerdict;
  shift: string;
  impact: string;
  note: string;
};

export function rigCheckRows(report: RigCheckReport | null): RigCheckRow[] {
  if (!report) return [];
  // Why a frame is missing is the gateway's knowledge, not the analysis's:
  // "no current frame" alone leaves an operator with nothing to act on.
  const failed = new Map((report.failed_captures ?? []).map((entry) => [entry.camera, entry.reason]));
  const joined = (...parts: (string | undefined)[]) => parts.filter(Boolean).join(" · ");

  const compared: RigCheckRow[] = Object.keys(report.cameras)
    .sort()
    .map((camera) => {
      const entry = report.cameras[camera];
      return {
        camera,
        verdict: entry.verdict,
        shift: formatShift(entry),
        impact: formatImpact(entry),
        note: joined(entry.reason, failed.get(camera)),
      };
    });
  const unbaselined: RigCheckRow[] = [...(report.cameras_without_baseline ?? [])]
    .sort()
    .map((camera) => ({
      camera,
      verdict: "no_baseline" as const,
      shift: "—",
      impact: "",
      note: joined("基线之后才出现的相机，没有比较对象", failed.get(camera)),
    }));
  // A camera that has no baseline *and* produced no frame is in neither list;
  // dropping it would hide exactly the camera that is misbehaving most.
  const seen = new Set([...compared, ...unbaselined].map((row) => row.camera));
  const missing: RigCheckRow[] = [...failed.keys()]
    .filter((camera) => !seen.has(camera))
    .sort()
    .map((camera) => ({
      camera,
      verdict: "unknown" as const,
      shift: "—",
      impact: "",
      note: failed.get(camera) ?? "未取到画面",
    }));
  return [...compared, ...unbaselined, ...missing];
}

/** How stale the baseline is, in words, or why there isn't one. */
export function baselineSummary(report: RigCheckReport | null, fallback?: RigCheckReport["baseline"]): string {
  const baseline = report?.baseline ?? fallback;
  if (!baseline?.exists) {
    return "尚未采集基线 — 标定完成后先采集一次，自检才有比较对象";
  }
  const runs = [baseline.intrinsics_run, baseline.extrinsics_run].filter(Boolean).join(" / ");
  const when = baseline.captured_at ?? "未知时间";
  return runs ? `${when} · 对应标定 ${runs}` : when;
}
