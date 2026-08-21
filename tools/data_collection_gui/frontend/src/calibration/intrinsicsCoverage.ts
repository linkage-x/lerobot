// How much of each camera's frame the intrinsics production uses was actually
// measured on, and how much of it the distortion model is extrapolating over.
//
// This reports a property of the calibration; it deliberately stops short of
// telling anyone to re-shoot. Coverage is worth reporting because no
// reprojection number contains it: a held-out RMSE is computed on the frames
// the board appeared in, so a camera whose outer ring was never swept scores
// just as well as one covered edge to edge. Two independent fits of this rig,
// indistinguishable at 0.01 px held-out, still sent the same pixel down rays
// differing by 1-2 mm at working distance, and the camera they disagreed about
// most (cam_06, 1.68 mm) was the one with the least edge coverage.
//
// But low coverage only costs anything where the camera actually works, and on
// this rig it mostly does not. Measured on 2026-08-21 against the cube in
// thor_gmsl2_9ch_v1_20260817_162847: cam_06 covers 79% of its frame radius and
// the cube never passes 52% of it, so its unmeasured band is never entered.
// The one camera that does work past its covered band, cam_09, turned out to
// carry a ~27 mm error that is flat in image radius (27.3 mm at 41 deg vs
// 25.6 mm at 67 deg) and flat in visible marker count -- neither of which an
// extrapolating distortion model produces. That error is extrinsic. So the
// verdict here names what was measured and leaves the re-shoot decision to
// whoever knows where the camera has to work.
//
// The one exception is a model that folds back inside its own frame: those
// pixels have no unique ray at any working distance, which is a defect in the
// fit itself rather than a question about where it gets used.
//
// As in the rig self-check, a verdict must never sound more certain than the
// measurement: intrinsics with no recorded coverage are "无法判定", never "ok".
import type { IntrinsicsCoverageCamera, IntrinsicsCoverageResponse } from "../types";

export type CoverageVerdict = "covered" | "extrapolated" | "folded" | "unknown";

export const coverageVerdictLabel: Record<CoverageVerdict, string> = {
  covered: "覆盖充分",
  extrapolated: "外侧外推",
  folded: "画幅内折返",
  unknown: "无法判定",
};

export const coverageVerdictDot: Record<CoverageVerdict, string> = {
  covered: "running",
  extrapolated: "warning",
  folded: "error",
  unknown: "idle",
};

export type CoverageRow = {
  camera: string;
  verdict: CoverageVerdict;
  coverage: string;
  fold: string;
  note: string;
};

type Thresholds = Pick<IntrinsicsCoverageResponse, "coverageTarget" | "foldMarginWarnDeg">;

export function coverageVerdict(camera: IntrinsicsCoverageCamera, response: Thresholds): CoverageVerdict {
  const coverage = camera.coverage;
  if (coverage == null) {
    return "unknown";
  }
  if (camera.foldsInsideFrame) {
    return "folded";
  }
  const margin = camera.foldMarginDeg;
  if (coverage < response.coverageTarget || (margin != null && margin < response.foldMarginWarnDeg)) {
    return "extrapolated";
  }
  return "covered";
}

export function formatCoverage(camera: IntrinsicsCoverageCamera): string {
  return camera.coverage == null ? "—" : `${(camera.coverage * 100).toFixed(0)}%`;
}

export function formatFold(camera: IntrinsicsCoverageCamera): string {
  if (camera.coverage == null) {
    return "—";
  }
  // null margin with a known coverage means the fit never folds anywhere it was
  // probed, which is the good case and must not render as a missing number.
  return camera.foldMarginDeg == null ? "无折返" : `余量 ${camera.foldMarginDeg.toFixed(1)}°`;
}

/** What was measured, in terms that say what it does and does not imply. */
export function coverageNote(camera: IntrinsicsCoverageCamera, response: Thresholds): string {
  const verdict = coverageVerdict(camera, response);
  if (verdict === "unknown") {
    return "该内参文件没有自标定记录，覆盖未知——不能当作达标";
  }
  if (verdict === "folded") {
    return "畸变模型在画幅内折返，该区域像素无唯一光线；这是拟合本身的缺陷，需重标";
  }
  if (verdict === "extrapolated") {
    const band = Math.round((1 - (camera.coverage as number)) * 100);
    const margin = camera.foldMarginDeg;
    const tight = margin != null && margin < response.foldMarginWarnDeg ? `，折返点距画角仅 ${margin.toFixed(1)}°` : "";
    return `外侧 ${band}% 半径无角点${tight}；只有当这台相机实际用到那一圈时才构成误差`;
  }
  return "";
}

export function coverageRows(response: IntrinsicsCoverageResponse | null): CoverageRow[] {
  if (!response?.cameras?.length) {
    return [];
  }
  return response.cameras.map((camera) => ({
    camera: camera.camera,
    verdict: coverageVerdict(camera, response),
    coverage: formatCoverage(camera),
    fold: formatFold(camera),
    note: coverageNote(camera, response),
  }));
}

/** One sentence a person can act on without reading the table. */
export function coverageSummary(response: IntrinsicsCoverageResponse | null): string {
  if (!response) {
    return "尚未读取生产内参";
  }
  if (response.error) {
    return response.error;
  }
  const rows = coverageRows(response);
  if (rows.length === 0) {
    return "生产内参目录里没有可读的相机";
  }
  const named = (verdict: CoverageVerdict) => rows.filter((row) => row.verdict === verdict).map((row) => row.camera);
  const folded = named("folded");
  if (folded.length > 0) {
    return `${rows.length} 台中 ${folded.length} 台畸变模型在画幅内折返，需重标：${folded.join("、")}`;
  }
  const parts: string[] = [];
  const extrapolated = named("extrapolated");
  const unknown = named("unknown");
  if (extrapolated.length > 0) {
    parts.push(`${extrapolated.length} 台外侧为外推（${extrapolated.join("、")}）`);
  }
  if (unknown.length > 0) {
    parts.push(`${unknown.length} 台无法判定（${unknown.join("、")}）`);
  }
  if (parts.length === 0) {
    return `${rows.length} 台全部覆盖到 ${(response.coverageTarget * 100).toFixed(0)}% 半径以上，无外推区`;
  }
  return `${rows.length} 台中 ${parts.join("、")}——是否需要重标取决于这些相机实际用到多远`;
}

/**
 * Only a defect in the fit itself pushes someone into a re-shoot from here.
 * Thin coverage is reported, not acted on: on this rig the least-covered camera
 * never works in its unmeasured band, and the camera that does work past its
 * band was wrong for an unrelated (extrinsic) reason.
 */
export function shouldOfferIntrinsicsRecapture(response: IntrinsicsCoverageResponse | null): boolean {
  return coverageRows(response).some((row) => row.verdict === "folded");
}
