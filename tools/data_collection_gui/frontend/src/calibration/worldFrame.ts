// Presentation logic for the canonical world frame (roadmap 2.4), kept out of
// the component so the wording of a verdict can be pinned by a test.
//
// The rule this file exists to enforce: a break in world continuity is not a
// failure and must not be worded as one. Relative motion, bimanual pose and
// contact-local trajectories survive it untouched -- only cross-session
// *absolute* replay needs the link. Telling an operator their data is broken
// would send them to re-record something that is fine.
import type {
  WorldConsensus,
  WorldContinuityState,
  WorldFrameResponse,
  WorldReferenceSummary,
  WorldRegistration,
  WorldStableSource,
} from "../types";

export const worldStateLabel: Record<WorldContinuityState, string> = {
  CONTINUOUS: "同一世界系",
  RECONNECTED: "已接回原世界系",
  BROKEN: "世界系已断开",
};

export const worldStateDot: Record<WorldContinuityState, string> = {
  CONTINUOUS: "running",
  RECONNECTED: "warning",
  BROKEN: "error",
};

const REASON_LABEL: Record<string, string> = {
  stable_cluster: "稳定相机共识成立",
  degraded_cluster: "稳定相机数量偏少，需人工确认",
  consensus_ambiguous: "存在多个同样大的一致相机组，无法自动判定",
  alignment_residual: "对齐残差过大，两套解算不像同一个 rig",
  independent_anchor: "由独立锚点恢复",
  no_stable_cluster: "没有足够的稳定相机",
};

export function worldReasonLabel(reason: string): string {
  return REASON_LABEL[reason] ?? reason;
}

export function formatMm(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(2)} mm`;
}

export function formatDeg(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(3)}°`;
}

export type WorldCameraRole = "stable" | "moved" | "new" | "missing";

export const worldRoleLabel: Record<WorldCameraRole, string> = {
  stable: "稳定基准",
  moved: "已移动",
  new: "新增",
  missing: "本次未解出",
};

export const worldRoleDot: Record<WorldCameraRole, string> = {
  stable: "running",
  moved: "error",
  new: "warning",
  missing: "idle",
};

export type WorldCameraRow = {
  camera: string;
  role: WorldCameraRole;
  detail: string;
};

function roleOf(consensus: WorldConsensus, camera: string): WorldCameraRole {
  if (consensus.stable_cameras.includes(camera)) return "stable";
  if (consensus.moved_cameras.includes(camera)) return "moved";
  if (consensus.new_cameras.includes(camera)) return "new";
  return "missing";
}

/** One row per camera: what it was judged to be, and the number behind it. */
export function worldCameraRows(registration: WorldRegistration | null): WorldCameraRow[] {
  if (!registration) return [];
  const consensus = registration.consensus;
  const cameras = [
    ...consensus.stable_cameras,
    ...consensus.moved_cameras,
    ...consensus.new_cameras,
    ...consensus.missing_cameras,
  ];
  return [...new Set(cameras)].sort().map((camera) => {
    const role = roleOf(consensus, camera);
    const translation = registration.alignment?.translation_residual_mm?.[camera];
    const rotation = registration.alignment?.rotation_residual_deg?.[camera];
    let detail = "";
    if (role === "stable" && translation != null) {
      // The residual of a camera that defined the frame is the check on the
      // others, so it is the number worth showing.
      detail = `对齐残差 ${formatMm(translation)} · ${formatDeg(rotation)}`;
    } else if (role === "stable") {
      detail = "参与世界系注册";
    } else if (role === "moved") {
      detail = "已排除出基准，仅更新自身位姿";
    } else if (role === "new") {
      detail = "不在冻结基准中，由共享变换定位";
    } else {
      detail = "冻结基准里有，本次标定没解出";
    }
    return { camera, role, detail };
  });
}

/** The headline numbers of the shared fit, or why there is no fit. */
export function alignmentSummary(registration: WorldRegistration | null): string {
  if (!registration) return "";
  const alignment = registration.alignment;
  if (!alignment) {
    return "本次没有可用的世界系对齐——没有可信的稳定相机组。";
  }
  if (alignment.cameras_used.length === 0) {
    return "由独立锚点单点确定，没有相机残差可交叉验证。";
  }
  return [
    `${alignment.cameras_used.length} 台相机参与`,
    `残差 ${formatMm(alignment.rms_translation_mm)} RMS（最大 ${formatMm(alignment.max_translation_mm)}）`,
    `${formatDeg(alignment.rms_rotation_deg)} RMS`,
    `世界系自身 σ ${formatMm(alignment.sigma_world_translation_mm)} / ${formatDeg(alignment.sigma_world_rotation_deg)}`,
  ].join(" · ");
}

/** The blind spot, stated. Silence here would read as "checked and fine". */
export function commonModeSummary(registration: WorldRegistration | null): string {
  const commonMode = registration?.common_mode;
  if (!commonMode) return "";
  if (!commonMode.observable) {
    return "整架相机一起移动无法由相机之间的几何发现；本次没有独立锚点观测，这一项未被检查。";
  }
  const drift = `${formatMm(commonMode.translation_mm)} · ${formatDeg(commonMode.rotation_deg)}`;
  return commonMode.drifted
    ? `独立锚点显示整架相机相对环境移动了 ${drift}——世界系内部自洽，但相对桌面/地面已经不同。`
    : `独立锚点复核通过（${drift}），整架相机没有共同移动。`;
}

export function referenceSummary(reference: WorldReferenceSummary | undefined): string {
  if (!reference?.exists) {
    return "尚未冻结基准世界系——在此之前每次外参标定都会重新定义世界，历史绝对轨迹无法互相比较。";
  }
  const parts = [reference.world_frame_id ?? "?", reference.created_utc ?? "未知时间"];
  if (reference.cameras?.length) parts.push(`${reference.cameras.length} 台相机`);
  if (reference.calibration_id) parts.push(`来源 ${reference.calibration_id}`);
  return parts.join(" · ");
}

/** Only worth showing once more than one world exists. */
export function graphSummary(payload: WorldFrameResponse | null): string {
  const graph = payload?.graph;
  if (!graph || graph.worlds <= 1) return "";
  return `已有 ${graph.worlds} 个世界系、${graph.edges} 条注册边——跨世界系的绝对回放需要先补上对应的注册边。`;
}

/** Where the stable set came from. Never left implicit: it decides everything.
 *
 * The two measurements have very different floors -- the image self-check
 * resolves about 1.7 mm at 1 m, the geometric consensus about a centimetre --
 * so when the finer one is available it should decide, and the coarser one's
 * per-camera residual becomes an independent check on that decision. */
export function stableSourceSummary(source: WorldStableSource | undefined): string {
  if (!source) return "";
  if (source.origin === "operator") {
    return `稳定相机由操作者指定：${source.cameras?.join("、") || "—"}`;
  }
  if (source.origin === "rig_check") {
    const when = source.generatedUtc ?? "未知时间";
    const moved = source.moved?.length ? `，判为移动：${source.moved.join("、")}` : "";
    return `稳定相机取自 ${when} 的相机自检（${source.cameras?.join("、") || "—"}${moved}）——自检能分辨约 1.7 mm，比几何共识的约 1 cm 灵敏；下方逐相机残差是对这一判定的独立复核。`;
  }
  const why = source.reason ? `（${source.reason}）` : "";
  return `稳定相机由几何共识自行判定${why}——检出下限约 1 cm，更小的碰动要靠上方的相机自检发现。`;
}

/** Only offer the fallback when the self-check is actually driving. */
/** Why the registration on screen is history rather than this run's answer.
 *
 * ``_world_frame_payload`` always returns the last registration written to
 * disk, and the gateway attaches it to *failures* too so the panel does not go
 * blank while showing an error. Rendering it unlabelled is how "找不到自标定 BA
 * 结果，先跑一次外参标定" ended up sitting directly above "7 台相机全部未移动，
 * 世界系保持不变" -- two statements that cannot both describe the same run.
 * Returns "" when the block may stand as the current verdict.
 */
export function staleRegistrationNote(
  payload: WorldFrameResponse | null,
  fresh: boolean,
): string {
  const registration = payload?.registration;
  if (!registration || fresh) return "";
  const when = registration.generated_utc || "未知时间";
  const note = `以下是上次检测的结果（${when}），不是本次的结论。`;
  const active = payload?.extrinsicsRun ?? "";
  const registered = registration.calibration_id ?? "";
  if (active && registered && active !== registered) {
    // A registration is a statement about one bundle. Once the active run is a
    // different one, the verdict is not merely old -- it is about other data.
    return `${note}它注册的是标定 ${registered}，与当前生效的外参 ${active} 不是同一次。`;
  }
  return note;
}

export function canFallBackToGeometry(source: WorldStableSource | undefined): boolean {
  return source?.origin === "rig_check";
}

export function canApplyWorld(registration: WorldRegistration | null): boolean {
  return registration != null && !registration.committed;
}

/** Committing a break means minting a new island, so it must not say "update". */
export function applyLabel(registration: WorldRegistration | null): string {
  if (registration?.world_continuity_state === "BROKEN") {
    return "新建世界系（world island）";
  }
  return "接受并更新基准世界系";
}

export function applyConfirmation(registration: WorldRegistration | null): string {
  if (!registration) return "";
  if (registration.world_continuity_state === "BROKEN") {
    return (
      "将新建一个世界系，历史数据仍属于旧世界系且保持可用；" +
      "只有跨世界系的绝对回放会被阻断，直到有独立基准补上一条注册边。"
    );
  }
  const replaced = registration.consensus.moved_cameras.length + registration.consensus.new_cameras.length;
  return replaced > 0
    ? `将在同一个世界系内为 ${replaced} 台相机写入新位姿；稳定相机的基准位姿保持冻结不变。`
    : "本次没有需要更新的相机，提交只会记录一次确认。";
}

/** Whether the operator has to name the unmoved cameras before this can resolve. */
export function needsOperatorChoice(registration: WorldRegistration | null): boolean {
  return Boolean(registration?.consensus.ambiguous);
}

/** The candidate clusters an ambiguous consensus could not choose between. */
export function candidateClusters(registration: WorldRegistration | null): string[][] {
  if (!registration?.consensus.ambiguous) return [];
  return [registration.consensus.stable_cameras, ...registration.consensus.alternative_clusters];
}

/** Cameras the operator may tick as "did not move". */
export function selectableCameras(registration: WorldRegistration | null): string[] {
  if (!registration) return [];
  const consensus = registration.consensus;
  return [...new Set([...consensus.stable_cameras, ...consensus.moved_cameras])].sort();
}
