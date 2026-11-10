// FR3 end-effector load (Desk) and BOX 6D payload fit: response types and the pure logic the
// two panels show. Gateway side: tools/data_collection_gui/fr3_payload_calibration.py.

export type SixDFitSummary = {
  error?: string;
  meetsTarget?: boolean;
  heldoutWorstForceN?: number | null;
  heldoutWorstMomentNm?: number | null;
  nStatic?: number;
  attitudes?: Record<string, number>;
  missingAttitudes?: string[];
  massKg?: number | null;
  axisGainKg?: number[] | null;
  comInFtM?: number[] | null;
  K?: number[][] | null;
  /** BOX clock (s since power-up) at the first sample; null when the log has none. */
  uptimeStartS?: number | null;
  drift?: SixDDrift | null;
  warnings?: string[];
};

/** box_six_d's same-attitude drift: the largest force change between two holds of one attitude. */
export type SixDDrift = {
  pairs: number;
  worstForceN?: number | null;
  axis?: string | null;
  changeN?: number | null;
  fromS?: number | null;
  toS?: number | null;
  maxAbsByAxisN?: number[] | null;
};

export type PayloadRun = {
  name: string;
  created: string;
  arm: string;
  robotIp: string;
  toolConfig: string;
  posesSet: string;
  complete: boolean;
  nPoses: number;
  twoWay: boolean;
  note: string;
  deskAtStart: number | null;
  sixD: { deviceId: number | null; log: string; fit: SixDFitSummary | null } | null;
};

export type DeskConfigEntry = {
  toolConfig: string;
  deskName: string;
  massKg: number;
  comFlangeM: number[];
  massStdG: number;
  predictedForceMeanN: number[];
  predictedForceMaxN: number;
  references: Record<string, { mass_kg: number; com_flange_m: number[] }>;
  cad: { mass_kg?: number; com_flange_m?: number[] };
  deskAtStart: {
    run: string;
    m_ee: number;
    measured_force_mean_n: number[];
    measured_force_max_n: number;
    reproduction_rms: number;
  }[];
  runs: string[];
  nPoses: number;
  deskJson: string;
  tcp: unknown;
};

export type TcpSource = { kind: "estimate" | "pivot" | "manual" | "template"; text: string; pivot?: string };
export type DeskConfigSet = { name: string; tcpSource?: TcpSource | null; configs: DeskConfigEntry[] };

export type PivotFit = {
  name: string;
  joint: boolean;
  created: string;
  robotIp: string;
  note: string;
  frame: string;
  sEeMm: number[];
  sigmaMm: number[] | number | null;
  residualRmsMm: number | null;
};

export const TCP_SOURCE_LABELS: Record<TcpSource["kind"], string> = {
  estimate: "新指尖估计",
  pivot: "FR3 pivot 实测",
  manual: "手动 z",
  template: "模板（旧指尖）",
};

/** The request body's ``tcp`` for desk-config; null when the choice is incomplete. */
export function deskTcpRequest(
  source: TcpSource["kind"],
  opts: { zM: string; pivot: string },
): { source: string; zM?: number; pivot?: string } | null {
  if (source === "manual") {
    const z = Number(opts.zM);
    return opts.zM.trim() !== "" && Number.isFinite(z) ? { source, zM: z } : null;
  }
  if (source === "pivot") return opts.pivot ? { source, pivot: opts.pivot } : null;
  return { source };
}

export type SixDCapture = {
  name: string;
  deviceId: number | null;
  endEffector: string;
  startedUnix: number | null;
  state: string;
  faces: Record<string, number>;
  rows: number;
  need: number | null;
  fit: SixDFitSummary | null;
  registered: { at_unix: number; valid_from_unix: number } | null;
};

export type BoxRegistryEntry = {
  deviceId: string;
  role: string;
  hand?: string;
  records: {
    validFromUnix: number;
    validFrom: string;
    endEffector: string;
    fitted: boolean;
    status: string;
    heldoutWorstForceN: number | null;
  }[];
};

export type PoseCheck = {
  arm: string;
  poses: string;
  pass: boolean;
  atUnixS: number;
  table: { edgeM: number; topM: number; marginMm: number };
  summary: string[];
};

export type ZeroCheck = { id: string; label: string; pass: boolean; detail: string };

export type ZeroReport = {
  name: string;
  created: string;
  arm: string;
  deviceId: number | null;
  mode: string;
  pass: boolean;
  state: string;
  checks: ZeroCheck[];
  warnings: string[];
};

export type PayloadJob = {
  kind?: "check" | "collect" | "six_d_capture" | "six_d_zero";
  label?: string;
  state?: "running" | "fitting" | "done" | "error" | "stopped";
  log?: string[];
  error?: string;
  progress?: { done: number; total: number };
  startedUnixS?: number;
  endedUnixS?: number;
  arm?: string;
  poses?: string;
  toolConfig?: string;
  sixD?: boolean;
  deviceId?: number | null;
  run?: string;
  name?: string;
  durationS?: number;
  // six_d_zero: the step it is on, what the operator must do now, and the verdicts so far
  mode?: string;
  phase?: string;
  prompt?: string;
  checks?: ZeroCheck[];
  warnings?: string[];
  pass?: boolean;
  report?: string;
  recheck?: boolean;
  capture?: {
    state?: string;
    faces?: Record<string, number>;
    rows?: number;
    need?: number;
    picked_up?: boolean;
    done?: boolean;
    opening_mm?: number | null;
    error?: string;
  } | null;
};

export type PayloadCalibrationResponse = {
  ok: boolean;
  error: string;
  output?: string;
  deskWritten?: string;
  registered?: string;
  job: PayloadJob;
  arms: { arm: string; inService: boolean; robotIp: string; gripperBox: number | null; payload: unknown }[];
  toolConfigs: { id: string; label: string }[];
  poseSets: { id: string; label: string; poses: number }[];
  table: { edgeM: number; topM: number; marginMm: number };
  checks: Record<string, PoseCheck>;
  checkValidS: number;
  runs: PayloadRun[];
  desks: DeskConfigSet[];
  captures: SixDCapture[];
  boxes: BoxRegistryEntry[];
  registryPath: string;
  runsRoot: string;
  captureRoot: string;
  captureDurationRangeS: [number, number];
  pivotFits?: PivotFit[];
  tcpEstimateZM?: number;
  tcpTemplateZM?: number;
  zeroReports?: ZeroReport[];
  /** newest passing report per (arm, BOX, mode): the BOX tab dates an arm BOX's zeros from it */
  zeroLatest?: ZeroReport[];
  zeroPoses?: string;
  zeroReportRoot?: string;
};

/** check-poses pose set of the arm-held 6D zero route (start -> fingertips up -> start -> two re-checks). */
export const ZERO_POSES = "six_d_zero";

export const ZERO_MODE_LABELS: Record<string, string> = {
  dynamic: "动态校零",
  origin: "origin + 重新上电 + 动态校零",
};

/** Whether a job belongs to the zero panel (its own route check included) rather than the load panel. */
export function isZeroJob(job: PayloadJob | null | undefined): boolean {
  return job?.kind === "six_d_zero" || (job?.kind === "check" && job.poses === ZERO_POSES);
}

/** Why the arm-held zero cannot start yet, as far as the page can tell (the gateway checks the recorder too). */
export function zeroBlocker(payload: PayloadCalibrationResponse | null, arm: string, nowS: number): string {
  if (!payload) return "网关没有响应";
  if (isJobActive(payload.job)) return "已有任务在进行";
  if (!arm) return "先选机械臂";
  const row = payload.arms.find((a) => a.arm === arm);
  if (row && row.gripperBox == null) return "这条臂没有登记夹爪 BOX";
  const check = payload.checks[checkKey(arm, ZERO_POSES)];
  if (!check) return "先跑零点路线的碰撞检查（含桌面）";
  if (!check.pass) return "零点路线的碰撞检查没有通过";
  if (nowS - check.atUnixS > payload.checkValidS) return "碰撞检查已过期，重新检查";
  return "";
}

// The six faces are the BOX IMU's axes (which one points up), not the gripper's. What they mean on
// the gripper comes from the fitted K of 3969377039 (11-03 and 11-06 fits agree): its force block is
// -m*g*R_ft_imu with IMU x = FT -z, IMU y = FT -x, IMU z = FT +y, and the fingertips sit at FT +z with
// the fingers opening along FT x (corenetic_gripper_ldt_tip URDF). Up/down of the fingertips holds for
// any mounting turned about FT z; A/B within a pair is only a name (the rig's L/R and the URDF's
// left/right fingers disagree, so neither is used).
export const FACES = ["x-", "x+", "y+", "y-", "z+", "z-"] as const;
export const FACE_LABEL: Record<string, string> = {
  "x-": "指尖朝上",
  "x+": "指尖朝下",
  "y+": "侧放·两指上下 A",
  "y-": "侧放·两指上下 B",
  "z+": "侧放·两指并排 A",
  "z-": "侧放·两指并排 B",
};
export const faceLabel = (face: string): string => FACE_LABEL[face] ?? face;
export const SIX_D_TARGET_N = 0.3;

export const TOOL_CONFIG_SHORT: Record<string, string> = {
  gripper: "仅夹爪",
  gripper_cam: "夹爪+相机",
  gripper_cam_smr: "夹爪+相机+SMR",
};

export function isJobActive(job: PayloadJob | null | undefined): boolean {
  return job?.state === "running" || job?.state === "fitting";
}

export function checkKey(arm: string, poses: string): string {
  return `${arm || "-"}|${poses}`;
}

/** Why a collect cannot start yet (the gateway refuses the same), or "". */
export function collectBlocker(
  payload: PayloadCalibrationResponse | null,
  opts: { arm: string; poses: string; toolConfig: string; sixD: boolean },
  nowS: number,
): string {
  if (!payload) return "网关没有响应";
  if (isJobActive(payload.job)) return "已有任务在进行";
  if (!opts.arm) return "先选机械臂";
  if (!opts.toolConfig) return "先选末端配置";
  const check = payload.checks[checkKey(opts.arm, opts.poses)];
  if (!check) return "先跑碰撞检查（含桌面）";
  if (!check.pass) return "碰撞检查没有通过";
  if (nowS - check.atUnixS > payload.checkValidS) return "碰撞检查已过期，重新检查";
  if (opts.sixD) {
    const arm = payload.arms.find((a) => a.arm === opts.arm);
    if (arm && arm.gripperBox == null) return "这条臂没有登记夹爪 BOX，不能记录 6D";
  }
  return "";
}

export function jobPercent(job: PayloadJob | null | undefined): number | null {
  const p = job?.progress;
  if (!p || !p.total) return null;
  return Math.max(0, Math.min(100, Math.round((100 * p.done) / p.total)));
}

export function fmtG(kg: number | null | undefined): string {
  return kg == null || !Number.isFinite(kg) ? "—" : `${(kg * 1000).toFixed(1)} g`;
}

export function fmtMm(m: number[] | null | undefined): string {
  return m && m.length ? `(${m.map((v) => (v * 1000).toFixed(1)).join(", ")}) mm` : "—";
}

export function fmtN(v: number[] | null | undefined, digits = 2): string {
  return v && v.length ? `(${v.map((x) => x.toFixed(digits)).join(", ")})` : "—";
}

/** The ``payload`` entry for tools/fr3/fr3_arms.json (pasted by hand: config changes stay manual). */
export function fr3ArmsPayloadSnippet(entry: DeskConfigEntry, setName: string): string {
  return JSON.stringify(
    {
      mass_kg: entry.massKg,
      com_m: entry.comFlangeM,
      source: `${entry.deskName}: calibration page desk-config ${setName} (${entry.runs.join(", ")}), fit without joint offsets`,
    },
    null,
    0,
  );
}

/** Per face: samples so far, and the fraction of ``need`` (capped at 1). */
export function faceProgress(
  faces: Record<string, number> | undefined,
  need: number | undefined,
): { face: string; n: number; frac: number }[] {
  const target = need && need > 0 ? need : 30;
  return FACES.map((face) => {
    const n = faces?.[face] ?? 0;
    return { face, n, frac: Math.min(1, n / target) };
  });
}

export type SixDVerdict = { tone: "ok" | "warn" | "bad"; text: string };

export function sixDVerdict(fit: SixDFitSummary | null | undefined): SixDVerdict {
  if (!fit) return { tone: "warn", text: "未拟合" };
  if (fit.error) return { tone: "bad", text: `拟合被拒：${fit.error}` };
  const f = fit.heldoutWorstForceN;
  const m = fit.heldoutWorstMomentNm;
  const missing = fit.missingAttitudes?.length ? `，缺 ${fit.missingAttitudes.map(faceLabel).join("、")}` : "";
  const text = `留一面最差 ${f ?? "—"} N / ${m ?? "—"} N·m${missing}`;
  return fit.meetsTarget ? { tone: "ok", text: `通过（≤${SIX_D_TARGET_N} N）· ${text}` } : { tone: "bad", text: `未达标 · ${text}` };
}

/** One line on how far one attitude drifted over the capture; "" for a fit made before the check. */
export function sixDDriftText(fit: SixDFitSummary | null | undefined): string {
  const d = fit && !fit.error ? fit.drift : null;
  if (!d) return "";
  if (!d.pairs) return "同一朝向漂移：没有隔 30 s 以上重复的朝向，估计不了";
  const minutes = ((d.toS ?? 0) - (d.fromS ?? 0)) / 60;
  const change = d.changeN ?? 0;
  const axes = d.maxAbsByAxisN?.length === 3 ? `；每轴最大 Fx/Fy/Fz ${d.maxAbsByAxisN.map((v) => v.toFixed(2)).join(" / ")} N` : "";
  return `同一朝向漂移：${d.axis ?? "—"} ${change >= 0 ? "+" : ""}${change.toFixed(2)} N / ${minutes.toFixed(1)} 分钟${axes}`;
}

/** What to show under a fit's verdict: its warnings, and the drift line when no warning already says it. */
export function sixDFitNotes(fit: SixDFitSummary | null | undefined): { tone: "warn" | "muted"; text: string }[] {
  if (!fit || fit.error) return [];
  const notes: { tone: "warn" | "muted"; text: string }[] = (fit.warnings ?? []).map((text) => ({ tone: "warn", text }));
  const d = fit.drift;
  const warned = !!d && d.pairs > 0 && (d.worstForceN ?? 0) > SIX_D_TARGET_N;
  const line = sixDDriftText(fit);
  if (line && !warned) notes.push({ tone: "muted", text: line });
  return notes;
}

/** ``YYYY-MM-DDTHH:MM`` (local) for a datetime-local input. */
export function toLocalInput(unixS: number | null | undefined): string {
  if (!unixS) return "";
  const d = new Date(unixS * 1000);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function fromLocalInput(value: string): number | null {
  if (!value) return null;
  const t = new Date(value).getTime();
  return Number.isFinite(t) ? Math.round(t / 1000) : null;
}

/**
 * BOXes that can be turned by hand: handheld first, then any other non-arm role. A BOX on an FR3
 * gripper (box_devices.json, or named by an arm in fr3_arms.json) is left out -- it cannot be turned
 * through six faces while mounted; the arm does that (负载辨识 with 6D).
 */
export function captureDeviceChoices(
  boxes: BoxRegistryEntry[],
  armBoxes: (number | string | null | undefined)[] = [],
): { id: string; label: string }[] {
  const onArm = new Set(armBoxes.filter((id) => id != null).map(String));
  const label = (b: BoxRegistryEntry) =>
    `${b.deviceId} · ${b.role === "handheld" ? `手持 ${b.hand === "left" ? "左" : b.hand === "right" ? "右" : ""}` : b.role}`;
  return boxes
    .filter((b) => b.role !== "fr3_gripper" && !onArm.has(b.deviceId))
    .sort((a, b) => Number(b.role === "handheld") - Number(a.role === "handheld"))
    .map((b) => ({ id: b.deviceId, label: label(b) }));
}
