// Reading a tracker-station / lever-arm fit, kept out of the component so the
// wording of each verdict is pinned by a test rather than by a JSX diff.
//
// Three things this file exists to keep straight, all of them ways the panel
// could otherwise overstate what was measured:
//
//   1. A refusal is an answer, and *which* refusal matters. Exit 2 means this
//      capture cannot determine c -- go rotate about a second axis. Exit 1 means
//      it fitted and does not certify -- the numbers are there to read. Collapsed
//      into "failed", the first sends an operator to re-record data that is fine.
//   2. A fitted lever arm absorbs a constant body-frame translation *and* a
//      constant body-frame rotation, which together are exactly the marker->TCP
//      constant. So a small residual is never evidence that cube->TCP is right,
//      and the panel has to say so where the small residual is displayed.
//   3. The residual's *size* cannot tell a flexing mount from an attitude-
//      dependent pipeline -- the per-pose lever-arm spread is algebraically the
//      same number as the RMS. Only the attitude structure separates them.
import type {
  TrackerAttitudeDependence,
  TrackerMountObservability,
  TrackerMountReport,
  TrackerMountSigma,
  TrackerStationReport,
  TrackerValidateReport,
  TrackerValidateSummary,
  TrackerMountCapture,
} from "../types";

export function fmtMm(value: number | null | undefined, digits = 3): string {
  return value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(digits)} mm`;
}

export function fmtDeg(value: number | null | undefined, digits = 2): string {
  return value == null || !Number.isFinite(value) ? "—" : `${value.toFixed(digits)}°`;
}

export function fmtPpm(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(1)} ppm`;
}

export type TrackerMountOutcome = "solved" | "not_certified" | "cannot_run" | "error";

export type TrackerMountVerdict = {
  outcome: TrackerMountOutcome;
  dot: "running" | "warning" | "error" | "idle";
  title: string;
  detail: string;
};

/** What an exit code means, in the operator's terms rather than the tool's. */
export function trackerMountVerdict(
  result: { ok?: boolean; returncode?: number; error?: string; summary?: string } | null,
): TrackerMountVerdict | null {
  if (!result) return null;
  const code = result.returncode ?? (result.ok ? 0 : -1);
  if (result.ok && code === 0) {
    return {
      outcome: "solved",
      dot: "running",
      title: "解算通过",
      detail: result.summary || "拟合成立，且精度足以进误差预算。",
    };
  }
  if (code === 1) {
    return {
      outcome: "not_certified",
      dot: "warning",
      title: "解出来了，但不认证",
      detail:
        (result.error || result.summary || "") +
        "（数值可读，但没通过「可用于评估轨迹」的门槛：看位姿数、bootstrap σ_c、holdout 三项）",
    };
  }
  if (code === 2) {
    return {
      outcome: "cannot_run",
      dot: "error",
      title: "这批采集跑不了",
      detail: result.error || "采集本身的问题：没有停稳的姿态，或者姿态不足以定出 c。",
    };
  }
  return { outcome: "error", dot: "error", title: "调用失败", detail: result.error || "未知错误" };
}

export type ObservabilityRow = { label: string; value: string; hint: string };

/**
 * Observability first, numbers second.
 *
 * ``c`` is determined by *rotation*: differencing two poses removes t_WG and
 * leaves (R_i - R_j) c, so a pure-translation pair contributes nothing at all.
 * ``c_gain_min`` is directly comparable to 2 sin(theta/2), which is why the
 * equivalent angle is what gets shown -- "ill-conditioned" is not an action,
 * "rotate about a second axis by this much" is.
 */
export function observabilityRows(obs: TrackerMountObservability | null | undefined): ObservabilityRow[] {
  if (!obs) return [];
  return [
    {
      label: "位姿数",
      value: String(obs.n_poses),
      hint: obs.station_frozen ? "站位已冻结，c 是线性解" : "站位与 c 一起解",
    },
    {
      label: "姿态张角",
      value: fmtDeg(obs.rotation_span_deg),
      hint: obs.fixed_attitude ? "定姿态采集：全部旋转误差都被 c 吸收" : "有姿态变化",
    },
    {
      label: "最弱方向增益",
      value: `${obs.c_gain_min.toFixed(4)}（≈ ${fmtDeg(obs.c_gain_min_equiv_deg)}）`,
      hint: "等效于只绕一根轴转这么多；要提高就换一根不平行的轴再转",
    },
    {
      label: "噪声放大",
      value: Number.isFinite(obs.c_sigma_amplification) ? `×${obs.c_sigma_amplification.toFixed(1)}` : "×∞",
      hint: "位姿噪声进到 c 的倍数，再除以 √N",
    },
  ];
}

/**
 * The mount-rigidity verdict.
 *
 * Deliberately not phrased as a gate. Structure here points at the mount or at
 * the pipeline's attitude dependence; which of the two it is takes a separate
 * measurement, and saying "flexing" from this number alone would be the same
 * overreach the residual-size version already made.
 */
export function attitudeVerdict(att: TrackerAttitudeDependence | null | undefined): {
  dot: "running" | "warning" | "idle";
  title: string;
  detail: string;
} {
  if (!att || att.n_poses <= 0) {
    return {
      dot: "idle",
      title: "未测量",
      detail: "这份结果没有姿态相关性诊断（旧产物）。「没测」不等于「刚性」。",
    };
  }
  if (att.n_poses < 6) {
    return {
      dot: "idle",
      title: "位姿太少，测不了",
      detail: `每个响应分量要拟合 4 个参数，${att.n_poses} 个位姿不够，判不了结构。`,
    };
  }
  if (att.structured) {
    return {
      dot: "warning",
      title: "残差与姿态相关",
      detail:
        `姿态能解释 ${(att.explained_frac * 100).toFixed(0)}% 的散布（纯噪声只会解释 ` +
        `${(att.null_explained_frac * 100).toFixed(0)}%），最坏方向 ${att.slope_mm_per_deg.toFixed(3)} mm/度。` +
        "指向两件事之一：装夹随姿态变形，或者相机侧的姿态误差本身与姿态相关。要分清得另做测量。",
    };
  }
  return {
    dot: "running",
    title: "未见姿态结构",
    detail:
      `姿态只解释 ${(att.explained_frac * 100).toFixed(0)}% 的散布，纯噪声的期望是 ` +
      `${(att.null_explained_frac * 100).toFixed(0)}%。散布看起来是无结构的，刚性假设没有被这批数据推翻。`,
  };
}

/** Does this fit have the precision the error budget's lever-arm row needs? */
export function sigmaVerdict(sigma: TrackerMountSigma | null | undefined, maxMm = 0.2): {
  dot: "running" | "warning" | "idle";
  text: string;
} {
  const value = sigma?.c_sigma_norm_mm;
  if (value == null || !Number.isFinite(value)) {
    return { dot: "idle", text: sigma?.note || "没有不确定度：这次拟合不能进预算表" };
  }
  return value <= maxMm
    ? { dot: "running", text: `σ_c = ${fmtMm(value)} ≤ ${maxMm} mm` }
    : { dot: "warning", text: `σ_c = ${fmtMm(value)} > ${maxMm} mm，精度不够进预算表` };
}

/**
 * Whether the held-out poses agree with the fitted ones.
 *
 * The only thing separating a lever arm that predicts from one fitted to its own
 * noise. Absent is reported as absent -- a fit run without a holdout has not
 * passed this check, it has skipped it.
 */
export function holdoutVerdict(report: TrackerMountReport | null | undefined, maxRatio = 2.0): {
  dot: "running" | "warning" | "idle";
  text: string;
} {
  if (!report) return { dot: "idle", text: "—" };
  const held = report.holdout_rms_mm;
  if (held == null || !Number.isFinite(held)) {
    return { dot: "idle", text: "没留 holdout：这一项是跳过了，不是通过了" };
  }
  const ratio = report.rms_mm > 0 ? held / report.rms_mm : Infinity;
  return ratio <= maxRatio
    ? { dot: "running", text: `holdout ${fmtMm(held)}（×${ratio.toFixed(2)}，≤${maxRatio}）` }
    : { dot: "warning", text: `holdout ${fmtMm(held)}（×${ratio.toFixed(2)}，>${maxRatio}：过拟合）` };
}

/**
 * The scale diagnostic, which is the one thing a laser tracker is uniquely
 * qualified to certify -- and which the registration deliberately does *not*
 * absorb. Fitting 7 parameters would hide a camera-pipeline scale error inside
 * T_WG and leave the residual looking clean; fitting 6 leaves it in the residual
 * where it can be seen, and reports the estimate separately.
 */
export function scaleVerdict(station: TrackerStationReport | null | undefined, warnPpm = 200): {
  dot: "running" | "warning" | "idle";
  text: string;
  detail: string;
} {
  if (!station || !Number.isFinite(station.scale_error_ppm)) {
    return { dot: "idle", text: "—", detail: "" };
  }
  const ppm = station.scale_error_ppm;
  const overOneMetre = Math.abs(ppm) * 1e-3; // ppm -> mm per metre
  const detail =
    `相机系统相对跟踪仪的尺度偏差，未施加（注册是刚性 6 参数）。` +
    `折合每米 ${overOneMetre.toFixed(3)} mm。`;
  return Math.abs(ppm) <= warnPpm
    ? { dot: "running", text: fmtPpm(ppm), detail }
    : { dot: "warning", text: fmtPpm(ppm), detail: detail + "偏差不小，值得查标定靶的实际尺寸。" };
}

const ABSORBED_LABEL: Record<string, string> = {
  constant_body_translation: "本体系常量平移",
  constant_body_rotation: "本体系常量旋转",
  all_rotation_error_at_fixed_attitude: "定姿态下的全部旋转误差",
};

export function absorbedLabel(mode: string): string {
  return ABSORBED_LABEL[mode] ?? mode;
}

/**
 * What the residual is silent about, stated wherever the residual is shown.
 *
 * The fit is a least-squares projection and will explain away any camera error
 * it can. The two it can -- a constant body-frame translation and a constant
 * body-frame rotation -- are together exactly the marker->TCP constant, which is
 * why a fitted lever arm can never certify cube->TCP no matter how small the
 * residual comes out.
 */
export function absorbedNote(modes: string[] | null | undefined): string {
  const list = (modes ?? []).map(absorbedLabel);
  if (!list.length) return "";
  return (
    `本次拟合会吸收：${list.join("、")}。` +
    "这些误差不会出现在残差里，所以残差再小也不能用来证明 cube→TCP 是对的。"
  );
}

// --- GT comparison: what the residual is, and what it is made of -------------

export function validateVerdict(
  result: { ok?: boolean; returncode?: number; error?: string; summary?: string } | null,
): TrackerMountVerdict | null {
  if (!result) return null;
  const code = result.returncode ?? (result.ok ? 0 : -1);
  if (result.ok && code === 0) {
    return {
      outcome: "solved",
      dot: "running",
      title: "比较完成，空间认证成立",
      detail: result.summary || "覆盖率达标，且注册来自评估轨迹之外的数据。",
    };
  }
  if (code === 1) {
    return {
      outcome: "not_certified",
      dot: "warning",
      title: "跑完了，但不认证",
      detail:
        (result.error || result.summary || "") +
        "（覆盖率不够，或者注册不是独立的——残差仍可读，但不能当认证用）",
    };
  }
  if (code === 2) {
    return {
      outcome: "cannot_run",
      dot: "error",
      title: "跑不了",
      detail: result.error || "配置问题：缺 sidecar、session 对不上、或时钟链接不起来。",
    };
  }
  return { outcome: "error", dot: "error", title: "调用失败", detail: result.error || "未知错误" };
}

/**
 * The camera-side time base, which is not a quality knob.
 *
 * Without the Argus sidecars the camera times fall back to the nominal N/fps
 * grid, which is episode-local and up to 55 ms from the hardware SOF. Residuals
 * computed on the two are not comparable, so this is reported as a *different
 * measurement* rather than as a slightly worse one.
 */
export function timeBaseVerdict(report: TrackerValidateReport | null | undefined): {
  dot: "running" | "warning" | "idle";
  text: string;
  detail: string;
} {
  const base = report?.camera_time_base;
  if (!base) return { dot: "idle", text: "—", detail: "" };
  if (base === "argus_sidecar_sof_plus_exposure") {
    return { dot: "running", text: "硬件 SOF（sidecar）", detail: "相机时刻取自 Argus sidecar 的硬件 SOF。" };
  }
  return {
    dot: "warning",
    text: "N/fps 名义栅格",
    detail:
      "没读到 Argus sidecar，相机时刻退回到 episode 本地的 N/fps 栅格，离硬件 SOF 最远 55 ms。" +
      "这是**另一个时间基准**，不是差一点的同一个——和用 sidecar 跑出来的残差不可比。",
  };
}

export type SpeedStratumRow = { label: string; count: number; p50: string; p95: string };

const STRATUM_LABEL: Record<string, string> = {
  speed_rest: "静止 (<0.05 m/s)",
  speed_slow: "慢 (0.05–0.2)",
  speed_medium: "中 (0.2–0.5)",
  speed_fast: "快 (>0.5)",
};

const STRATUM_ORDER = ["speed_rest", "speed_slow", "speed_medium", "speed_fast"];

/**
 * Residual by speed -- the split that separates geometry from timing.
 *
 * A constant time offset dt shows up as |v| * dt: invisible at rest and linear
 * in speed. One pooled RMS over a trajectory that spends most of its frames near
 * zero speed hides exactly that. So the rest bucket is the geometry error and
 * the growth across buckets is the timing error, and they are read as two
 * different numbers rather than one average.
 */
export function speedStrataRows(summary: TrackerValidateSummary | null | undefined): SpeedStratumRow[] {
  const strata = summary?.strata ?? {};
  return STRATUM_ORDER.filter((key) => strata[key]).map((key) => ({
    label: STRATUM_LABEL[key] ?? key,
    count: strata[key].count,
    p50: fmtMm(strata[key].p50_mm),
    p95: fmtMm(strata[key].p95_mm),
  }));
}

/**
 * Whether the residual grows with speed, i.e. whether a timing term is visible.
 *
 * Reported as a ratio and not a verdict on the exposure sign: this is how the
 * sign becomes measurable (run the same episode at -0.5 / 0 / +0.5 and take the
 * one that flattens it), but a single run cannot say which way it should go.
 */
export function timingSignal(summary: TrackerValidateSummary | null | undefined): {
  dot: "running" | "warning" | "idle";
  text: string;
  detail: string;
} {
  const rest = summary?.strata?.speed_rest?.p95_mm;
  const fast = summary?.strata?.speed_fast?.p95_mm ?? summary?.strata?.speed_medium?.p95_mm;
  if (rest == null || fast == null || !Number.isFinite(rest) || !Number.isFinite(fast)) {
    return { dot: "idle", text: "—", detail: "这条轨迹没有同时覆盖静止和运动，分不出几何项和时间项。" };
  }
  const grew = fast - rest;
  const detail =
    `静止段 p95 ${fmtMm(rest)} 是纯几何误差；运动段涨到 ${fmtMm(fast)}。` +
    "多出来的部分按 |v|·Δt 走，是时间基准的残差。" +
    "同一个 episode 用 −0.5 / 0 / +0.5 跑三遍，把这一项压平的那个就是曝光符号。";
  return grew > Math.max(0.3 * rest, 0.2)
    ? { dot: "warning", text: `+${fmtMm(grew)}（随速度增长）`, detail }
    : { dot: "running", text: `+${fmtMm(grew)}（未见速度相关增长）`, detail };
}

export function coverageVerdict(report: TrackerValidateReport | null | undefined): {
  dot: "running" | "warning" | "idle";
  text: string;
} {
  const summary = report?.summary;
  if (!summary) return { dot: "idle", text: "—" };
  const min = report?.min_coverage ?? 0.8;
  const pct = `${(summary.coverage * 100).toFixed(1)}% (${summary.n_paired}/${summary.n_camera_frames})`;
  return summary.coverage >= min
    ? { dot: "running", text: pct }
    : { dot: "warning", text: `${pct}，低于门槛 ${(min * 100).toFixed(0)}%` };
}

// --- One-click capture: what is blocking the solve, in the operator's terms --

/** Hard floors from the solver, restated once so the UI cannot drift from them. */
export const DWELLS_PER_EPISODE_MIN = 3;
export const POSES_TO_CERTIFY = 15;
export const DWELL_SECONDS_MIN = 2.0;

export type CaptureBlocker =
  | "none"
  | "not_connected"
  | "tracker_off"
  | "beam_waiting"
  | "no_session"
  | "ready_to_record"
  | "recording_in_flight"
  | "not_landed"
  | "tracker_silent";

export type CaptureReadiness = {
  blocker: CaptureBlocker;
  dot: "running" | "warning" | "error" | "idle";
  title: string;
  detail: string;
  usable: TrackerMountCapture[];
  /** Which of the capture buttons this state allows. */
  canConnect: boolean;
  canStartSession: boolean;
  canRecord: boolean;
  canSave: boolean;
  canDisconnect: boolean;
  canEndSession: boolean;
};

/** The recorder state this panel has to agree with, as Live Record renders it. */
export type RecorderLiveState = {
  /** The recorder is up and devices are open (anything but idle/error). */
  connected: boolean;
  /** The tracker was asked for on *this* Connect. */
  trackerEnabled: boolean;
  /** The beam is locked on the SMR. */
  trackerReady: boolean;
  /** The recorder's own sentence about the beam; shown verbatim. */
  trackerDetail: string;
  /** An episode is being recorded or is awaiting save/discard. */
  episodeInFlight: boolean;
  /**
   * A gateway-held mount session owns the recorder.
   *
   * Not derived from anything on this page: it is the same flag Live Record
   * reads to know why its StartEpisode is refused, so the two cannot disagree
   * about who holds the recorder.
   */
  sessionActive: boolean;
  /** The gateway's name for the run in flight; "" when none. */
  sessionName: string;
};

/**
 * Why the solve cannot run yet -- from the captures **and** the live recorder.
 *
 * Reading only the captures is what made this panel contradict Live Record:
 * with nothing recorded yet it said "先 Connect" while the recorder was armed
 * with the beam locked, because "no episodes on disk" and "not connected" are
 * different facts and only the first one was in scope. The live half is the
 * same `recording.laserTrackerReady` the other page renders, so the two cannot
 * disagree any more.
 *
 * Disk state wins when it is decisive, because it describes work already done:
 * once dwells are recorded the next action is Disconnect, not Connect, and at
 * that moment the recorder is still connected. Only when the captures have
 * nothing to say does the live state decide what to do next.
 *
 * The one that costs a session if it is not said out loud is ``not_landed``:
 * the tracker session seals and lands at **Disconnect**, not when an episode
 * ends. Between the last dwell and Disconnect the episodes are all correct and
 * none of them is usable, and a solver error two clicks later does not explain
 * that.
 */
export function captureReadiness(
  captures: TrackerMountCapture[],
  live: RecorderLiveState,
): CaptureReadiness {
  const buttons = {
    // Mirrors Live Record, where Connect greys out once the recorder is up.
    canConnect: !live.connected,
    canStartSession: !live.sessionActive,
    // A dwell needs somewhere to go and a name to go under, and the gateway
    // holds both. Without the session the press is refused server-side anyway;
    // greying it out is how that stops being a surprise.
    canRecord: live.connected && live.sessionActive && !live.episodeInFlight,
    canSave: live.connected && live.episodeInFlight,
    // Disconnect **discards** an episode in flight: the gateway sends `q` to a
    // recording recorder and `n\nexit` to one awaiting review, both of which
    // throw the take away, and the tracker session then seals with no episode
    // boundaries at all. So it is closed off until the take has been saved --
    // this is not a style choice, it is the difference between a session and
    // nothing.
    canDisconnect: live.connected && !live.episodeInFlight,
    // Releasing the claim mid-take would let Live Record queue a task episode
    // into the middle of this one.
    canEndSession: live.sessionActive && !live.episodeInFlight,
  };
  if (live.connected && live.episodeInFlight) {
    return {
      blocker: "recording_in_flight",
      dot: "warning",
      title: "正在录这一段",
      detail:
        "摆完最后一个姿态后点「保存本段」，或者等计时自己收尾。" +
        "现在点 Disconnect 会把这一段**丢掉**——录制器收到的是 q/n，不是保存，" +
        "而且 tracker session 照样会 seal，只是里面一个 episode 边界都没有。",
      usable: captures.filter((c) => c.landed),
      ...buttons,
    };
  }
  if (!captures.length) {
    if (!live.connected) {
      return {
        blocker: "not_connected",
        dot: "idle",
        title: "还没连接",
        detail: "先 Connect（带跟踪仪），连上之后跟踪仪要锁到 SMR 才能开录。",
        usable: [],
        ...buttons,
      };
    }
    if (!live.trackerEnabled) {
      return {
        blocker: "tracker_off",
        dot: "error",
        title: "这次 Connect 没带跟踪仪",
        detail:
          "录制器连上了，但这一次没有启用跟踪仪，录出来的段不会有 session。" +
          "先 Disconnect，再勾上跟踪仪重新 Connect。",
        usable: [],
        ...buttons,
      };
    }
    if (!live.trackerReady) {
      return {
        blocker: "beam_waiting",
        dot: "warning",
        title: "跟踪仪还没锁上 SMR",
        detail: live.trackerDetail || "光束还没锁定，homing 会自动重试。锁上之前录出来的段没有跟踪仪数据。",
        usable: [],
        ...buttons,
      };
    }
    if (!live.sessionActive) {
      return {
        blocker: "no_session",
        dot: "idle",
        title: "还没开始一次站位采集",
        detail:
          "点「开始一次站位采集」。这一步是向网关认领录制器：session 名由网关生成并持有，" +
          "刷新页面不会改名，采集页也会看到录制器被占用，不会把任务 episode 插进来。",
        usable: [],
        ...buttons,
      };
    }
    return {
      blocker: "ready_to_record",
      dot: "running",
      title: `已连接，跟踪仪已锁定 SMR（${live.sessionName}）`,
      detail: `可以开录。${live.trackerDetail}`.trim(),
      usable: [],
      ...buttons,
    };
  }
  const landed = captures.filter((c) => c.landed);
  if (!landed.length) {
    return {
      blocker: "not_landed",
      dot: "warning",
      title: "录到了，但 session 还没落地",
      detail:
        "跟踪仪 session 是在 Disconnect 时才 seal + land 的，不是每段结束就落。" +
        "现在这些段本身没问题，只是还取不到数据——点「Disconnect 并落地」。",
      usable: [],
      ...buttons,
    };
  }
  const silent = landed.filter((c) => !c.streamAdvanced || c.beamValidFraction === 0);
  if (silent.length === landed.length) {
    return {
      blocker: "tracker_silent",
      dot: "error",
      title: "session 落地了，但跟踪仪没数据",
      detail:
        "每段的流都没有推进，或者光束全程无效。多半是没锁上目标，或者 responder 这次没重启。" +
        "重录之前先确认跟踪仪确实锁在 SMR 上。",
      usable: landed,
      ...buttons,
    };
  }
  return {
    blocker: "none",
    dot: "running",
    title: `可用录制 ${landed.length} 段`,
    detail: `每段内部按停驻切分位姿，单段至少要 ${DWELLS_PER_EPISODE_MIN} 个停驻才收。`,
    usable: landed,
    ...buttons,
  };
}

/** One line per capture, with the tracker's own health next to it. */
export function captureLabel(capture: TrackerMountCapture): string {
  const beam =
    capture.beamValidFraction < 0
      ? "光束未知"
      : `光束有效 ${(capture.beamValidFraction * 100).toFixed(0)}%`;
  const landed = capture.landed ? "" : " · 未落地";
  return `${capture.datasetName} ep${capture.episode} · ${capture.sessionId} · ${beam}${landed}`;
}

/**
 * How long one dwell recording has to be.
 *
 * Dwells are segmented *inside* one episode from the tracker stream, so the
 * poses come from pauses within a single recording rather than from separate
 * ones -- an episode holding a single dwell is refused outright. At
 * `DWELL_SECONDS_MIN` still per pose plus the trimmed settling and the time to
 * re-orient between them, certifying needs a couple of minutes in one take.
 */
export function suggestedDwellSeconds(poses = POSES_TO_CERTIFY): number {
  const perPose = DWELL_SECONDS_MIN + 0.6 + 4.0; // still + trimmed ends + re-orient
  return Math.ceil((poses * perPose) / 10) * 10;
}
