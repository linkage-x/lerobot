import { describe, expect, it } from "vitest";
import type {
  TrackerAttitudeDependence,
  TrackerMountObservability,
  TrackerMountReport,
  TrackerStationReport,
} from "../types";
import {
  absorbedNote,
  attitudeVerdict,
  holdoutVerdict,
  observabilityRows,
  scaleVerdict,
  sigmaVerdict,
  trackerMountVerdict,
} from "./trackerMount";

const OBS: TrackerMountObservability = {
  n_poses: 20,
  station_frozen: true,
  ok: true,
  rotation_span_deg: 42.0,
  fixed_attitude: false,
  c_gain_min: 0.31,
  c_gain_max: 0.98,
  c_gain_min_equiv_deg: 17.8,
  c_sigma_amplification: 3.2,
  planarity: 0.21,
  extent_m: 0.7,
  reasons: [],
};

const ATT: TrackerAttitudeDependence = {
  n_poses: 20,
  explained_frac: 0.05,
  null_explained_frac: 0.16,
  slope_mm_per_deg: 0.002,
  rotation_span_deg: 42.0,
  structured: false,
};

function mountReport(over: Partial<TrackerMountReport> = {}): TrackerMountReport {
  return {
    mount_id: "plate_v1",
    session_id: "s0",
    c_m: [0.02, -0.05, 0.15],
    lever_arm_mm: 160.1,
    rotation_sensitivity_mm_per_deg: 2.79,
    T_world_tracker: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    station_fitted: false,
    n_poses: 20,
    rms_mm: 0.08,
    max_mm: 0.19,
    holdout_rms_mm: 0.1,
    per_pose_c_spread_mm: 0.08,
    attitude: ATT,
    observability: OBS,
    sigma: {
      num_resamples: 200,
      c_sigma_mm: [0.03, 0.04, 0.05],
      c_sigma_norm_mm: 0.07,
      rotation_sigma_deg: null,
      translation_sigma_mm: null,
      note: "",
    },
    absorbed_modes: ["constant_body_translation", "constant_body_rotation"],
    certifies: true,
    certifies_marker_to_tcp: false,
    ...over,
  };
}

describe("a refusal is an answer, and which refusal matters", () => {
  it("keeps 'cannot run' apart from 'ran and does not certify'", () => {
    // Exit 2 is about the capture; exit 1 is about the fit. Collapsing them
    // sends an operator to re-record poses that were fine.
    const cannot = trackerMountVerdict({ ok: false, returncode: 2, error: "no dwell survived" });
    const notCertified = trackerMountVerdict({ ok: false, returncode: 1, summary: "rms 0.41 mm" });
    expect(cannot?.outcome).toBe("cannot_run");
    expect(notCertified?.outcome).toBe("not_certified");
    expect(cannot?.dot).not.toBe(notCertified?.dot);
  });

  it("does not call a non-zero exit a success", () => {
    expect(trackerMountVerdict({ ok: true, returncode: 1 })?.outcome).toBe("not_certified");
  });

  it("carries the tool's own words for a setup failure rather than rewording them", () => {
    const v = trackerMountVerdict({ ok: false, returncode: 2, error: "rotation about one axis only" });
    expect(v?.detail).toContain("rotation about one axis only");
  });
});

describe("observability is shown as an action, not as a condition number", () => {
  it("turns the weakest gain into the rotation that would fix it", () => {
    const rows = observabilityRows(OBS);
    const gain = rows.find((r) => r.label === "最弱方向增益");
    expect(gain?.value).toContain("17.80°");
    expect(gain?.hint).toContain("不平行");
  });

  it("says out loud when a fixed-attitude capture has absorbed every rotation error", () => {
    const rows = observabilityRows({ ...OBS, fixed_attitude: true, rotation_span_deg: 2.0 });
    expect(rows.find((r) => r.label === "姿态张角")?.hint).toContain("吸收");
  });
});

describe("mount rigidity", () => {
  it("reads a structureless cloud as 'not refuted', never as proof of rigidity", () => {
    const v = attitudeVerdict(ATT);
    expect(v.dot).toBe("running");
    expect(v.title).toBe("未见姿态结构");
    expect(v.detail).toContain("没有被这批数据推翻");
  });

  it("flags structure and refuses to name the cause from this number alone", () => {
    const v = attitudeVerdict({ ...ATT, explained_frac: 0.97, slope_mm_per_deg: 0.028, structured: true });
    expect(v.dot).toBe("warning");
    // Two candidate causes, and it says so: the diagnostic cannot separate a
    // flexing plate from an attitude-dependent pipeline on its own.
    expect(v.detail).toContain("变形");
    expect(v.detail).toContain("相机侧");
  });

  it("does not let an old artifact's silence read as a clean bill", () => {
    const v = attitudeVerdict(null);
    expect(v.dot).toBe("idle");
    expect(v.detail).toContain("「没测」不等于「刚性」");
  });

  it("refuses to judge structure from too few poses", () => {
    expect(attitudeVerdict({ ...ATT, n_poses: 5 }).dot).toBe("idle");
  });
});

describe("what the residual is silent about", () => {
  it("names the absorbed modes wherever the residual is shown", () => {
    const note = absorbedNote(["constant_body_translation", "constant_body_rotation"]);
    expect(note).toContain("本体系常量平移");
    expect(note).toContain("本体系常量旋转");
  });

  it("says a small residual cannot vindicate cube->TCP", () => {
    // The two absorbed modes together *are* the marker->TCP constant, so this
    // sentence has to travel with the number, not sit in a help modal.
    expect(absorbedNote(["constant_body_translation"])).toContain("不能用来证明 cube→TCP 是对的");
  });

  it("says nothing when nothing was absorbed", () => {
    expect(absorbedNote([])).toBe("");
  });
});

describe("precision and prediction are separate gates", () => {
  it("passes a tight sigma and fails a loose one", () => {
    expect(sigmaVerdict({ ...mountReport().sigma }).dot).toBe("running");
    expect(sigmaVerdict({ ...mountReport().sigma, c_sigma_norm_mm: 0.4 }).dot).toBe("warning");
  });

  it("treats a missing sigma as disqualifying, not as fine", () => {
    const v = sigmaVerdict({ ...mountReport().sigma, c_sigma_norm_mm: null, note: "too few poses" });
    expect(v.dot).toBe("idle");
    expect(v.text).toContain("too few poses");
  });

  it("reports a skipped holdout as skipped", () => {
    const v = holdoutVerdict(mountReport({ holdout_rms_mm: null }));
    expect(v.text).toContain("跳过了，不是通过了");
    expect(v.dot).toBe("idle");
  });

  it("flags a holdout that is much worse than the fit", () => {
    expect(holdoutVerdict(mountReport({ rms_mm: 0.08, holdout_rms_mm: 0.5 })).dot).toBe("warning");
  });
});

describe("scale is a diagnostic, never applied", () => {
  const station = {
    scale_error_ppm: 120,
    scale_applied: false,
  } as TrackerStationReport;

  it("says it was not applied, and converts ppm into something physical", () => {
    const v = scaleVerdict(station);
    expect(v.text).toBe("+120.0 ppm");
    expect(v.detail).toContain("未施加");
    expect(v.detail).toContain("0.120 mm");
  });

  it("points at the calibration target when the scale error is large", () => {
    expect(scaleVerdict({ ...station, scale_error_ppm: 900 } as TrackerStationReport).detail).toContain(
      "标定靶",
    );
  });
});

// --- GT comparison ----------------------------------------------------------

import type { TrackerValidateReport, TrackerValidateSummary } from "../types";
import {
  coverageVerdict,
  speedStrataRows,
  timeBaseVerdict,
  timingSignal,
  validateVerdict,
} from "./trackerMount";

function summary(over: Partial<TrackerValidateSummary> = {}): TrackerValidateSummary {
  return {
    n_paired: 900,
    n_camera_frames: 1000,
    coverage: 0.9,
    residual_mm: { count: 900, p50_mm: 0.4, p95_mm: 0.9 },
    strata: {
      speed_rest: { count: 300, p50_mm: 0.2, p95_mm: 0.4 },
      speed_slow: { count: 300, p50_mm: 0.3, p95_mm: 0.6 },
      speed_fast: { count: 300, p50_mm: 0.5, p95_mm: 1.9 },
    },
    clock: {},
    registration: { source: "parked_poses", absorbed_modes: [], certifies_space: true },
    lever_arm_mm: [20, -50, 150],
    rotation_sensitivity_mm_per_deg: 2.79,
    interp_error_mm_bound: 0.01,
    time_crosscheck_s: null,
    certifies_space: true,
    ...over,
  };
}

function validateReport(over: Partial<TrackerValidateReport> = {}): TrackerValidateReport {
  return {
    dataset: "/d/rig",
    episode: 12,
    target: "april_cube",
    sidecar: "/d/rig/derived/april_cube.csv",
    summary: summary(),
    min_coverage: 0.8,
    lever_arm_m: [0.02, -0.05, 0.15],
    mount_fit: {},
    camera_time_base: "argus_sidecar_sof_plus_exposure",
    ...over,
  };
}

describe("GT comparison verdicts", () => {
  it("keeps 'ran and does not certify' apart from 'could not run'", () => {
    expect(validateVerdict({ ok: false, returncode: 1 })?.outcome).toBe("not_certified");
    expect(validateVerdict({ ok: false, returncode: 2 })?.outcome).toBe("cannot_run");
  });

  it("flags coverage below the run's own threshold", () => {
    const low = validateReport({ summary: summary({ coverage: 0.5, n_paired: 500 }) });
    expect(coverageVerdict(low).dot).toBe("warning");
    expect(coverageVerdict(low).text).toContain("低于门槛");
    expect(coverageVerdict(validateReport()).dot).toBe("running");
  });
});

describe("the camera time base is a different measurement, not a worse one", () => {
  it("accepts the hardware SOF quietly", () => {
    expect(timeBaseVerdict(validateReport()).dot).toBe("running");
  });

  it("warns on the nominal grid and says the two are not comparable", () => {
    const v = timeBaseVerdict(validateReport({ camera_time_base: "dataset_nfps_grid_episode_local" }));
    expect(v.dot).toBe("warning");
    expect(v.detail).toContain("55 ms");
    expect(v.detail).toContain("不可比");
  });
});

describe("speed strata split geometry from timing", () => {
  it("orders the buckets from rest upward and skips the ones with no frames", () => {
    const rows = speedStrataRows(summary());
    expect(rows.map((r) => r.label)).toEqual([
      "静止 (<0.05 m/s)",
      "慢 (0.05–0.2)",
      "快 (>0.5)",
    ]);
  });

  it("reads a residual that grows with speed as a timing term, not as geometry", () => {
    const v = timingSignal(summary());
    expect(v.dot).toBe("warning");
    expect(v.detail).toContain("|v|·Δt");
    // And it says how to turn that into the exposure sign, which is the point.
    expect(v.detail).toContain("−0.5 / 0 / +0.5");
  });

  it("does not invent a timing term when the residual is flat in speed", () => {
    const flat = summary({
      strata: {
        speed_rest: { count: 300, p50_mm: 0.2, p95_mm: 0.4 },
        speed_fast: { count: 300, p50_mm: 0.2, p95_mm: 0.45 },
      },
    });
    expect(timingSignal(flat).dot).toBe("running");
  });

  it("refuses the split when the episode never stopped or never moved", () => {
    const onlyMoving = summary({ strata: { speed_fast: { count: 900, p50_mm: 0.5, p95_mm: 1.9 } } });
    const v = timingSignal(onlyMoving);
    expect(v.dot).toBe("idle");
    expect(v.detail).toContain("分不出");
  });
});

// --- one-click capture readiness --------------------------------------------

import type { TrackerMountCapture } from "../types";
import type { RecorderLiveState } from "./trackerMount";
import {
  DWELLS_PER_EPISODE_MIN,
  POSES_TO_CERTIFY,
  SEGMENTS_SUGGESTED,
  SEGMENT_MIN_S,
  captureLabel,
  captureReadiness,
  earlySaveWarning,
  pivotVerdict,
  protocolLabel,
  segmentLengthVerdict,
  suggestedDwellSeconds,
} from "./trackerMount";

function capture(over: Partial<TrackerMountCapture> = {}): TrackerMountCapture {
  return {
    dataset: "/d/tm",
    datasetName: "tracker_mount",
    episode: 0,
    episodeDir: "/d/tm/episodes/episode_000000",
    sessionId: "20260921_a",
    sessionPath: "/d/tm/laser_tracker/20260921_a",
    landed: true,
    poseLabel: "",
    purpose: "calibration_tracker_mount",
    beamValidFraction: 0.9,
    streamAdvanced: true,
    trackerError: "",
    modifiedUnixS: 1_700_000_000,
    ...over,
  };
}

const OFFLINE: RecorderLiveState = {
  connected: false,
  trackerEnabled: false,
  trackerReady: false,
  trackerDetail: "",
  episodeInFlight: false,
  sessionActive: false,
  sessionName: "",
};
// Connected, beam locked, and the gateway is holding a session for this page --
// the state every "can I record now" question is asked from.
const LOCKED: RecorderLiveState = {
  connected: true,
  trackerEnabled: true,
  trackerReady: true,
  trackerDetail: "locked on the SMR",
  episodeInFlight: false,
  sessionActive: true,
  sessionName: "tm_1",
};

describe("who owns the recorder", () => {
  it("will not record a dwell before a session has been claimed", () => {
    // The gateway refuses it anyway; greying out the button is how that stops
    // being a surprise the operator meets after摆好姿势.
    const r = captureReadiness([], { ...LOCKED, sessionActive: false, sessionName: "" });
    expect(r.blocker).toBe("no_session");
    expect(r.canRecord).toBe(false);
    expect(r.canStartSession).toBe(true);
  });

  it("names the session it is recording into", () => {
    // The name is the gateway's, not this page's: a reload used to mint a new
    // one and orphan every dwell already on disk under the old name.
    const r = captureReadiness([], LOCKED);
    expect(r.blocker).toBe("ready_to_record");
    expect(r.title).toContain("tm_1");
    expect(r.canRecord).toBe(true);
    expect(r.canStartSession).toBe(false);
  });

  it("does not let the claim be released mid-take", () => {
    // Releasing it would let Live Record queue a task episode into the middle
    // of this one -- the exact interleaving the claim exists to prevent.
    const r = captureReadiness([], { ...LOCKED, episodeInFlight: true });
    expect(r.canEndSession).toBe(false);
    expect(r.canSave).toBe(true);
  });

  it("still allows ending a session that recorded nothing", () => {
    // Otherwise an abandoned capture holds the recorder until the gateway is
    // restarted, and Live Record stays blocked with no way out on screen.
    const r = captureReadiness([], LOCKED);
    expect(r.canEndSession).toBe(true);
  });
});

describe("what is blocking the solve, said before the solver fails", () => {
  it("names Disconnect when nothing has landed yet", () => {
    // The session seals at Disconnect, not at the end of an episode. Without
    // this the operator meets it as a solver error two clicks later.
    const r = captureReadiness([capture({ landed: false })], LOCKED);
    expect(r.blocker).toBe("not_landed");
    expect(r.detail).toContain("Disconnect");
    expect(r.usable).toHaveLength(0);
  });

  it("calls out a landed session the tracker never contributed to", () => {
    const r = captureReadiness([capture({ streamAdvanced: false, beamValidFraction: 0 })], LOCKED);
    expect(r.blocker).toBe("tracker_silent");
    expect(r.dot).toBe("error");
  });

  it("passes when at least one landed capture has a live beam", () => {
    const r = captureReadiness(
      [capture({ streamAdvanced: false, beamValidFraction: 0 }), capture()],
      LOCKED,
    );
    expect(r.blocker).toBe("none");
    expect(r.usable).toHaveLength(2);
    // One pose per segment -- the old detail told operators three per take.
    expect(r.detail).toContain("一段一个姿态");
    expect(r.detail).toContain(String(POSES_TO_CERTIFY));
    expect(r.detail).toContain(String(SEGMENTS_SUGGESTED));
  });
});

describe("the panel and Live Record cannot disagree about the tracker", () => {
  it("does not tell a connected, beam-locked rig to Connect", () => {
    // The bug this pins: readiness read only the captures, so with nothing
    // recorded yet it said "先 Connect" while Live Record showed armed, 9/9
    // cameras and the beam locked on the SMR.
    const r = captureReadiness([], LOCKED);
    expect(r.blocker).toBe("ready_to_record");
    expect(r.dot).toBe("running");
    expect(r.detail).not.toContain("先 Connect");
    expect(r.canConnect).toBe(false);
    expect(r.canRecord).toBe(true);
  });

  it("asks for Connect only when the recorder really is down", () => {
    const r = captureReadiness([], OFFLINE);
    expect(r.blocker).toBe("not_connected");
    expect(r.canConnect).toBe(true);
    expect(r.canRecord).toBe(false);
    expect(r.canDisconnect).toBe(false);
  });

  it("separates 'connected without the tracker' from 'not connected'", () => {
    // Recording now would produce episodes with no session at all, and the way
    // out is a reconnect -- not something a 'please Connect' would have said.
    const r = captureReadiness([], { ...LOCKED, trackerEnabled: false, trackerReady: false });
    expect(r.blocker).toBe("tracker_off");
    expect(r.dot).toBe("error");
    expect(r.detail).toContain("重新 Connect");
  });

  it("shows the recorder's own sentence while the beam is still homing", () => {
    const r = captureReadiness([], {
      ...LOCKED,
      trackerReady: false,
      trackerDetail: "waiting for the SMR — put it in the home nest, homing retries automatically",
    });
    expect(r.blocker).toBe("beam_waiting");
    expect(r.detail).toContain("home nest");
  });

  it("names a beam break since Home instead of reading as locked", () => {
    const r = captureReadiness([], { ...LOCKED, trackerReady: false, trackerBeamBroken: true });
    expect(r.blocker).toBe("beam_waiting");
    expect(r.title).toContain("断过光");
  });

  it("lets recorded work decide the next action even though the rig is connected", () => {
    // After the last dwell the recorder is still up, but the next action is
    // Disconnect. Live state must not override work already done.
    const r = captureReadiness([capture({ landed: false })], LOCKED);
    expect(r.blocker).toBe("not_landed");
    expect(r.canDisconnect).toBe(true);
  });
});

describe("the recording protocol follows the solver's own floors", () => {
  it("sizes one segment for one pose, not for the whole session", () => {
    // The CLI merges segments by mount, so a segment needs one dwell. The old
    // ~100 s default made operators save early at every pose, and the
    // 2026-09-21 capture lost 4 of 15 poses to it.
    expect(DWELLS_PER_EPISODE_MIN).toBe(1);
    expect(suggestedDwellSeconds()).toBeGreaterThanOrEqual(SEGMENT_MIN_S);
    expect(suggestedDwellSeconds()).toBeLessThanOrEqual(6);
    expect(SEGMENT_MIN_S).toBeCloseTo(2.6, 5);
    expect(SEGMENTS_SUGGESTED).toBe(POSES_TO_CERTIFY + 5);
  });

  it("judges a segment length before anything is recorded", () => {
    expect(segmentLengthVerdict(2).level).toBe("bad");
    expect(segmentLengthVerdict(2).text).toContain("2.6");
    expect(segmentLengthVerdict(3).level).toBe("warn");
    expect(segmentLengthVerdict(4).level).toBe("ok");
    expect(segmentLengthVerdict(Number("abc")).level).toBe("bad");
  });

  it("calls out a take saved before it could hold a dwell", () => {
    expect(earlySaveWarning(1.5)).toContain("重录");
    expect(earlySaveWarning(3.0)).toBeNull();
    expect(earlySaveWarning(Number.NaN)).toBeNull();
  });

  it("labels which protocol a capture was recorded under", () => {
    expect(protocolLabel({ protocol: "tcp_pivot_dwell" })).toBe("pivot");
    expect(protocolLabel({ protocol: "smr_parked_pose_dwell" })).toBe("驻点");
    expect(protocolLabel({})).toBe("");
  });

  it("puts the tracker's own health in the row label", () => {
    expect(captureLabel(capture())).toContain("光束有效 90%");
    expect(captureLabel(capture({ landed: false }))).toContain("未落地");
    expect(captureLabel(capture({ beamValidFraction: -1 }))).toContain("光束未知");
  });
});

describe("Disconnect must not eat the take that is being recorded", () => {
  const RECORDING: RecorderLiveState = { ...LOCKED, episodeInFlight: true };

  it("closes off Disconnect while an episode is in flight", () => {
    // Observed 2026-09-21: the operator parked through several poses, pressed
    // Disconnect to land the session, and got nothing -- the gateway sends `q`
    // to a recording recorder, which discards. The tracker session sealed with
    // "episodes": [] and Saved stayed 0.
    const r = captureReadiness([], RECORDING);
    expect(r.blocker).toBe("recording_in_flight");
    expect(r.canDisconnect).toBe(false);
    expect(r.detail).toContain("丢掉");
  });

  it("offers Save instead, which is the only way to end a take early and keep it", () => {
    const r = captureReadiness([], RECORDING);
    expect(r.canSave).toBe(true);
    expect(r.canRecord).toBe(false);
  });

  it("re-opens Disconnect once the take is no longer in flight", () => {
    const r = captureReadiness([], LOCKED);
    expect(r.canDisconnect).toBe(true);
    expect(r.canSave).toBe(false);
  });

  it("keeps the guard even when earlier takes are already landed", () => {
    // The landed ones are still listed as usable; the in-flight one still must
    // not be thrown away to get at them.
    const r = captureReadiness([capture()], RECORDING);
    expect(r.blocker).toBe("recording_in_flight");
    expect(r.usable).toHaveLength(1);
    expect(r.canDisconnect).toBe(false);
  });
});


describe("E1p pivot results are read as position-only, reference first", () => {
  const base = {
    static_tcp_error_mm: { p95: 1.2 },
    tcp_budget_mm: 3,
    static_p95_within_budget: true,
    c_tcp_error_norm_mm: 0.8,
    certifies: true,
    certify_reasons: [] as string[],
  };

  it("reports a pass against the budget as static only", () => {
    const v = pivotVerdict(base);
    expect(v?.dot).toBe("running");
    expect(v?.detail).toContain("静态");
  });

  it("points at the calibration constant when the budget is exceeded", () => {
    const v = pivotVerdict({
      ...base, static_tcp_error_mm: { p95: 13.4 }, static_p95_within_budget: false, c_tcp_error_norm_mm: 13.0,
    });
    expect(v?.dot).toBe("error");
    expect(v?.detail).toContain("c_TCP");
  });

  it("refuses to let a weakly pinned reference be quoted, whatever the number", () => {
    const v = pivotVerdict({ ...base, certifies: false, certify_reasons: ["socket weakly pinned"] });
    expect(v?.dot).toBe("warning");
    expect(v?.detail).toContain("socket weakly pinned");
  });

  it("is silent without a report", () => {
    expect(pivotVerdict(null)).toBeNull();
  });
});

describe("pivot segments are continuous sweeps", () => {
  it("suggests a sweep length, not a pose length", () => {
    expect(suggestedDwellSeconds("pivot")).toBe(30);
    expect(segmentLengthVerdict(30, "pivot").level).toBe("ok");
    expect(segmentLengthVerdict(5, "pivot").level).toBe("warn");
    expect(segmentLengthVerdict(2, "pivot").level).toBe("bad");
  });

  it("never calls an early save lost, because every seated frame counts", () => {
    expect(earlySaveWarning(1.5, "pivot")).toBeNull();
    expect(earlySaveWarning(1.5)).not.toBeNull();
  });

  it("labels the continuous protocol and still reads the old one", () => {
    expect(protocolLabel({ protocol: "tcp_pivot_sweep" })).toBe("pivot 连续扫动");
    expect(protocolLabel({ protocol: "tcp_pivot_dwell" })).toBe("pivot");
  });
});
