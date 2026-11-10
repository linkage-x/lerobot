import { describe, expect, it } from "vitest";
import type { Fr3BaseEdgeResponse } from "../types";
import {
  applyConfirmation,
  edgeEnds,
  cartAttentionLines,
  cartBlockingLines,
  graphGuardBlocks,
  graphGuardLines,
  candidateSpread,
  laserSelectionWarnings,
  liveEdgeSummary,
  liveEdgeWarnings,
  moveSummary,
  methodLabel,
  qualitySummary,
  replayRelation,
  verdictLabel,
} from "./fr3BaseEdge";

type Candidate = Fr3BaseEdgeResponse["candidates"][number];
type Station = Fr3BaseEdgeResponse["stations"][number];
type LaserReplay = Fr3BaseEdgeResponse["laserReplays"][number];

const current: NonNullable<Fr3BaseEdgeResponse["current"]> = {
  world: "world_20260928_063531",
  createdUtc: "2026-10-09T06:55:38Z",
  method: "fr3_single_tag_hand_eye",
  label: "manual_run_20261002T014903Z",
  p0Run: "outputs/calibration/p0_single_tag_camera_calibration/manual_run_20261002T014903Z",
  intrinsicsRun: "calib_20260928_143107_intrinsics",
  fromReferenceWorld: true,
  referenceUnchanged: true,
  supersedes: 0,
  edgeCount: 1,
  quality: { method: "singleTag", captures: 137, outOfSampleP95Mm: 4.17, tagScale: 0.9912, folds: [] },
};

function candidate(run: string, meanMm: [number, number, number] | null, overrides: Partial<Candidate> = {}): Candidate {
  return {
    name: `${run}__x.json`,
    generatedUtc: "2026-10-13T08:00:00Z",
    label: run,
    p0Run: run,
    verdict: "ok",
    reasons: [],
    warnings: [],
    intrinsicsRun: "calib_20260928_143107_intrinsics",
    quality: { captures: 85, outOfSampleP95Mm: 5.55, tagScale: 0.9908, folds: [] },
    moveVsLive: meanMm
      ? { points: 85, meanMm, meanNormMm: Math.hypot(...meanMm), maxMm: Math.hypot(...meanMm) + 0.5, rotationDeg: 0.25 }
      : null,
    sameAsLive: false,
    applyBlockers: [],
    ...overrides,
  };
}

describe("fr3BaseEdge", () => {
  it("summarises the live edge with its run and its out-of-sample number", () => {
    expect(liveEdgeSummary(current)).toContain("manual_run_20261002T014903Z");
    expect(liveEdgeSummary(current)).toContain("p95 4.17 mm");
    expect(liveEdgeSummary(null)).toBe("没有注册边");
  });

  it("says what a missing or stale edge does to the FR3", () => {
    expect(liveEdgeWarnings(null, "w")[0]).toContain("都会被拒绝");
    expect(liveEdgeWarnings(current, "world_20260928_063531")).toEqual([]);
    expect(liveEdgeWarnings({ ...current, fromReferenceWorld: false }, "world_new")[0]).toContain("world_new");
    expect(liveEdgeWarnings({ ...current, referenceUnchanged: false }, "w")[0]).toContain("改过");
  });

  it("describes the arm-arm edge and confirms it in terms of its ends", () => {
    const q = { method: "armArm" as const, positions: 5, baselineM: 0.661, leaveOneOutRmsMm: 0.42, leaveOneOutMaxMm: 0.8, residualRmsMm: 0.3, folds: [] };
    expect(methodLabel(q)).toBe("两臂同球");
    expect(qualitySummary(q)).toContain("两基座相距 0.661 m");
    expect(edgeEnds()).toBe("world → fr3_base");
    expect(edgeEnds("fr3_base", "fr3_base_right")).toBe("fr3_base → fr3_base_right");
    const text = applyConfirmation({ ...candidate("armarm_x", null), from: "fr3_base", to: "fr3_base_right", quality: q }, current);
    expect(text).toContain("fr3_base → fr3_base_right");
    expect(text).toContain("闭合");
  });

  it("turns red for a moved cart, yellow for what only a maintainer can settle", () => {
    const base = {
      base: "fr3_base_right",
      why: "",
      episode: "ds/episodes/episode_000000",
      episodeStartUtc: "2026-11-10T10:00:00+00:00",
      checkedUtc: "",
      tagsMissing: [],
      currentEdge: true,
      blocking: false,
      note: "",
    };
    expect(cartBlockingLines(undefined)).toEqual([]);
    const check = (verdict: "ok" | "moved" | "inconsistent" | "partial" | "reference_stale", extra = {}) => ({
      enabled: true,
      references: [],
      bases: [{ ...base, verdict, ...extra }],
    });
    expect(cartBlockingLines(check("ok"))).toEqual([]);
    expect(cartAttentionLines(check("ok"))).toEqual([]);
    expect(cartAttentionLines(check("partial"))).toEqual([]);
    const moved = check("moved", { blocking: true, note: "fr3_base_right 的小车被推动过：…" });
    expect(cartBlockingLines(moved)).toEqual(["fr3_base_right 的小车被推动过：…"]);
    expect(cartAttentionLines(moved)).toEqual([]);
    expect(cartAttentionLines(check("inconsistent"))[0]).toContain("tag 彼此对不上");
    expect(cartAttentionLines(check("reference_stale", { currentEdge: false }))[0]).toContain("重新绑定");
  });

  it("turns red only while the world graph is not the trusted one", () => {
    expect(graphGuardBlocks(undefined)).toBe(false);
    expect(graphGuardLines({ state: "unrecorded", problems: [], recordedUtc: "" })).toEqual([]);
    expect(graphGuardLines({ state: "ok", problems: [], recordedUtc: "t" })).toEqual([]);
    const lines = graphGuardLines({ state: "diverged", problems: ["已停用的边 … 又成了现役"], recordedUtc: "2026-11-03T09:00:00Z" });
    expect(lines[0]).toContain("导出不换算到基座系");
    expect(lines[0]).toContain("2026-11-03T09:00:00Z");
    expect(lines[1]).toContain("又成了现役");
    expect(graphGuardBlocks({ state: "unreadable", problems: [], recordedUtc: "" })).toBe(true);
  });

  it("names a retired edge instead of saying there never was one", () => {
    const retired = [
      { label: "ball_check_20261015T030840Z", createdUtc: "2026-10-15T03:08:47Z", retiredUtc: "2026-11-03T08:00:00Z", reason: "base moved" },
    ];
    const warnings = liveEdgeWarnings(null, "w", retired);
    expect(warnings[0]).toContain("已停用");
    expect(warnings[0]).toContain("保留相机世界系");
    expect(warnings[1]).toContain("ball_check_20261015T030840Z");
    expect(warnings[1]).toContain("base moved");
  });

  it("reports the move in the workspace with its direction", () => {
    const move = candidate("run", [-2, -0.3, -3.8]).moveVsLive;
    expect(moveSummary(move)).toMatch(/^平均 4\.3 mm \(-2\.0, -0\.3, -3\.8\)/);
    expect(moveSummary(null)).toBe("—");
  });

  // The 2026-10-13 numbers at the replay trajectory: both 10-10 runs pass, sit
  // 3.9 mm apart, and the first is 4.3 mm from the live edge (the 10-02 run
  // re-solved to itself).
  it("shows how far passing candidates are from each other, the live edge included", () => {
    const spread = candidateSpread([
      candidate("manual_run_20261010T000513Z", [-2.0, -0.3, -3.8]),
      candidate("manual_run_20261010T070331Z", [-0.8, -3.5, -1.9]),
      candidate("manual_run_20261002T014903Z", [0, 0, 0], { sameAsLive: true }),
    ]);
    expect(spread?.mm).toBeCloseTo(Math.hypot(2.0, 0.3, 3.8), 6);
    expect([spread?.a, spread?.b].sort()).toEqual(["manual_run_20261002T014903Z", "manual_run_20261010T000513Z"]);
  });

  it("leaves rejected candidates out of the spread and needs two to compare", () => {
    expect(candidateSpread([candidate("a", [9, 9, 9], { verdict: "rejected" })])).toBeNull();
    expect(candidateSpread([candidate("a", [3, 4, 0])])).toEqual({ mm: 5, a: "a", b: "现役边" });
  });

  it("labels and confirms in terms of what changes", () => {
    expect(verdictLabel(candidate("a", [0, 0, 0], { sameAsLive: true }))).toBe("= 现役边");
    expect(verdictLabel(candidate("a", null, { verdict: "rejected" }))).toBe("未通过");
    const text = applyConfirmation(candidate("manual_run_20261010T070331Z", [-0.8, -3.5, -1.9]), current);
    expect(text).toContain("manual_run_20261010T070331Z");
    expect(text).toContain("已经导出的数据集");
    expect(text).toContain("提交");
  });

  it("summarises a laser edge by its station and its replays", () => {
    const text = qualitySummary({
      method: "laser",
      stationPoses: 15,
      stationRmsMm: 0.74,
      stationCertifies: true,
      replays: 2,
      laserVsFkP95Mm: 0.47,
      orientationSpanDeg: 11.6,
      replaySpreadMm: 0.21,
      folds: [],
    });
    expect(text).toBe("站位 15 点 rms 0.74 mm · 2 次回放 激光-FK p95 0.47 mm · 姿态跨度 11.6° · 回放之间相差 0.21 mm");
    expect(qualitySummary({ method: "laser", stationCertifies: false, folds: [] })).toContain("（未认证）");
  });

  it("summarises a ball edge by how well it predicts each ball from the others", () => {
    const q = {
      method: "ball" as const,
      positions: 5,
      residualRmsMm: 2.34,
      leaveOneOutRmsMm: 3.63,
      leaveOneOutMaxMm: 5.3,
      scalePct: -0.34,
      handheldS: "shared" as const,
      folds: [],
    };
    expect(methodLabel(q)).toBe("同球检验");
    expect(qualitySummary(q)).toBe(
      "5 个球位 · 留一预测 rms 3.6 mm / 最差 5.3 mm · 拟合残差 rms 2.3 mm · 相机 world 尺度 -0.34%（未吸收）"
    );
    expect(qualitySummary({ ...q, handheldS: "per-position" })).toContain("手持各自杠杆臂");
  });

  const station: Station = {
    name: "station_20261013_150434.json",
    world: "world_20260928_063531",
    matchesReference: true,
    poses: 15,
    rmsMm: 0.8,
    certifies: true,
    sessions: ["lt_20261013_061349"],
    sessionUtc: "2026-10-13T06:13:49Z",
  };
  const replay = (sessionUtc: string): LaserReplay => ({
    path: `ds/derived/c/real_replay/${sessionUtc}`,
    name: sessionUtc,
    dataset: "ds",
    sessionId: "lt",
    sessionUtc,
  });

  it("places each replay before or after the station", () => {
    expect(replayRelation(replay("2026-10-13T03:41:18Z"), station)).toBe("驻点前 2.5 小时");
    expect(replayRelation(replay("2026-10-13T06:40:00Z"), station)).toBe("驻点后 26 分钟");
    expect(replayRelation(replay("2026-10-12T08:00:00Z"), station)).toBe("另一天");
    expect(replayRelation(replay(""), station)).toBe("—");
  });

  // 2026-10-13: the station came three hours after the only replay with the laser.
  it("asks for replays on both sides of the station", () => {
    const before = replay("2026-10-13T03:41:18Z");
    const after = replay("2026-10-13T06:40:00Z");
    expect(laserSelectionWarnings(station, [before])[0]).toContain("只有一次回放");
    expect(laserSelectionWarnings(station, [before, replay("2026-10-13T03:10:00Z")])[0]).toContain("两侧");
    expect(laserSelectionWarnings(station, [before, after])).toEqual([]);
    expect(laserSelectionWarnings(station, [replay("2026-10-12T08:00:00Z")])[0]).toContain("不是同一天");
    expect(laserSelectionWarnings(station, [])).toEqual(["至少选一次带激光的 FR3 回放。"]);
    expect(laserSelectionWarnings({ ...station, certifies: false }, [before, after])[0]).toContain("认证");
    expect(laserSelectionWarnings({ ...station, rmsMm: 3.43 }, [before, after])[0]).toContain("3.43 mm > 1.5");
    expect(laserSelectionWarnings(null, [])).toEqual(["先选一个站位拟合。"]);
  });
});
