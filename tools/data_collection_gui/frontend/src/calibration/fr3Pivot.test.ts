import { describe, expect, it } from "vitest";
import type { BallCheckReport, Fr3PivotResponse, PivotFitSummary } from "../types";
import {
  anchorText,
  ballCheckVerdict,
  ballEdgeSaveBlocker,
  ballEdgeSummary,
  biasConsistency,
  biasLabel,
  contactText,
  planSummaryText,
  episodeKey,
  fmtVec,
  handheldSharedSummary,
  isAnchored,
  jobStatusText,
  jointFitLines,
  pivotFitWarnings,
  positionDraftWarnings,
  rankedEdges,
  socketFrameVerdict,
} from "./fr3Pivot";

type Run = Fr3PivotResponse["runs"][number];
type Episode = Fr3PivotResponse["episodes"][number];

const fit: PivotFitSummary = {
  s_ee_mm: [0.2, -0.4, -10.1],
  s_sigma_mm: [0.03, 0.03, 0.05],
  c_base_mm: [612.3, -45.1, 48.0],
  c_sigma_mm: [0.02, 0.02, 0.03],
  residual_rms_mm: 0.12,
  residual_p95_mm: 0.25,
  n_used: 310,
  n_total: 4500,
  span_deg: 34,
  weak_singular: 0.28,
};

const run = (over: Partial<Run> = {}): Run => ({
  name: "20261014_101500_pivot.json",
  fitName: "20261014_101500_pivot.fit.json",
  note: "位置 1",
  created: "2026-10-14T10:15:00",
  fk: fit,
  deskOTEE: null,
  socketFrame: { ...fit, frame: "corenetic_socket_ee", s_ee_mm: [0.2, -0.4, 0.2] },
  ...over,
});

const episode = (over: Partial<Episode> = {}): Episode => ({
  dataset: "/data/hh",
  datasetName: "hh",
  episode: 0,
  world: "world_20260928_063531",
  recordedUnixS: 0,
  hasTrajectory: true,
  ...over,
});

const position = (usable = true) => ({
  fr3_pivot: "a.fit.json",
  dataset: "hh",
  episode: 0,
  usable,
  reasons: usable ? [] : ["没有 world 戳"],
  warnings: [],
});

const report = (over: Partial<BallCheckReport> = {}): BallCheckReport => ({
  name: "check_x.json",
  schema: "fr3_base_ball_check/v1",
  generated_utc: "",
  positions: [position()],
  edges: [
    { name: "live", rms_mm: 4.6, max_mm: 4.6, positions: [{ position: 0, error_mm: [0.8, -3.6, 2.9], norm_mm: 4.6 }] },
    { name: "laser__station_x.json", rms_mm: 1.2, max_mm: 1.2, positions: [{ position: 0, error_mm: [1, 0, 0.6], norm_mm: 1.2 }] },
    { name: "other.json", error: "候选边的世界系 world_b 不是这些录制的世界系" },
  ],
  ball_edges: {},
  ...over,
});

describe("fr3Pivot", () => {
  it("formats vectors and refuses partial ones", () => {
    expect(fmtVec([1, -2.345, 0])).toBe("(1.0, -2.3, 0.0)");
    expect(fmtVec([1, 2])).toBe("—");
    expect(fmtVec(null)).toBe("—");
  });

  it("says what the job is doing, and that a stop writes nothing", () => {
    expect(jobStatusText({ state: "collecting", elapsedS: 12.4, durationS: 45, log: [] })).toContain("12 / 45 s");
    expect(jobStatusText({ state: "stopped", log: [] })).toContain("没有写入");
    expect(jobStatusText({ state: "error", returncode: 1, log: ["ConnectionError: 192.168.11.102"] })).toBe(
      "失败（exit 1）：ConnectionError: 192.168.11.102"
    );
    expect(jobStatusText({ log: [] })).toBe("空闲");
  });

  it("warns when the sweep cannot separate the centre from the lever arm", () => {
    expect(pivotFitWarnings(fit)).toEqual([]);
    const w = pivotFitWarnings({ ...fit, span_deg: 9, residual_rms_mm: 0.9 });
    expect(w[0]).toContain("转得不够");
    expect(w[1]).toContain("插件中途离开过球");
  });

  it("reads the socket frame: within 1 mm it is the socket, beyond it the URDF offset absorbed something", () => {
    const good = socketFrameVerdict(run());
    expect(good.ok).toBe(true);
    expect(good.text).toContain("URDF 里的偏距是真实的 TCP 偏距");
    const bad = socketFrameVerdict(run({ socketFrame: { ...fit, frame: "corenetic_socket_ee", s_ee_mm: [0.1, 0.0, 4.0] } }));
    expect(bad.ok).toBe(false);
    expect(bad.text).toContain("差 4.0 mm");
    expect(bad.text).toContain("(0.2, -0.4, -10.1)");
    expect(socketFrameVerdict(run({ socketFrame: null })).text).toContain("没有 corenetic_socket_ee");
    expect(socketFrameVerdict(run({ fk: null })).ok).toBeNull();
  });

  it("ranks the scored edges and says one position cannot see rotation", () => {
    expect(rankedEdges(report()).map((e) => e.name)).toEqual(["laser__station_x.json", "live"]);
    const v = ballCheckVerdict(report());
    expect(v).toContain("只有 1 个球位置");
    expect(v).toContain("最贴合的是 laser__station_x（rms 1.2 mm，比次好的 现役边 小 3.4 mm）");
    expect(ballCheckVerdict(report({ positions: [position(), position()] }))).toMatch(/^2 个球位置/);
    expect(ballCheckVerdict(report({ positions: [position(false)] }))).toContain("没有可用的球位置");
  });

  it("marks a ball edge scored on balls as self-assessed", () => {
    const r = report({
      edges: [
        { name: "live", rms_mm: 11.9, max_mm: 14.4, positions: [{ position: 0, error_mm: [3, 10, 5], norm_mm: 11.9 }] },
        { name: "ball_check_x.json", from_balls: true, rms_mm: 2.3, max_mm: 3.0, positions: [{ position: 0, error_mm: [1, 2, 0], norm_mm: 2.3 }] },
      ],
    });
    expect(ballCheckVerdict(r)).toContain("最贴合的是 ball_check_x（球位拟合，自评）");
  });

  it("reads the ball edge by leave one out, and saves it only when it passes against the recorded reference", () => {
    const be = {
      positions: 5,
      position_indices: [0, 1, 2, 3, 4],
      extent_mm: [631, 240, 191],
      residual_rms_mm: 2.34,
      leave_one_out_mm: [1.08, 5.03, 2.59, 5.3, 2.15],
      leave_one_out_rms_mm: 3.63,
      leave_one_out_max_mm: 5.3,
      scale_pct: -0.34,
      verdict: "ok" as const,
      reasons: [],
    };
    expect(ballEdgeSummary(be)).toBe(
      "5 个位置（跨度 (631, 240, 191) mm）刚体拟合残差 rms 2.3 mm；每个位置由其余预测：1.1 / 5.0 / 2.6 / 5.3 / 2.1 mm（rms 3.6）" +
        "；相机 world 尺度 -0.34%（刚体边不吸收）。可以存为候选边"
    );
    const rejected = { ...be, verdict: "rejected" as const, reasons: ["只有 3 个球位：要 ≥ 4 个"] };
    expect(ballEdgeSummary(rejected)).toContain("不能作候选：只有 3 个球位");

    const w = "world_20260928_063531";
    const ok = report({ ball_edges: { [w]: be }, reference_world: w, world_reference_sha256: "abc" });
    expect(ballEdgeSaveBlocker(ok, w)).toBe("");
    expect(ballEdgeSaveBlocker({ ...ok, world_reference_sha256: undefined }, w)).toContain("旧版检验");
    expect(ballEdgeSaveBlocker({ ...ok, ball_edges: { [w]: rejected } }, w)).toBe("判定未通过");
    expect(ballEdgeSaveBlocker(ok, "world_b")).toBe("没有球位边");
  });

  it("states the shared handheld lever arm and how far the per-position ones wandered", () => {
    const r = report({
      handheld_shared: {
        w: { positions: [0, 1, 2], s_mm: [2.4, -2.5, 2.1], s_per_position_mm: [[0.3, -1.4, 0.4], [2.3, -4.3, -3.2], [3.9, -0.2, 6.2]] },
      },
    });
    const [line] = handheldSharedSummary(r);
    expect(line).toContain("3 个位置共用一个杠杆臂 (2.4, -2.5, 2.1) mm");
    expect(line).toContain("来回差 9.4 mm");
    expect(handheldSharedSummary(report({ handheld_shared: { w: { positions: [0, 1], error: "拟合失败" } } }))).toEqual(["w：拟合失败"]);
  });

  it("flags reused pivots or episodes, unfitted runs and unstamped episodes", () => {
    const runs = [run(), run({ name: "b_pivot.json", fk: null })];
    const eps = [episode(), episode({ episode: 1, world: "" })];
    const k0 = episodeKey(eps[0]);
    const k1 = episodeKey(eps[1]);
    expect(positionDraftWarnings([{ pivot: runs[0].name, episodeKey: k0 }], runs, eps)).toEqual([]);
    expect(positionDraftWarnings([{ pivot: "", episodeKey: "" }], runs, eps)[0]).toContain("每个位置");
    const w = positionDraftWarnings(
      [
        { pivot: runs[0].name, episodeKey: k0 },
        { pivot: runs[0].name, episodeKey: k0 },
        { pivot: "b_pivot.json", episodeKey: k1 },
      ],
      runs,
      eps
    );
    expect(w).toContain("同一次 FR3 pivot 用在了两个位置");
    expect(w).toContain("同一个手持 episode 用在了两个位置");
    expect(w).toContain("b_pivot.json 还没拟合");
    expect(w).toContain("hh ep1 没有 world 戳，检验会拒绝它");
  });

  it("says what an auto run is doing and that a stop leaves the arm where it is", () => {
    expect(jobStatusText({ state: "collecting", mode: "auto", elapsedS: 40, durationS: 136, biasMm: [3, 0, 0], log: [] })).toBe(
      "自动 pivot 执行中 40 / 136 s（柔顺阻抗，偏置 (3, 0, 0) mm）"
    );
    expect(jobStatusText({ state: "stopped", mode: "auto", log: [] })).toContain("停在当前位置");
    expect(jobStatusText({ state: "collecting", mode: "guided", elapsedS: 12, durationS: 60, biasMm: [-3, 0, 0], log: [] })).toBe(
      "引导 pivot 中 12 / 60 s：扶着夹爪倾斜、旋转，机器人把球窝压在球上（偏置 (-3, 0, 0) mm）"
    );
    expect(jobStatusText({ state: "collecting", mode: "guided", finishRequested: true, log: [] })).toContain("收尾");
  });

  it("summarises a plan, or lists why it cannot run", () => {
    const plan = {
      name: "p_plan.json",
      centre: "c_centre.json",
      created: "",
      passed: true,
      problems: [],
      durationS: 136,
      biasMm: [3, 0, 0],
      params: {},
      jointTravelDeg: [30, 21, 16, 49, 63, 78, 116],
      maxJointSpeedDegS: [3, 3, 2, 6, 7, 8, 7],
      minMarginWallDeg: [151, 70, 158, 28, 134, 32, 63],
      minMarginFr3Deg: [140, 69, 150, 27, 130, 80, 60],
      tableClearanceMm: 74.6,
      tableClearanceBody: "link_gripper_tip_right",
      tableZMm: 120,
      mujocoChecked: true,
    };
    expect(planSummaryText(plan)).toBe(
      "通过：136 s，最快关节 8°/s，离 panda_py 的墙最近 28°，j6 转 78°、j7 转 116°，离桌面（z=120 mm）最近 75 mm（link_gripper_tip_right）"
    );
    expect(planSummaryText({ ...plan, passed: false, problems: ["a", "b"] })).toBe("没通过：a；b");
    const dwells = {
      ...plan,
      kind: "dwells",
      dwells: 14,
      sigmaPerUnit: { s: [0.54, 0.43, 0.6], c: [0.5, 0.5, 0.5] },
      socketStandMm: 1.73,
      gripperStandMm: 3.6,
      gripperStandBody: "link_gripper_tip_right",
      anchorFrom: "x.fit.json",
      anchorMm: [0.1, 0.5, -1.5],
    };
    expect(planSummaryText(dwells)).toBe(
      "通过：14 个驻点，每个驻点误差 0.5 mm 时球窝 σ ≤ 0.30 mm，绕拟合的球心 (0.1, 0.5, -1.5) mm，136 s，最快关节 8°/s，" +
        "离 panda_py 的墙最近 28°，j6 转 78°、j7 转 116°，离桌面（z=120 mm）最近 75 mm（link_gripper_tip_right），" +
        "离球的支架：球窝 1.7 mm、link_gripper_tip_right 3.6 mm"
    );
    expect(biasLabel([-3, 0, 0])).toBe("偏置 −3 mm（x）");
    expect(biasLabel([1, 2, 0])).toBe("偏置 (1.0, 2.0, 0.0) mm");
  });

  it("reads contact and tells a seated +-3 mm pair from one that followed the command", () => {
    const commanded = (bias: number[], fraction = 1) => ({
      plan: "p",
      centre: "c1",
      bias_mm: bias,
      press_mm: 3,
      completed: true,
      contact: { press_median_n: 1.8, press_p05_n: 1.2, in_contact_fraction: fraction, max_force_n: 2.5 },
    });
    expect(contactText(commanded([0, 0, 0]))!.text).toBe("按压 1.8 N（5% 分位 1.2 N），接触 100%");
    expect(contactText(commanded([0, 0, 0], 0.8))!.ok).toBe(false);

    const at = (name: string, bias: number[], c: number[]) =>
      run({ name, commanded: commanded(bias), fk: { ...fit, c_base_mm: c } });
    const seated = biasConsistency([at("a", [3, 0, 0], [600, 0, 50]), at("b", [-3, 0, 0], [600.1, 0, 50]), run()]);
    expect(seated).toHaveLength(1);
    expect(seated[0].seated).toBe(true);
    expect(seated[0].text).toBe("球心差 0.10 mm、球窝差 0.00 mm（指令差 6.0 mm）：结果由球决定，测量成立");
    const followed = biasConsistency([at("a", [3, 0, 0], [603, 0, 50]), at("b", [-3, 0, 0], [597, 0, 50])]);
    expect(followed[0].seated).toBe(false);
    expect(biasConsistency([at("a", [3, 0, 0], [600, 0, 50])])).toEqual([]);
  });
});

describe("fr3Pivot anchor", () => {
  it("shows the fitted ball centre it would hold, or why it is off", () => {
    expect(anchorText(undefined).ok).toBe(false);
    const bad = { available: false, reason: "20261013_181013_pivot.json 的拟合不能用：转得不够（16.3°，弱方向 0.03）", sMm: [0, -4.1, 5.8] };
    expect(anchorText(bad)).toEqual({ ok: false, text: bad.reason, warning: "" });
    const good = { available: true, reason: "", sMm: [0.13, -0.53, 7.87], run: "20261013_181645_pivot.json", residualRmsMm: 2.76 };
    expect(anchorText(good)).toEqual({
      ok: true,
      text: "球心在 corenetic_socket_ee 里 (0.1, -0.5, 7.9) mm，来自 20261013_181645_pivot.json，残差 2.76 mm",
      warning: "",
    });
    // a re-clamp since the fit is said, but the fit stays usable (the insert re-seats within 0.2 mm)
    const reclamped = { ...good, warning: "20261013_181645_pivot.json 的拟合：夹爪遥控在 10-14 11:14:24 用过（这次拟合的运行之后），夹持可能变了" };
    expect(anchorText(reclamped)).toMatchObject({ ok: true, warning: reclamped.warning });
  });

  it("tells an anchored run from the frame's origin", () => {
    expect(isAnchored([0, 0, 0])).toBe(false);
    expect(isAnchored(null)).toBe(false);
    expect(isAnchored([0.1, -0.5, 7.9])).toBe(true);
    expect(jobStatusText({ state: "collecting", mode: "guided", durationS: 60, elapsedS: 3, biasMm: [0, 0, 0], anchorMm: [0.1, -0.5, 7.9], log: [] })).toContain(
      "弹簧挂在 (0.1, -0.5, 7.9) mm"
    );
  });
});

describe("fr3Pivot joint fit", () => {
  const base = {
    name: "20261014_120000_joint.json",
    created: "",
    runs: ["20261014_113021_pivot.json", "20261014_113121_pivot.json", "20261014_114709_pivot.json"],
    sharedCentre: true,
    fk: { s_ee_mm: [1.5, -0.5, -4.3] },
    warnings: [],
  };

  it("passes a fit that leaving any run out barely moves", () => {
    const lines = jointFitLines({
      ...base,
      socketFrame: {
        s_ee_mm: [1.5, -0.5, 6.0],
        residual_rms_mm: 0.98,
        span_deg: 70,
        weak_singular: 0.2,
        s_leave_one_run_out_mm: [[1.5, -0.4, 5.8], [1.6, -0.5, 6.2], [1.4, -0.6, 6.0]],
        run_centre_offset_mm: [[0.1, 0, 0], [0, 0.2, 0], [-0.1, -0.2, 0]],
      },
    });
    expect(lines[0].text).toBe("3 次采集，共用一个球心：残差 rms 0.98 mm，总转角 70°");
    expect(lines.every((l) => l.ok)).toBe(true);
  });

  it("says the tool axis is still loose and the ball moved (10-14)", () => {
    const lines = jointFitLines({
      ...base,
      socketFrame: {
        s_ee_mm: [1.6, -0.5, 4.9],
        residual_rms_mm: 1.12,
        span_deg: 36,
        weak_singular: 0.12,
        s_leave_one_run_out_mm: [[1.55, -0.15, 4.59], [1.5, -0.58, 5.03], [1.5, -0.5, 6.12]],
        run_centre_offset_mm: [[1.5, -2.1, 0.1], [0.2, 0, 0], [-1.2, 1.0, 0.4]],
      },
    });
    expect(lines.find((l) => l.text.startsWith("去掉任意一次"))).toMatchObject({ ok: false });
    expect(lines.find((l) => l.text.includes("球动过"))?.text).toContain("20261014_113021_pivot.json 2.6 mm");
    expect(jointFitLines({ ...base, socketFrame: null })[0].ok).toBe(false);
  });
});
