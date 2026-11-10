import { describe, expect, it } from "vitest";
import {
  deskTcpRequest,
  captureDeviceChoices,
  collectBlocker,
  faceLabel,
  faceProgress,
  fr3ArmsPayloadSnippet,
  fromLocalInput,
  isZeroJob,
  jobPercent,
  sixDDriftText,
  sixDFitNotes,
  sixDVerdict,
  toLocalInput,
  zeroBlocker,
  type DeskConfigEntry,
  type PayloadCalibrationResponse,
} from "./payloadCalibration";
import { CALIBRATION_TABS, isTabId } from "./calibrationTabs";

function payload(over: Partial<PayloadCalibrationResponse> = {}): PayloadCalibrationResponse {
  return {
    ok: true,
    error: "",
    job: {},
    arms: [{ arm: "right", inService: true, robotIp: "192.168.1.210", gripperBox: 3100715645, payload: null }],
    toolConfigs: [],
    poseSets: [],
    table: { edgeM: 0.2, topM: 0.13, marginMm: 50 },
    checks: {},
    checkValidS: 3600,
    runs: [],
    desks: [],
    captures: [],
    boxes: [],
    registryPath: "tools/thor/box_devices.json",
    runsRoot: "outputs/fr3_payload",
    captureRoot: "outputs/calibration/box_six_d",
    captureDurationRangeS: [30, 900],
    ...over,
  };
}

describe("collectBlocker", () => {
  const opts = { arm: "right", poses: "payload+six_d", toolConfig: "gripper", sixD: true };
  const check = { arm: "right", poses: "payload+six_d", pass: true, atUnixS: 1000, table: { edgeM: 0.2, topM: 0.13, marginMm: 50 }, summary: [] };

  it("asks for the arm, the configuration and a passing, fresh check, in that order", () => {
    expect(collectBlocker(payload(), { ...opts, arm: "" }, 1000)).toBe("先选机械臂");
    expect(collectBlocker(payload(), { ...opts, toolConfig: "" }, 1000)).toBe("先选末端配置");
    expect(collectBlocker(payload(), opts, 1000)).toContain("碰撞检查");
    expect(collectBlocker(payload({ checks: { "right|payload+six_d": { ...check, pass: false } } }), opts, 1000)).toContain("没有通过");
    expect(collectBlocker(payload({ checks: { "right|payload+six_d": check } }), opts, 1000 + 3601)).toContain("过期");
    expect(collectBlocker(payload({ checks: { "right|payload+six_d": check } }), opts, 1100)).toBe("");
  });

  it("refuses 6D logging on an arm without a gripper BOX, and anything while a job runs", () => {
    const bare = payload({
      checks: { "left|payload+six_d": { ...check, arm: "left" } },
      arms: [{ arm: "left", inService: true, robotIp: "192.168.11.102", gripperBox: null, payload: null }],
    });
    expect(collectBlocker(bare, { ...opts, arm: "left" }, 1100)).toContain("夹爪 BOX");
    expect(collectBlocker(bare, { ...opts, arm: "left", sixD: false }, 1100)).toBe("");
    expect(collectBlocker(payload({ job: { state: "running" } }), opts, 1100)).toBe("已有任务在进行");
  });
});

describe("job and face progress", () => {
  it("reads the measurement count", () => {
    expect(jobPercent({ progress: { done: 45, total: 90 } })).toBe(50);
    expect(jobPercent({ progress: { done: 0, total: 0 } })).toBeNull();
    expect(jobPercent({})).toBeNull();
  });

  it("lists all six faces, capped at the need", () => {
    const f = faceProgress({ "x+": 45, "z-": 15 }, 30);
    expect(f.map((x) => x.face)).toEqual(["x-", "x+", "y+", "y-", "z+", "z-"]);
    expect(f[1]).toEqual({ face: "x+", n: 45, frac: 1 });
    expect(f[5].frac).toBe(0.5);
    expect(f[0].n).toBe(0);
    expect(faceLabel("x-")).toBe("指尖朝上");
  });
});

describe("sixDVerdict", () => {
  it("separates refused, failing and passing fits", () => {
    expect(sixDVerdict(null).tone).toBe("warn");
    expect(sixDVerdict({ error: "attitudes {...}" }).tone).toBe("bad");
    const bad = sixDVerdict({ meetsTarget: false, heldoutWorstForceN: 0.4, heldoutWorstMomentNm: 0.02, missingAttitudes: ["z-"] });
    expect(bad.tone).toBe("bad");
    expect(bad.text).toContain("缺 侧放·两指并排 B");
    expect(sixDVerdict({ meetsTarget: true, heldoutWorstForceN: 0.22, heldoutWorstMomentNm: 0.02 }).tone).toBe("ok");
  });
});

describe("sixD drift notes", () => {
  const drift = { pairs: 199, worstForceN: 1.59, axis: "Fx", changeN: -1.59, fromS: 19.8, toS: 484.7, maxAbsByAxisN: [1.59, 0.528, 0.171] };
  it("says how far one attitude moved and over how long", () => {
    expect(sixDDriftText({ drift })).toBe("同一朝向漂移：Fx -1.59 N / 7.7 分钟；每轴最大 Fx/Fy/Fz 1.59 / 0.53 / 0.17 N");
    expect(sixDDriftText({ drift: { pairs: 0 } })).toContain("估计不了");
    expect(sixDDriftText({})).toBe(""); // a fit from before the check
  });
  it("shows the warnings, and the drift line only when no warning covers it", () => {
    const warned = sixDFitNotes({ drift, warnings: ["BOX 开始采集时才上电 1 分钟", "同一朝向 … 漂了 -1.59 N"] });
    expect(warned.map((n) => n.tone)).toEqual(["warn", "warn"]);
    const quiet = sixDFitNotes({ drift: { ...drift, worstForceN: 0.08, changeN: 0.08 }, warnings: [] });
    expect(quiet).toEqual([{ tone: "muted", text: expect.stringContaining("Fx +0.08 N") }]);
    expect(sixDFitNotes({ error: "attitudes" })).toEqual([]);
  });
});

it("round-trips a datetime-local value", () => {
  const t = 1793894400; // 2026-11-06 00:00 +08:00
  expect(fromLocalInput(toLocalInput(t))).toBe(t);
  expect(fromLocalInput("")).toBeNull();
});

it("puts hand-held BOXes first", () => {
  const c = captureDeviceChoices([
    { deviceId: "3100715645", role: "fr3_gripper", records: [] },
    { deviceId: "3906437858", role: "handheld", hand: "right", records: [] },
  ]);
  expect(c[0].id).toBe("3906437858");
  expect(c[0].label).toContain("手持 右");
  expect(c[1].label).toContain("FR3 夹爪");
});

it("writes the fr3_arms.json payload entry from a Desk fit", () => {
  const entry = { deskName: "Corenetic gripper", massKg: 0.7412, comFlangeM: [0.0128, -0.019, 0.0569], runs: ["a.json"] } as DeskConfigEntry;
  const parsed = JSON.parse(fr3ArmsPayloadSnippet(entry, "20261109_120000"));
  expect(parsed.mass_kg).toBe(0.7412);
  expect(parsed.com_m).toEqual([0.0128, -0.019, 0.0569]);
  expect(parsed.source).toContain("20261109_120000");
});

it("has unique tab ids, overview first", () => {
  const ids = CALIBRATION_TABS.map((t) => t.id);
  expect(new Set(ids).size).toBe(ids.length);
  expect(ids[0]).toBe("overview");
  expect(isTabId("fr3")).toBe(true);
  expect(isTabId("nope")).toBe(false);
});

describe("deskTcpRequest", () => {
  it("sends the chosen source and refuses an incomplete one", () => {
    expect(deskTcpRequest("estimate", { zM: "", pivot: "" })).toEqual({ source: "estimate" });
    expect(deskTcpRequest("template", { zM: "", pivot: "" })).toEqual({ source: "template" });
    expect(deskTcpRequest("manual", { zM: "0.19", pivot: "" })).toEqual({ source: "manual", zM: 0.19 });
    expect(deskTcpRequest("manual", { zM: " ", pivot: "" })).toBeNull();
    expect(deskTcpRequest("manual", { zM: "abc", pivot: "" })).toBeNull();
    expect(deskTcpRequest("pivot", { zM: "", pivot: "" })).toBeNull();
    expect(deskTcpRequest("pivot", { zM: "", pivot: "joint/x_joint.json" })).toEqual({ source: "pivot", pivot: "joint/x_joint.json" });
  });
});

describe("the arm-held 6D zero", () => {
  const check = { arm: "right", poses: "six_d_zero", pass: true, atUnixS: 1000, table: { edgeM: 0.2, topM: 0.13, marginMm: 50 }, summary: [] };

  it("needs its own route check, not the load's", () => {
    const loadOnly = payload({ checks: { "right|payload+six_d": { ...check, poses: "payload+six_d" } } });
    expect(zeroBlocker(loadOnly, "right", 1100)).toContain("零点路线");
    expect(zeroBlocker(payload({ checks: { "right|six_d_zero": { ...check, pass: false } } }), "right", 1100)).toContain("没有通过");
    expect(zeroBlocker(payload({ checks: { "right|six_d_zero": check } }), "right", 1000 + 3601)).toContain("过期");
    expect(zeroBlocker(payload({ checks: { "right|six_d_zero": check } }), "right", 1100)).toBe("");
    expect(zeroBlocker(payload(), "", 1100)).toBe("先选机械臂");
  });

  it("refuses an arm without a gripper BOX and anything while a job runs", () => {
    const bare = payload({
      checks: { "left|six_d_zero": { ...check, arm: "left" } },
      arms: [{ arm: "left", inService: true, robotIp: "192.168.11.102", gripperBox: null, payload: null }],
    });
    expect(zeroBlocker(bare, "left", 1100)).toContain("夹爪 BOX");
    expect(zeroBlocker(payload({ job: { state: "running" } }), "right", 1100)).toBe("已有任务在进行");
  });

  it("claims the zero job and its route check, and leaves the load panel's checks alone", () => {
    expect(isZeroJob({ kind: "six_d_zero" })).toBe(true);
    expect(isZeroJob({ kind: "check", poses: "six_d_zero" })).toBe(true);
    expect(isZeroJob({ kind: "check", poses: "payload+six_d" })).toBe(false);
    expect(isZeroJob({ kind: "collect" })).toBe(false);
  });
});
