import { describe, it, expect } from "vitest";
import {
  computeValidity,
  canTransition,
  pointerPromotionHint,
  promotionView,
  intrinsicsPromotionNote,
  pointerRows,
  summarizeKinds,
  worstValidity,
} from "./status";
import { MS } from "./config";

describe("computeValidity (dynamic: valid 30min, due-soon last 5min)", () => {
  const now = 1_000_000_000_000;
  it("is unknown when never run", () => {
    expect(computeValidity(null, "force_dynamic", now).state).toBe("unknown");
  });
  it("is valid with more than 5 min remaining", () => {
    const last = now - 10 * MS.minute; // 20 min left of 30
    expect(computeValidity(last, "force_dynamic", now).state).toBe("valid");
  });
  it("is due_soon within the last 5 min", () => {
    const last = now - 26 * MS.minute; // 4 min left
    expect(computeValidity(last, "force_dynamic", now).state).toBe("due_soon");
  });
  it("is overdue past 30 min", () => {
    const last = now - 31 * MS.minute;
    const v = computeValidity(last, "force_dynamic", now);
    expect(v.state).toBe("overdue");
    expect(v.remainingMs).toBeLessThan(0);
  });
  it("origin uses a daily window", () => {
    const last = now - 2 * MS.hour;
    expect(computeValidity(last, "force_origin", now).state).toBe("valid");
  });
});

describe("state machine transitions", () => {
  it("allows the happy path", () => {
    expect(canTransition("idle", "checking_prerequisites")).toBe(true);
    expect(canTransition("checking_prerequisites", "waiting_for_stability")).toBe(true);
    expect(canTransition("waiting_for_stability", "calibrating")).toBe(true);
    expect(canTransition("calibrating", "validating")).toBe(true);
    expect(canTransition("calibrating", "waiting_for_reboot")).toBe(true);
    expect(canTransition("waiting_for_reboot", "reconnecting")).toBe(true);
    expect(canTransition("reconnecting", "validating")).toBe(true);
    expect(canTransition("validating", "passed")).toBe(true);
    expect(canTransition("validating", "failed")).toBe(true);
  });
  it("rejects illegal jumps", () => {
    expect(canTransition("idle", "calibrating")).toBe(false);
    expect(canTransition("passed", "validating")).toBe(false);
    expect(canTransition("waiting_for_stability", "passed")).toBe(false);
  });
  it("allows cancel from any in-flight state", () => {
    expect(canTransition("calibrating", "cancelled")).toBe(true);
    expect(canTransition("reconnecting", "cancelled")).toBe(true);
  });
});

describe("summarizeKinds", () => {
  const now = 1_000_000_000_000;
  it("treats never-run kinds as unknown (needs attention)", () => {
    const s = summarizeKinds({ force_origin: null, force_dynamic: null, touch: null }, now);
    expect(s.unknown).toBe(3);
    expect(s.valid).toBe(0);
  });
  it("counts a mix of valid / due-soon / overdue", () => {
    const s = summarizeKinds(
      {
        force_origin: now - 1 * MS.hour, // valid (daily)
        force_dynamic: now - 26 * MS.minute, // due soon
        touch: now - 2 * MS.day, // overdue (daily)
      },
      now,
    );
    expect(s.valid).toBe(1);
    expect(s.dueSoon).toBe(1);
    expect(s.overdue).toBe(1);
  });
});

describe("worstValidity", () => {
  it("prefers the most severe state", () => {
    expect(worstValidity(["valid", "overdue", "due_soon"])).toBe("overdue");
    expect(worstValidity(["valid", "due_soon"])).toBe("due_soon");
    expect(worstValidity(["valid", "valid"])).toBe("valid");
    expect(worstValidity([])).toBe("unknown");
  });
});

describe("pointerRows — the solved run vs the one production loads", () => {
  const production = (over: Partial<{ intrinsicsRun: string; extrinsicsRun: string; error: string }> = {}) => ({
    configPath: "config_thor/april.yaml",
    intrinsicsRun: "intr_0804",
    extrinsicsRun: "extr_0804",
    error: "",
    ...over,
  });

  it("reports no difference when the pointers agree", () => {
    const rows = pointerRows({
      intrinsicsRun: "intr_0804",
      extrinsicsRun: "extr_0804",
      production: production(),
    });
    expect(rows.map((r) => r.differs)).toEqual([false, false]);
  });

  it("flags the field that drifted, and only that one", () => {
    // The failure that actually happened: a new extrinsics solve, the same
    // lenses, and a production pointer nobody updated.
    const rows = pointerRows({
      intrinsicsRun: "intr_0804",
      extrinsicsRun: "calib_20260820_extrinsics",
      production: production(),
    });
    expect(rows.find((r) => r.label === "外参")?.differs).toBe(true);
    expect(rows.find((r) => r.label === "内参")?.differs).toBe(false);
  });

  it("still shows both values when they differ", () => {
    const rows = pointerRows({
      intrinsicsRun: "intr_0804",
      extrinsicsRun: "calib_20260820_extrinsics",
      production: production(),
    });
    const extr = rows.find((r) => r.label === "外参");
    expect(extr?.solved).toBe("calib_20260820_extrinsics");
    expect(extr?.production).toBe("extr_0804");
  });

  it("does not claim a mismatch when the config could not be read", () => {
    // We do not know what production loads, so asserting a disagreement would
    // be a false alarm stacked on an already-broken deployment.
    const rows = pointerRows({
      intrinsicsRun: "intr_0804",
      extrinsicsRun: "calib_20260820_extrinsics",
      production: production({ error: "读不到生产配置" }),
    });
    expect(rows.map((r) => r.differs)).toEqual([false, false]);
  });

  it("does not flag a run the gateway has never learned", () => {
    const rows = pointerRows({ production: production() });
    expect(rows.map((r) => r.differs)).toEqual([false, false]);
  });

  it("renders empty strings rather than undefined when nothing is known", () => {
    const rows = pointerRows({});
    expect(rows.map((r) => [r.solved, r.production])).toEqual([
      ["", ""],
      ["", ""],
    ]);
  });
});

describe("pointerPromotionHint", () => {
  it("is empty when nothing drifted", () => {
    expect(pointerPromotionHint({})).toBe("");
    expect(pointerPromotionHint({ pointerMismatch: { fields: [], configPath: "x.yaml" } })).toBe("");
  });

  it("survives the empty object the gateway actually sends when they agree", () => {
    // Not hypothetical: the gateway serialised "no mismatch" as `{}`, which is
    // truthy, so `mismatch.fields.length` threw inside render and the whole
    // calibration page went blank -- on the healthy path. A client one deploy
    // behind the gateway still sees this shape.
    expect(pointerPromotionHint({ pointerMismatch: {} })).toBe("");
    expect(pointerPromotionHint({ pointerMismatch: {}, production: { configPath: "x.yaml", error: "" } })).toBe("");
  });

  it("falls back to the production config path when the mismatch omits one", () => {
    const hint = pointerPromotionHint({
      production: { configPath: "config_thor/april.yaml", error: "" },
      pointerMismatch: { fields: [{}] },
    });
    expect(hint).toContain("config_thor/april.yaml");
  });

  it("names the file to edit, because the solve will not edit it", () => {
    const hint = pointerPromotionHint({
      pointerMismatch: { fields: [{}], configPath: "config_thor/april.yaml" },
    });
    expect(hint).toContain("config_thor/april.yaml");
    expect(hint).toContain("fixed_camera_run_name");
  });

  it("surfaces a config read error ahead of any mismatch advice", () => {
    const hint = pointerPromotionHint({
      production: { configPath: "x.yaml", error: "生产配置解析失败" },
      pointerMismatch: { fields: [{}], configPath: "x.yaml" },
    });
    expect(hint).toBe("生产配置解析失败");
  });
});

describe("promotionView", () => {
  const review = {
    candidates: { extrinsics: "calib_20260902_103833_extrinsics" },
    configPath: "config_thor/april.yaml",
    extrinsics: {
      ok: true,
      live: "calib_20260820_173825_extrinsics",
      candidate: "calib_20260902_103833_extrinsics",
      pairCount: 21,
      medianBaselineShiftMm: 0.233,
      medianRotationDeg: 0.0911,
      worstPair: { a: "cam_07", b: "cam_14", liveMm: 817.05, candidateMm: 817.88, shiftMm: 0.832, rotationDeg: 0.1238 },
      cameras: [
        { camera: "cam_07", medianBaselineShiftMm: 0.637, maxBaselineShiftMm: 0.832, medianRotationDeg: 0.128, maxRotationDeg: 0.1846 },
        { camera: "cam_14", medianBaselineShiftMm: 0.151, maxBaselineShiftMm: 0.832, medianRotationDeg: 0.0719, maxRotationDeg: 0.1238 },
      ],
      candidateWorld: {
        worldFrameId: "world_20260819_031843",
        referenceWorldFrameId: "world_20260819_031843",
        continuityState: "CONTINUOUS",
        reason: "stable_cluster",
        stableCameras: ["cam_06", "cam_07"],
      },
      liveRmsePx: 0.2728,
      candidateRmsePx: 0.2263,
    },
    extrinsicsBlockers: [],
  };

  it("stays hidden when production already loads the newest run", () => {
    expect(promotionView(undefined).visible).toBe(false);
    expect(promotionView({ candidates: {}, configPath: "x.yaml" }).visible).toBe(false);
  });

  it("names the run that would take effect", () => {
    const view = promotionView(review);
    expect(view.visible).toBe(true);
    expect(view.headline).toContain("calib_20260902_103833_extrinsics");
    expect(view.kinds).toEqual(["extrinsics"]);
  });

  it("orders cameras as the gateway ranked them and keeps the median, not the max", () => {
    const rows = promotionView(review).rows;
    expect(rows.map((r) => r.camera)).toEqual(["cam_07", "cam_14"]);
    // 0.637 is cam_07's median; its max (0.832) belongs to the worst-pair line.
    expect(rows[0].baselineMm).toBe("0.64");
  });

  it("shows the reprojection numbers but refuses to let them read as a verdict", () => {
    const view = promotionView(review);
    expect(view.rmseNote).toContain("0.2728");
    expect(view.rmseNote).toContain("0.2263");
    // The candidate scores better here, which is exactly when the warning has
    // to be present: in August the better-scoring run was the wrong one.
    expect(view.rmseNote).toContain("不能用来择优");
    expect(view.headline).not.toMatch(/更好|推荐|建议采用/);
    expect(view.summary).not.toMatch(/更好|推荐/);
  });

  it("carries world continuity into the summary line", () => {
    expect(promotionView(review).world).toContain("CONTINUOUS");
    expect(promotionView(review).world).toContain("world_20260819_031843");
  });

  it("passes blockers through so the button can stay disabled", () => {
    const blocked = {
      ...review,
      extrinsicsBlockers: [{ kind: "world_frame_changed", message: "世界系变了" }],
    };
    expect(promotionView(blocked).blockers).toHaveLength(1);
    expect(promotionView(blocked).blockers[0].kind).toBe("world_frame_changed");
  });

  it("reports a run it could not read instead of rendering an empty table", () => {
    const broken = {
      candidates: { extrinsics: "calib_x_extrinsics" },
      configPath: "x.yaml",
      extrinsics: { ok: false, error: "读不到 summary.json", live: "a", candidate: "b" },
    };
    const view = promotionView(broken);
    expect(view.visible).toBe(true);
    expect(view.rows).toEqual([]);
    expect(view.summary).toContain("读不到 summary.json");
  });
});

describe("intrinsicsPromotionNote", () => {
  it("is empty when no lens run is up for promotion", () => {
    expect(intrinsicsPromotionNote(undefined)).toBe("");
    expect(intrinsicsPromotionNote({ candidates: {}, configPath: "x" })).toBe("");
  });

  it("names the model, because that is what makes a lens run unloadable", () => {
    const note = intrinsicsPromotionNote({
      candidates: { intrinsics: "run_intrinsics" },
      configPath: "x",
      intrinsics: {
        ok: true,
        live: "old",
        candidate: "run_intrinsics",
        cameras: ["cam_06", "cam_07"],
        model: "opencv_fisheye",
        trackerModel: "opencv_fisheye",
      },
    });
    expect(note).toContain("2 台");
    expect(note).toContain("opencv_fisheye");
  });
});
