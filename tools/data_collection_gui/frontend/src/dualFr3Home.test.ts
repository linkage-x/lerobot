import { describe, expect, it } from "vitest";
import {
  dualHomeArmRows,
  dualHomeArmsBlocker,
  dualHomeControls,
  dualHomeGateLine,
  dualHomeOptionRows,
  dualHomeResultRows,
  dualHomeSecondsLeft,
  dualHomeStagesText,
  dualHomeStatus,
  dualHomeVerdict
} from "./dualFr3Home";
import type { DualHomeJob, DualHomePlan, RobotArm, RobotArms } from "./types";

const arm = (name: "right" | "left", inService = true): RobotArm => ({
  arm: name,
  inService,
  whyNot: inService ? "" : "away",
  robotIp: name === "right" ? "192.168.1.210" : "192.168.11.102",
  gripperBox: null,
  payload: null,
  startJointsRad: null,
  replayConfig: ""
});
const ARMS: RobotArms = { ok: true, required: true, arms: [arm("right"), arm("left")], error: "" };

const armPlan = (moving: boolean, deg = 0) => ({
  moving,
  q_now: [],
  q_goal: [],
  max_joint_move_deg: deg,
  joint: 1,
  min_env_gap_mm: 252.7,
  min_env_with: ["link7.0", "table"] as [string, string]
});

// the sequential case of the Python tests: together and left-first pass through each other
const SEQUENTIAL: DualHomePlan = {
  arms: { right: armPlan(true, 62.6), left: armPlan(true, 47.7) },
  start_gap_mm: 98.1,
  d_gate_mm: 60,
  gate_mm: 60,
  d_env_mm: 10,
  options: {
    together: { min_gap_mm: -76.3, pair: ["link5.1", "link6.0"], where: "right 40% / left 55% of the way" },
    right_then_left: { min_gap_mm: 98.1, pair: ["link7.0", "gripper.0"], where: "right 0% of the way" },
    left_then_right: { min_gap_mm: -76.1, pair: ["link5.1", "link6.0"], where: "left 60% of the way" }
  },
  stages: [["right"], ["left"]],
  chosen: "right_then_left",
  reasons: [],
  placement: { sigma_mm: 20, source: "tape 10-30", T_right_left: [] },
  speed_factor: 0.1,
  ip: { right: "192.168.1.210", left: "192.168.11.102" }
};

describe("dualHomeVerdict", () => {
  it("names the order, one arm alone, both at home and a refusal", () => {
    expect(dualHomeVerdict(SEQUENTIAL)?.headline).toBe("One after the other: the right arm first, then the left");
    expect(dualHomeVerdict({ ...SEQUENTIAL, stages: [["right", "left"]], chosen: "together" })?.headline).toBe(
      "Both arms move at once"
    );
    expect(dualHomeVerdict({ ...SEQUENTIAL, stages: [["left"]] })?.headline).toBe("Only the left arm moves; the other is home");
    expect(dualHomeVerdict({ ...SEQUENTIAL, stages: [] })?.tone).toBe("ok");
    expect(dualHomeVerdict({ ...SEQUENTIAL, stages: null, chosen: null })?.tone).toBe("no");
    expect(dualHomeVerdict(null)).toBeNull();
  });
});

describe("dualHomeOptionRows", () => {
  it("marks each order clear or too close against the gate the paths were held to", () => {
    const rows = dualHomeOptionRows(SEQUENTIAL);
    expect(rows.map((row) => [row.label, row.gap, row.clear, row.chosen])).toEqual([
      ["together", "-76 mm", false, false],
      ["right first, then left", "98 mm", true, true],
      ["left first, then right", "-76 mm", false, false]
    ]);
    expect(rows[0].where).toBe("link5.1 / link6.0 · right 40% / left 55% of the way");
  });

  it("lists only the orders the plan weighed (one arm moving: together only)", () => {
    const one = { ...SEQUENTIAL, options: { together: SEQUENTIAL.options.together } };
    expect(dualHomeOptionRows(one).map((row) => row.name)).toEqual(["together"]);
  });
});

describe("dualHomeArmRows and the gate", () => {
  it("shows each arm's way home and its closest approach to the table", () => {
    const rows = dualHomeArmRows({ ...SEQUENTIAL, arms: { right: armPlan(true, 34.4), left: armPlan(false) } });
    expect(rows[0]).toMatchObject({ arm: "right", ip: "192.168.1.210", move: "34° (largest: j1)", table: "253 mm (link7.0 / table)" });
    expect(rows[1].move).toBe("at start — stays");
  });

  it("says when the arms are already inside the gate and held to never closer than now", () => {
    expect(dualHomeGateLine(SEQUENTIAL)).toBe("Arm–arm gate 60 mm (3 × the placement's 20 mm σ); table 10 mm.");
    const inside = dualHomeGateLine({ ...SEQUENTIAL, start_gap_mm: 32.3, gate_mm: 27.3 });
    expect(inside).toContain("32 mm apart now, inside the gate");
    expect(inside).toContain("(27 mm)");
  });

  it("tells the confirmation what moves when", () => {
    expect(dualHomeStagesText(SEQUENTIAL)).toBe(
      "The right arm moves to its start joints at speed factor 0.1; once it is there, the left follows."
    );
    expect(dualHomeStagesText({ ...SEQUENTIAL, stages: [["left"]] })).toContain("the right stays");
    expect(dualHomeStagesText({ ...SEQUENTIAL, stages: [] })).toBe("");
  });
});

describe("dualHomeControls", () => {
  it("checks when idle and both arms serve, executes a ready plan that moves, cancels while live", () => {
    expect(dualHomeControls(undefined, "")).toEqual({ canCheck: true, canExecute: false, canCancel: false, cancelStops: false });
    expect(dualHomeControls(undefined, "Both arms must be in service").canCheck).toBe(false);
    const ready: DualHomeJob = { state: "ready", plan: SEQUENTIAL };
    expect(dualHomeControls(ready, "")).toEqual({ canCheck: false, canExecute: true, canCancel: true, cancelStops: false });
    expect(dualHomeControls({ state: "ready", plan: { ...SEQUENTIAL, stages: null } }, "").canExecute).toBe(false);
    expect(dualHomeControls({ state: "moving", plan: SEQUENTIAL }, "").cancelStops).toBe(true);
    expect(dualHomeControls({ state: "done", plan: SEQUENTIAL }, "").canCheck).toBe(true);
  });

  it("needs both arms in service", () => {
    expect(dualHomeArmsBlocker(ARMS)).toBe("");
    expect(dualHomeArmsBlocker({ ...ARMS, arms: [arm("right"), arm("left", false)] })).toContain("now: right");
    expect(dualHomeArmsBlocker(null)).toContain("Loading");
  });
});

describe("status, countdown and results", () => {
  it("counts down the runtime's wait for Execute", () => {
    const ready: DualHomeJob = { state: "ready", readyUnixS: 1000, waitS: 120 };
    expect(dualHomeSecondsLeft(ready, 1030)).toBe(90);
    expect(dualHomeSecondsLeft(ready, 2000)).toBe(0);
    expect(dualHomeSecondsLeft({ state: "moving" }, 1030)).toBeNull();
  });

  it("reads each state, and a cancel after the move started as a stop", () => {
    expect(dualHomeStatus({ state: "done", result: { ok: true, stages: [] } })).toBe("both arms at their start joints");
    expect(dualHomeStatus({ state: "done", result: null })).toBe("nothing to move");
    expect(dualHomeStatus({ state: "cancelled", result: null })).toBe("cancelled; nothing moved");
    expect(dualHomeStatus({ state: "refused", error: "left moved since the check" })).toBe("refused: left moved since the check");
    expect(dualHomeStatus(undefined)).toBe("idle");
  });

  it("lists each stage's arms with their error to the start joints", () => {
    const job: DualHomeJob = {
      state: "done",
      result: {
        ok: false,
        stages: [{ right: { ok: true, max_error_deg: 0.04 } }, { left: { ok: false, error: "reflex" } }]
      }
    };
    expect(dualHomeResultRows(job)).toEqual([
      { stage: 1, arm: "right", ok: true, text: "0.04° from its start joints" },
      { stage: 2, arm: "left", ok: false, text: "reflex" }
    ]);
  });
});
