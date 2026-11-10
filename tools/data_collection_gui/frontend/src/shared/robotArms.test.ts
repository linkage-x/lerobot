import { describe, expect, it } from "vitest";
import type { RobotArms } from "../types";
import { armChoices, armIp, armOfIp, armPrompt } from "./robotArms";

const arms: RobotArms = {
  ok: true,
  required: true,
  error: "",
  arms: [
    {
      arm: "right",
      inService: true,
      whyNot: "",
      robotIp: "192.168.11.102",
      gripperBox: 596523097,
      payload: null,
      startJointsRad: [0, -0.785, 0, -2.356, 0, 1.571, 0.785],
      replayConfig: "x.yaml"
    },
    {
      arm: "left",
      inService: false,
      whyNot: "B1: the second FR3 is not installed yet",
      robotIp: "",
      gripperBox: null,
      payload: null,
      startJointsRad: null,
      replayConfig: ""
    }
  ]
};

describe("robot arms (roadmap B2)", () => {
  it("gives an IP only for an arm in service", () => {
    expect(armIp(arms, "right")).toBe("192.168.11.102");
    expect(armIp(arms, "left")).toBe("");
    expect(armIp(arms, "")).toBe("");
    expect(armIp(null, "right")).toBe("");
    expect(armOfIp(arms, " 192.168.11.102 ")).toBe("right");
    expect(armOfIp(arms, "192.168.1.208")).toBe("");
  });

  it("lists an arm out of service, disabled, with its reason", () => {
    const [right, left] = armChoices(arms);
    expect(right).toMatchObject({ value: "right", enabled: true, title: "192.168.11.102 · BOX 596523097" });
    expect(left).toMatchObject({ value: "left", enabled: false });
    expect(left.title).toContain("B1: the second FR3");
  });

  it("offers every arm in service for the preflight, saying which", () => {
    expect(armChoices(arms, true).at(-1)).toMatchObject({ value: "both", label: "all in service (right)" });
    const two = { ...arms, arms: arms.arms.map((row) => ({ ...row, inService: true, robotIp: row.robotIp || "192.168.11.103" })) };
    expect(armChoices(two, true).at(-1)?.label).toBe("both");
  });

  it("has no default and says why when nothing can be chosen", () => {
    expect(armPrompt(arms)).toBe("Choose the arm: there is no default.");
    expect(armPrompt({ ...arms, arms: [arms.arms[1]] })).toContain("No FR3 is in service");
    expect(armPrompt({ ...arms, ok: false, error: "bad json" })).toContain("bad json");
  });
});
