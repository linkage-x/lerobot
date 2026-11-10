import { describe, expect, it } from "vitest";
import type { DeviceStatus } from "../types";
import { armZeroTimes, boxDeviceNumber, boxRoleRank, resolveBoxRole } from "./boxRoles";
import { computeValidity } from "./status";
import type { PayloadCalibrationResponse, ZeroReport } from "./payloadCalibration";

const ARM_BOX = "3100715645";
const HAND_RIGHT = "3906437858";
const HAND_LEFT = "3969377039";

function payload(over: Partial<PayloadCalibrationResponse> = {}): PayloadCalibrationResponse {
  return {
    ok: true,
    error: "",
    job: {},
    arms: [
      { arm: "right", inService: true, robotIp: "192.168.1.210", gripperBox: Number(ARM_BOX), payload: null },
      { arm: "left", inService: false, robotIp: "192.168.11.102", gripperBox: null, payload: null },
    ],
    toolConfigs: [],
    poseSets: [],
    table: { edgeM: 0.2, topM: 0.13, marginMm: 50 },
    checks: {},
    checkValidS: 3600,
    runs: [],
    desks: [],
    captures: [],
    boxes: [
      { deviceId: HAND_LEFT, role: "handheld", hand: "left", records: [] },
      { deviceId: HAND_RIGHT, role: "handheld", hand: "right", records: [] },
      { deviceId: ARM_BOX, role: "fr3_gripper", records: [] },
    ],
    registryPath: "tools/thor/box_devices.json",
    runsRoot: "outputs/fr3_payload",
    captureRoot: "outputs/calibration/box_six_d",
    captureDurationRangeS: [30, 900],
    ...over,
  };
}

function device(config: Record<string, unknown>): DeviceStatus {
  return { id: "box_six_d_force", kind: "box_collection", label: "", state: "running", fps: 0, latencyMs: 0, detail: "", config };
}

function report(created: string, over: Partial<ZeroReport> = {}): ZeroReport {
  return { name: created, created, arm: "right", deviceId: Number(ARM_BOX), mode: "dynamic", pass: true, state: "done", checks: [], warnings: [], ...over };
}

describe("boxDeviceNumber", () => {
  it("reads the box<id> namespace, else a row's device_id (single-BOX rigs keep box_id empty)", () => {
    expect(boxDeviceNumber(`box${ARM_BOX}`, [])).toBe(ARM_BOX);
    expect(boxDeviceNumber("", [device({}), device({ device_id: Number(HAND_RIGHT) })])).toBe(HAND_RIGHT);
    expect(boxDeviceNumber("", [device({})])).toBe("");
  });
});

describe("resolveBoxRole", () => {
  it("names handheld BOXes by hand and the arm BOX by the arm that carries it", () => {
    const p = payload();
    expect(resolveBoxRole(HAND_RIGHT, p)).toMatchObject({ kind: "handheld", hand: "right", title: `手持右 BOX ${HAND_RIGHT}` });
    const arm = resolveBoxRole(ARM_BOX, p);
    expect(arm).toMatchObject({ kind: "fr3_gripper", arm: "right", title: `右臂夹爪 BOX ${ARM_BOX}` });
    expect(arm.note).toBeUndefined();
  });

  it("trusts fr3_arms.json for an arm BOX box_devices.json has not registered yet", () => {
    const p = payload({ boxes: [] });
    const r = resolveBoxRole(ARM_BOX, p);
    expect(r).toMatchObject({ kind: "fr3_gripper", arm: "right" });
    expect(r.note).toContain("box_devices.json");
  });

  it("flags an fr3_gripper BOX no arm names, so the zero panel lets the operator pick the arm", () => {
    const p = payload({ arms: [] });
    const r = resolveBoxRole(ARM_BOX, p);
    expect(r).toMatchObject({ kind: "fr3_gripper", arm: "", title: `FR3 夹爪 BOX ${ARM_BOX}` });
    expect(r.note).toContain("fr3_arms.json");
  });

  it("shows an unregistered BOX as before, with a warning; no payload yet means no warning", () => {
    const r = resolveBoxRole("42", payload());
    expect(r.kind).toBe("unregistered");
    expect(r.note).toContain("没有在 tools/thor/box_devices.json 里登记");
    expect(resolveBoxRole("42", null)).toEqual({ kind: "unregistered", deviceId: "42", title: "BOX 42" });
  });

  it("orders handheld, then unregistered, then arm BOXes", () => {
    const p = payload();
    const roles = [ARM_BOX, "42", HAND_LEFT].map((id) => resolveBoxRole(id, p));
    expect(roles.sort((a, b) => boxRoleRank(a) - boxRoleRank(b)).map((r) => r.deviceId)).toEqual([HAND_LEFT, "42", ARM_BOX]);
  });
});

describe("armZeroTimes", () => {
  const at = (s: string) => new Date(s).getTime();

  it("dates the dynamic zero from any passing run and the origin only from origin runs", () => {
    const t = armZeroTimes(
      [
        report("2026-11-10T08:00:00", { mode: "origin" }),
        report("2026-11-10T09:00:00"),
        report("2026-11-10T09:30:00", { pass: false }), // failed: does not count
        report("2026-11-10T10:00:00", { arm: "left" }), // the other arm
        report("2026-11-10T10:30:00", { deviceId: Number(HAND_RIGHT) }), // another BOX
      ],
      "right",
      ARM_BOX,
    );
    expect(t.force_origin).toBe(at("2026-11-10T08:00:00"));
    expect(t.force_dynamic).toBe(at("2026-11-10T09:00:00"));
  });

  it("an origin run ends with a dynamic zero, so it refreshes both", () => {
    const t = armZeroTimes([report("2026-11-10T09:00:00"), report("2026-11-10T11:00:00", { mode: "origin" })], "right", ARM_BOX);
    expect(t.force_dynamic).toBe(at("2026-11-10T11:00:00"));
  });

  it("says never when there is nothing, and the validity follows (30 min dynamic)", () => {
    expect(armZeroTimes(undefined, "right", ARM_BOX)).toEqual({ force_origin: null, force_dynamic: null });
    const t = armZeroTimes([report("2026-11-10T09:00:00")], "right", ARM_BOX);
    expect(computeValidity(t.force_dynamic, "force_dynamic", at("2026-11-10T09:10:00")).state).toBe("valid");
    expect(computeValidity(t.force_dynamic, "force_dynamic", at("2026-11-10T09:40:00")).state).toBe("overdue");
    expect(computeValidity(t.force_origin, "force_origin").state).toBe("unknown");
  });
});
