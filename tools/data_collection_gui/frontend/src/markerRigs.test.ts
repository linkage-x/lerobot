import { describe, expect, it } from "vitest";
import { isMarkerRigTarget, markerRigPeriod } from "./markerRigs";
import type { MarkerRig } from "./types";

const rig = (target: string, from = "", until = ""): MarkerRig => ({
  target,
  hand: "left",
  deviceId: "box3969377039",
  markerIds: [102, 103, 104, 105],
  label: target,
  title: target,
  description: "",
  pivotNote: "",
  mountedFromUtc: from,
  mountedUntilUtc: until,
  replacedBy: "",
});

describe("markerRigPeriod", () => {
  // Local dates; the fixtures sit mid-day UTC so every lab timezone reads the same day.
  it("says from when a rig is on its BOX, and until when it was", () => {
    expect(markerRigPeriod(rig("v12", "2026-11-25T06:49:14Z"))).toBe("（11-25 起）");
    expect(markerRigPeriod(rig("v9b", "", "2026-11-25T06:49:14Z"))).toBe("（11-25 前）");
    expect(markerRigPeriod(rig("mid", "2026-11-06T06:00:00Z", "2026-11-25T06:49:14Z"))).toBe("（11-06–11-25）");
  });

  it("says nothing for a rig that was the only one on its BOX, or an unreadable time", () => {
    expect(markerRigPeriod(rig("right"))).toBe("");
    expect(markerRigPeriod(rig("bad", "not a time"))).toBe("");
  });
});

describe("isMarkerRigTarget", () => {
  it("is true only for a rig the gateway listed", () => {
    const rigs = [rig("gripper_rig_v12")];
    expect(isMarkerRigTarget(rigs, "gripper_rig_v12")).toBe(true);
    expect(isMarkerRigTarget(rigs, "april_cube")).toBe(false);
    expect(isMarkerRigTarget(undefined, "gripper_rig_v12")).toBe(false);
    expect(isMarkerRigTarget(rigs, "")).toBe(false);
  });
});
