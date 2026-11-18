import { describe, expect, it } from "vitest";
import { lastCommandText, nextGripperAction } from "./armGripper";
import type { ArmGripperRow } from "../types";

const row = (widthMm: number | null, last: ArmGripperRow["last"] = null): ArmGripperRow => ({ arm: "right", deviceId: 1, widthMm, last });

describe("nextGripperAction", () => {
  it("closes an open jaw and opens a closed or holding one, by the live width", () => {
    expect(nextGripperAction(row(88.7), 88)).toBe("close");
    expect(nextGripperAction(row(47.6), 88)).toBe("open"); // holding a cube
    expect(nextGripperAction(row(0.4), 88)).toBe("open");
  });

  it("trusts the live width over the last command (teleop or a BOX reboot may have moved the jaw)", () => {
    expect(nextGripperAction(row(88.7, { deviceId: 1, action: "close", ok: true }), 88)).toBe("close");
  });

  it("without a width, undoes the last command that went through, else closes", () => {
    expect(nextGripperAction(row(null, { deviceId: 1, action: "close", ok: true }), 88)).toBe("open");
    expect(nextGripperAction(row(null, { deviceId: 1, action: "close", ok: false }), 88)).toBe("close");
    expect(nextGripperAction(row(null), 88)).toBe("close");
  });
});

describe("lastCommandText", () => {
  it("says what went, what is pending and why it failed", () => {
    expect(lastCommandText({ deviceId: 1, action: "close", targetM: 0, ok: true, modeBefore: 0 })).toBe("已闭合（目标 0.0 mm，先切到控制模式）");
    expect(lastCommandText({ deviceId: 1, action: "open", ok: null })).toBe("张开：等录制器回话…");
    expect(lastCommandText({ deviceId: 1, action: "open", ok: false, error: "boom" })).toBe("张开失败：boom");
  });
});
