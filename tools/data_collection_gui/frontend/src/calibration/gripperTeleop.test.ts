import { describe, expect, it } from "vitest";
import type { GripperTeleopResponse, GripperTeleopStatus } from "../types";
import { grippingNote, isTeleopRunning, teleopStatusText, teleopWarnings } from "./gripperTeleop";

const payload = (state?: "running" | "exited" | "error", status?: GripperTeleopStatus, over: Partial<GripperTeleopResponse> = {}): GripperTeleopResponse => ({
  ok: true,
  job: { state, status, log: [] },
  armDeviceId: 596523097,
  graspTriggerClosePct: 90,
  graspClosedWidthM: 0,
  hold: null,
  ...over,
});

describe("gripperTeleop", () => {
  it("says it waits for the handheld to match before following", () => {
    const t = teleopStatusText(payload("running", { state: "waiting", commandMm: 90, targetMm: 0 }));
    expect(t.text).toContain("等待对齐：手持指令 90.0 mm，FR3 保持 0.0 mm");
    expect(isTeleopRunning(payload("running").job)).toBe(true);
    expect(teleopStatusText(payload("running")).text).toContain("启动中");
  });

  it("follows, and notices a grip on something", () => {
    const t = teleopStatusText(
      payload("running", { state: "following", handheldFresh: true, handheldWidthMm: 35.9, triggerPct: 97, targetMm: 0, armWidthMm: 35.6 })
    );
    expect(t.text).toBe("跟随中：手持 35.9 mm · 扳机 97% → FR3 目标 0.0 mm（实测 35.6 mm）（夹着东西：目标 0.0 mm，停在 35.6 mm）");
    expect(grippingNote(40, 40.5)).toBe("");
    const stale = teleopStatusText(payload("running", { state: "following", handheldFresh: false }));
    expect(stale.warn).toBe(true);
  });

  it("confirms the exit kept the grip, or says it may not have", () => {
    const kept = teleopStatusText(payload("exited", { state: "exited", targetMm: 0, armWidthMm: 35.6, armMode: 1 }));
    expect(kept.warn).toBe(false);
    expect(kept.text).toContain("不卸力");
    const lost = teleopStatusText(payload("exited", { state: "exited", targetMm: 0, armMode: 0 }));
    expect(lost.warn).toBe(true);
    expect(lost.text).toContain("模式读回 0");
  });

  it("idles with what the last exit left, and names failures", () => {
    const idle = teleopStatusText(payload(undefined, undefined, { hold: { device_id: 1, target_m: 0.0, mode: 1, written: "2026-10-13T20:00:00" } }));
    expect(idle.text).toBe("未在遥控。上次退出时 FR3 夹爪保持 0.0 mm（2026-10-13T20:00:00）");
    expect(teleopStatusText(null).text).toBe("未在遥控");
    const failed = teleopStatusText({ ...payload("error"), job: { state: "error", returncode: 2, error: "没发现 FR3 夹爪 BOX", log: [] } });
    expect(failed.text).toBe("失败（exit 2）：没发现 FR3 夹爪 BOX");
  });

  it("warns about an unknown arm BOX and arm BOX restarts", () => {
    expect(teleopWarnings(payload(undefined, undefined, { armDeviceId: null }))[0]).toContain("corenetic_device_id");
    expect(teleopWarnings(payload("running", { state: "following", armReboots: 2 }))[0]).toContain("重启过 2 次");
    expect(teleopWarnings(payload())).toEqual([]);
  });
});
