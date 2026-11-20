import { describe, expect, it } from "vitest";
import type { HardwareLease } from "../types";
import { leaseBanner } from "./hardwareLease";

const lease = (over: Partial<HardwareLease>): HardwareLease => ({
  role: "production",
  holder: "production",
  since: "2026-11-18T10:00:00Z",
  by: "hph",
  devUrl: "http://127.0.0.1:8766",
  devRepo: "/home/nvidia/sandboxes/dev",
  devCommit: "abc",
  mine: true,
  refusal: null,
  ...over
});

describe("leaseBanner", () => {
  it("says nothing on a sandbox, an older gateway, or production holding its own hardware", () => {
    expect(leaseBanner(null)).toBeNull();
    expect(leaseBanner(undefined)).toBeNull();
    expect(leaseBanner(lease({}))).toBeNull();
  });
  it("tells production who has the hardware and offers Reclaim", () => {
    const b = leaseBanner(lease({ holder: "dev", mine: false }))!;
    expect(b.action).toBe("reclaim");
    expect(b.text).toContain("http://127.0.0.1:8766");
    expect(b.confirm).toContain("正在动的机械臂");
  });
  it("offers the dev gateway Borrow, then Return", () => {
    expect(leaseBanner(lease({ role: "dev", mine: false }), "0123456789", "hph/x")!.action).toBe("borrow");
    const held = leaseBanner(lease({ role: "dev", holder: "dev", mine: true }), "0123456789", "hph/x")!;
    expect(held.action).toBe("return");
    expect(held.text).toContain("01234567 (hph/x)");
  });
  it("lets only production clear an unreadable lease", () => {
    expect(leaseBanner(lease({ holder: "unknown", mine: false }))!.action).toBe("reclaim");
    expect(leaseBanner(lease({ role: "dev", holder: "unknown", mine: false }))!.action).toBeNull();
  });
});
