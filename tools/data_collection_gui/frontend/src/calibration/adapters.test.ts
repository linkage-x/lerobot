import { describe, expect, it } from "vitest";

import { deviceBoxIdentity } from "./adapters";
import type { DeviceStatus } from "../types";

const box = (config: Record<string, unknown>) => ({ config }) as unknown as DeviceStatus;

describe("deviceBoxIdentity", () => {
  it("names the BOX by device_id, not the factory serial every BOX shares", () => {
    const device = box({ box_id: "", sn: "box_collection_controller_000001", device_id: 3969377039 });
    expect(deviceBoxIdentity(device)).toBe("box3969377039");
  });

  it("keeps a namespaced box_id and falls back to the serial without a device_id", () => {
    expect(deviceBoxIdentity(box({ box_id: "box42", device_id: 7 }))).toBe("box42");
    expect(deviceBoxIdentity(box({ sn: "personalised" }))).toBe("personalised");
    expect(deviceBoxIdentity(box({}))).toBe("");
  });
});
