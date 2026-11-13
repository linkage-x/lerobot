import { describe, expect, it } from "vitest";

import { gatewayErrorMessage } from "./httpError";

describe("gatewayErrorMessage", () => {
  it("shows the gateway's own error text", () => {
    expect(gatewayErrorMessage(500, { error: "FR3 夹爪遥控正在运行" })).toBe("FR3 夹爪遥控正在运行");
    expect(gatewayErrorMessage(409, { error: "sandbox has no hardware" })).toBe("sandbox has no hardware");
  });

  it("says the gateway did not answer when a 5xx has no gateway body (the dev-server proxy)", () => {
    for (const status of [500, 502, 503, 504]) {
      const message = gatewayErrorMessage(status, null);
      expect(message).toContain(`HTTP ${status}`);
      expect(message).toContain("gateway 没有应答");
    }
    expect(gatewayErrorMessage(500, { detail: "x" })).toContain("gateway 没有应答");
  });

  it("keeps other statuses as they are", () => {
    expect(gatewayErrorMessage(404, null)).toBe("HTTP 404");
  });
});
