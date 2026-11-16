import { describe, expect, it } from "vitest";

import { toView } from "./useBoxPreview";

const payload = (sensor: Record<string, unknown>) => ({ active: true, deviceId: "box_six_d_force", staleS: 0.05, sensor });

describe("BOX preview: the external wrench", () => {
  it("reads the recorder's external wrench, never the SDK's no_gravity", () => {
    const view = toView(
      payload({
        fxyz_mxyz: [1, 2, 3, 0, 0, 0],
        fxyz_mxyz_no_gravity: [0, 0, 7.8, 0, 0, 0],
        fxyz_mxyz_external: [0.1, -0.2, 0.3, 0.01, 0, 0],
      }),
    );
    expect(view.forceExternal).toEqual({ fx: 0.1, fy: -0.2, fz: 0.3, mx: 0.01, my: 0, mz: 0 });
    expect(view.externalNote).toBeUndefined();
  });

  it("says why there is no external wrench instead of falling back to the SDK's", () => {
    const view = toView(
      payload({ fxyz_mxyz: [1, 2, 3, 0, 0, 0], fxyz_mxyz_no_gravity: [0, 0, 7.8, 0, 0, 0], external_note: "这次连接后还没做动态校准" }),
    );
    expect(view.forceExternal).toBeNull();
    expect(view.externalNote).toBe("这次连接后还没做动态校准");
  });
});
