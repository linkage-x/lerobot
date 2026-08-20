import { describe, expect, it } from "vitest";

import appSource from "../App.tsx?raw";

/**
 * The Training View page decides three things about a build -- the action contract, the camera
 * crop and the view rate -- and hands all three to `onExportApprovedDataset`. App used to
 * re-wrap that callback with only the first two parameters, so the crop and the rate were
 * dropped on the way to the gateway: nothing failed, the page kept showing the box that had
 * been drawn, and the view was built full-frame at the default rate anyway.
 *
 * A dropped argument is invisible at the type level (the extra parameters are optional), which
 * is exactly why it is worth a test.
 */
describe("Training View export wiring", () => {
  const start = appSource.indexOf("onExportApprovedDataset=");
  const handler = start < 0 ? "" : appSource.slice(start, appSource.indexOf("onOpenProcessing=", start));

  it("hands the page an export handler", () => {
    expect(start, "App no longer wires onExportApprovedDataset into DatasetExportPage").toBeGreaterThan(-1);
  });

  for (const argument of ["actionMode", "cameraCrops", "viewFps"]) {
    it(`forwards ${argument}`, () => {
      expect(
        handler.includes(argument),
        `App drops ${argument} between DatasetExportPage and api.exportApprovedDataset. The ` +
          "control that sets it goes on producing builds that ignore it, with no error anywhere."
      ).toBe(true);
    });
  }
});
