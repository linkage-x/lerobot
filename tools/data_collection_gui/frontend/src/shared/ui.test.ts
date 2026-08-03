import { describe, expect, it } from "vitest";

import { mujocoValidationMatchesSelection } from "./ui";

describe("mujocoValidationMatchesSelection", () => {
  it("holds a Thor validation to the cube it was run for", () => {
    expect(mujocoValidationMatchesSelection("left", "left", true)).toBe(true);
    expect(mujocoValidationMatchesSelection("left", "right", true)).toBe(false);
    expect(mujocoValidationMatchesSelection("both", "left", true)).toBe(false);
  });

  it("defaults a validation with no recorded cube to left, the gateway's default", () => {
    expect(mujocoValidationMatchesSelection(undefined, "left", true)).toBe(true);
    expect(mujocoValidationMatchesSelection(undefined, "right", true)).toBe(false);
  });

  it("ignores the cube mode on a rig that has no cubes", () => {
    // The workstation replays the arm's own EE stream and the gateway drops the cube mode for
    // it, so a mismatch here would lock Real Robot behind a picker that changed nothing --
    // and would relabel a validation that genuinely passed as merely "recommended".
    expect(mujocoValidationMatchesSelection("left", "right", false)).toBe(true);
    expect(mujocoValidationMatchesSelection(undefined, "both", false)).toBe(true);
  });
});
