import { describe, expect, it } from "vitest";

import pageSource from "./TrainingPage.tsx?raw";

/**
 * W&B logging defaults to on, which is only safe because the launch sends the *gated* flag.
 * The gateway refuses a run with `wandbEnabled` true and no API key stored for the machine
 * (`gateway.py`), and the checkbox is disabled in exactly that case -- so sending the raw
 * `wandbEnabled` would refuse every launch on a keyless machine, with no control on the page
 * able to turn it off. The failure is total (nothing can be trained) and its cause is two
 * files away from where it surfaces, so it is asserted here rather than left to be rediscovered.
 */
describe("W&B launch wiring", () => {
  it("defaults the checkbox to on", () => {
    expect(
      pageSource.includes("const [wandbEnabled, setWandbEnabled] = useState(true);"),
      "wandbEnabled no longer starts on; a long run that nobody can plot afterwards has to be " +
        "repeated to be compared"
    ).toBe(true);
  });

  it("sends the flag gated on a stored key, never the raw checkbox state", () => {
    expect(
      pageSource.includes("wandbEnabled: wandbWillLog"),
      "The launch payload sends the ungated wandbEnabled. On a machine with no W&B key stored " +
        "the gateway refuses the run and the disabled checkbox cannot turn it off, so no " +
        "training can start at all."
    ).toBe(true);
    expect(
      pageSource.includes("const wandbWillLog = wandbEnabled && Boolean(wandb?.configured);"),
      "wandbWillLog no longer gates on a stored key"
    ).toBe(true);
  });
});
