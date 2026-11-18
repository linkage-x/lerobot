import { describe, expect, it } from "vitest";
import type { RolloutCheckpoint, RolloutStatus } from "./rolloutApi";
import { executeBlockers } from "./rolloutExecute";

const status = (over: Partial<RolloutStatus> = {}): RolloutStatus => ({
  stages: [],
  jobs: [],
  profile: "thor",
  sandbox: null,
  recorderConnected: true,
  unlock: { right: { arm: "right", unlocked: true, by: "m" }, left: { arm: "left", unlocked: false, why: "no file" } },
  ...over
});
const ckpt = { path: "p", hasWorkspace: true } as RolloutCheckpoint;

describe("executeBlockers", () => {
  it("lets a person at the e-stop run an unlocked arm whose preview replayed identically", () => {
    expect(executeBlockers({ status: status(), checkpoint: ckpt, arm: "right", attended: true, previewReplayed: true })).toEqual([]);
  });
  it("names every missing condition", () => {
    const out = executeBlockers({
      status: status({ recorderConnected: false, sandbox: "hph" }),
      checkpoint: ckpt,
      arm: "left",
      attended: false,
      previewReplayed: false
    });
    expect(out.join("|")).toContain("sandbox");
    expect(out.join("|")).toContain("connect the recorder");
    expect(out.join("|")).toContain("left arm is locked: no file");
    expect(out.join("|")).toContain("Replay check");
    expect(out.join("|")).toContain("e-stop");
  });
});
