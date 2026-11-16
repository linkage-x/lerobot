import { describe, expect, it } from "vitest";

import type { CollectionTask, ProcessingItem, ProcessingStatus } from "../types";
import { mujocoValidationMatchesSelection, taskExportPlan } from "./ui";

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

describe("taskExportPlan", () => {
  const task = { name: "Pick", datasetRepoId: "local/pick", completedEpisodes: 12 } as CollectionTask;
  const item = (name: string, status: ProcessingStatus, totalEpisodes: number, excluded: number[] = []) =>
    ({
      name,
      path: `/d/${name}`,
      status,
      totalEpisodes,
      qcExcludedEpisodes: excluded.map((episode) => ({ episode, checks: ["quat_norm"] })),
    }) as ProcessingItem;

  it("takes passed sessions whole, partial ones without their failed episodes, and nothing else", () => {
    const plan = taskExportPlan(task, [
      item("pick_20261116_100000", "qc_pass", 3),
      item("pick_20261116_110000", "qc_partial", 4, [1, 2]),
      item("pick_20261116_120000", "qc_failed", 2),
      item("pick_20261116_130000", "qc_stale", 2),
      item("pick_20261116_140000", "pose_ready", 1),
      item("other_20261116_100000", "qc_pass", 9),
    ]);

    expect(plan.included.map((row) => row.status)).toEqual(["qc_pass", "qc_partial"]);
    expect(plan.leftOut.map((row) => row.status)).toEqual(["qc_failed", "qc_stale", "pose_ready"]);
    expect(plan.includedEpisodes).toBe(5);
    expect(plan.leftOutEpisodes).toBe(2);
    expect(plan.missingEpisodeCount).toBe(0);
  });
});
