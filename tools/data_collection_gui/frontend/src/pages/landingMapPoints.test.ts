import { describe as suite, expect, it } from "vitest";

import type { RolloutOutcomeEntry } from "../types";
import { buildLandingPoints, pointFill, stageFill } from "./landingMapPoints";

function entry(
  recordedAt: string,
  outcome: RolloutOutcomeEntry["outcome"],
  xy: [number, number],
  rolloutIndex: number
): RolloutOutcomeEntry {
  return {
    recordedAt,
    checkpointId: "L4/030000",
    outcome,
    mode: "real",
    steps: 600,
    note: "",
    logPath: "",
    rolloutIndex,
    geometry: { graspXyz: [xy[0], xy[1], 0.06], closed: true }
  };
}

suite("landing map point order", () => {
  it("paints the newest grade last, whatever order the history arrives in", () => {
    // The history endpoint serves newest first. Drawing it in that order buries every new dot
    // under the older ones -- which is how a success graded on top of an earlier failure at the
    // same coordinates disappeared from the map entirely.
    const newestFirst = [
      entry("2026-08-31T04:06:01+00:00", "success", [0.3095, -0.0022], 6),
      entry("2026-08-31T04:02:20+00:00", "failure", [0.3095, -0.0025], 3)
    ];

    const points = buildLandingPoints(newestFirst, 0, undefined);

    expect(points.map((point) => point.outcome)).toEqual(["failure", "success"]);
  });

  it("keeps the ungraded point above everything, since it is the one asking a question", () => {
    const points = buildLandingPoints([entry("2026-08-31T04:06:01+00:00", "success", [0.4, -0.1], 6)], 7, {
      graspXyz: [0.4, -0.1, 0.06],
      closed: true
    });

    expect(points[points.length - 1].outcome).toBe("pending");
  });

  it("leaves out a grade the runtime reported no landing point for", () => {
    const ungeometried = { ...entry("2026-08-31T04:06:01+00:00", "failure", [0, 0], 6) };
    delete ungeometried.geometry;

    expect(buildLandingPoints([ungeometried], 0, undefined)).toEqual([]);
  });
});

suite("stage colours", () => {
  // The ramp exists because `stage` is ordinal. A palette would let two adjacent stages read
  // as unrelated categories, which is the failure this replaces: the 08-31 batch broke in three
  // different places and came back one colour.
  it("runs the full ramp whatever the task's chain length", () => {
    expect(stageFill(1, 7)).toBe(stageFill(1, 4));
    expect(stageFill(7, 7)).toBe(stageFill(4, 4));
    expect(stageFill(1, 7)).not.toBe(stageFill(7, 7));
  });

  it("keeps the ordering monotone, so 'further along' is never a darker step backwards", () => {
    const ramp = [1, 2, 3, 4, 5, 6, 7].map((stage) => stageFill(stage, 7));

    expect(new Set(ramp).size).toBe(7);
  });

  it("colours a rollout graded on a ladder by how far it got", () => {
    const point = { outcome: "failure" as const, stage: 2, terminalStage: 7 };

    expect(pointFill(point as never)).toBe(stageFill(2, 7));
  });

  it("leaves an aborted rollout out of the ramp whatever stage it reached", () => {
    // Stopped for reasons that say nothing about the policy, so it must not be counted by eye
    // among the rollouts the ramp is there to tally.
    const point = { outcome: "aborted" as const, stage: 5, terminalStage: 7 };

    expect(pointFill(point as never)).not.toBe(stageFill(5, 7));
  });

  it("falls back to the outcome colour for rollouts graded before ladders existed", () => {
    const older = { outcome: "success" as const };
    const graded = { outcome: "success" as const, stage: 7, terminalStage: 7 };

    expect(pointFill(older as never)).toBe(pointFill(graded as never));
  });

  it("puts the stage ahead of the geometry in the tooltip", () => {
    const graded: RolloutOutcomeEntry = {
      ...entry("2026-08-31T07:18:47+00:00", "failure", [0.37, -0.05], 4),
      stage: 2,
      stageId: "contact",
      terminalStage: 7,
      blocker: "object_pose_offset"
    };

    const [point] = buildLandingPoints([graded], 0, undefined);

    expect(point.title).toContain("stage 2/7 contact");
    expect(point.title).toContain("object_pose_offset");
  });
});

suite("who produced a landing point", () => {
  it("marks the point rather than dropping it when the operator was driving", () => {
    // Dropping them would leave the map showing only the attempts that went well: an
    // intervention is usually a rescue, so the rescued placements are exactly the interesting
    // ones. The dot stays; a ring says whose hand put it there.
    const assisted = entry("2026-08-31T04:06:01+00:00", "failure", [0.34, -0.13], 4);
    assisted.geometry = { ...assisted.geometry, graspBy: "expert" };

    const [point] = buildLandingPoints([assisted], 0, undefined);

    expect(point.drivenBy).toBe("expert");
    expect(point.title).toContain("operator was driving");
  });

  it("leaves a point from an older log unattributed rather than crediting the policy", () => {
    const [point] = buildLandingPoints(
      [entry("2026-08-31T04:06:01+00:00", "success", [0.34, -0.13], 5)],
      0,
      undefined
    );

    expect(point.drivenBy).toBeUndefined();
    expect(point.title).not.toContain("operator was driving");
  });
});
