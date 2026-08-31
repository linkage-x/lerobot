import { describe as suite, expect, it } from "vitest";

import type { RolloutOutcomeEntry } from "../types";
import { buildLandingPoints } from "./landingMapPoints";

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
