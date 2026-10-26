import { describe, expect, it } from "vitest";
import { dualFr3Rows } from "./dualFr3Executability";
import type { DualFr3Executability } from "./types";

const stats = (p95: number, max: number) => ({ p50: 0, p95, max });

describe("dualFr3Rows", () => {
  it("lists absolute before relative and reads each mode's row", () => {
    const payload: DualFr3Executability = {
      datasetRoot: "/d",
      episode: 1,
      modes: {
        relative: { summary: {}, episode: null, createdUtc: "2026-10-26T03:00:00Z" },
        absolute: {
          summary: { episodes: 2, executable: 1 },
          createdUtc: "2026-10-26T04:00:00Z",
          episode: {
            dual_fr3_executable: false,
            reasons: ["arms 10 mm apart at frame 3"],
            placement_assumed: true,
            d_safe_mm: 75.6,
            min_arm_arm_gap_mm: 10.2,
            mesh_min_gap_mm: 14.0,
            retarget_change_position_mm: { right: stats(0.01, 0.13), left: stats(3.2, 21.5) },
            t_lr_change_position_mm: stats(3, 21.47),
            reference_duration_s: 31.3,
            slowdown_vs_nominal: 1.2
          }
        }
      }
    };
    const rows = dualFr3Rows(payload);
    expect(rows.map((row) => row.mode)).toEqual(["absolute", "relative"]);
    expect(rows[0]).toMatchObject({
      verdict: "not executable",
      gap: "10 / 76 (mesh 14)",
      placement: "assumed",
      tlr: "21.5",
      change: "R 0.01 / 0.13 · L 3.20 / 21.50",
      timing: "31.3 s (×1.20 retimed)",
      reasons: ["arms 10 mm apart at frame 3"]
    });
    expect(rows[1].verdict).toBe("not evaluated");
  });

  it("is empty without a payload", () => {
    expect(dualFr3Rows(null)).toEqual([]);
  });
});
