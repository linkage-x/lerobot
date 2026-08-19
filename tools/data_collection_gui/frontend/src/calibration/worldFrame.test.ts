import { describe, expect, it } from "vitest";
import type { WorldRegistration } from "../types";
import {
  alignmentSummary,
  applyConfirmation,
  applyLabel,
  candidateClusters,
  canFallBackToGeometry,
  commonModeSummary,
  graphSummary,
  needsOperatorChoice,
  referenceSummary,
  stableSourceSummary,
  worldCameraRows,
} from "./worldFrame";

function registration(overrides: Partial<WorldRegistration> = {}): WorldRegistration {
  return {
    generated_utc: "2026-08-19T02:00:00Z",
    world_continuity_state: "CONTINUOUS",
    world_frame_id: "world_20260819_100000",
    parent_world_frame_id: null,
    reference_world_frame_id: "world_20260819_100000",
    calibration_id: "calib_20260819",
    reason: "stable_cluster",
    guidance: "6 台稳定相机把本次标定注册回原世界系。",
    auto_declarable: true,
    committed: false,
    min_stable_cameras: 4,
    consensus: {
      stable_cameras: ["cam_06", "cam_07", "cam_12", "cam_13"],
      moved_cameras: ["cam_09"],
      new_cameras: [],
      missing_cameras: [],
      ambiguous: false,
      alternative_clusters: [],
      thresholds: { translation_mm: 3, rotation_deg: 0.2 },
      pairs: [],
    },
    alignment: {
      cameras_used: ["cam_06", "cam_07", "cam_12", "cam_13"],
      translation_residual_mm: { cam_06: 0.31, cam_07: 0.12, cam_12: 0.44, cam_13: 0.2 },
      rotation_residual_deg: { cam_06: 0.004, cam_07: 0.002, cam_12: 0.006, cam_13: 0.003 },
      rms_translation_mm: 0.29,
      max_translation_mm: 0.44,
      rms_rotation_deg: 0.004,
      max_rotation_deg: 0.006,
      sigma_world_translation_mm: 0.15,
      sigma_world_rotation_deg: 0.002,
    },
    common_mode: { observable: false, note: "" },
    ...overrides,
  };
}

describe("canonical world frame presentation", () => {
  it("separates the cameras that define the frame from the one that moved", () => {
    const rows = worldCameraRows(registration());
    const byName = Object.fromEntries(rows.map((row) => [row.camera, row]));

    expect(byName.cam_06.role).toBe("stable");
    expect(byName.cam_06.detail).toContain("0.31 mm");
    // A moved camera still gets a pose; it just does not vote on the frame.
    expect(byName.cam_09.role).toBe("moved");
    expect(byName.cam_09.detail).toContain("仅更新自身位姿");
  });

  it("reports a missing camera as unsolved, not as stable", () => {
    const rows = worldCameraRows(
      registration({
        consensus: {
          ...registration().consensus,
          stable_cameras: ["cam_06", "cam_07"],
          moved_cameras: [],
          missing_cameras: ["cam_14"],
        },
      }),
    );

    expect(rows.find((row) => row.camera === "cam_14")?.role).toBe("missing");
  });

  it("does not word a continuity break as a failure of the data", () => {
    const broken = registration({
      world_continuity_state: "BROKEN",
      reason: "no_stable_cluster",
      alignment: null,
      auto_declarable: false,
    });

    // Committing a break mints a new world; calling it an "update" would imply
    // the old one was edited.
    expect(applyLabel(broken)).toContain("新建世界系");
    expect(applyConfirmation(broken)).toContain("历史数据仍属于旧世界系且保持可用");
    expect(alignmentSummary(broken)).toContain("没有可信的稳定相机组");
  });

  it("says how many cameras a commit would re-place, in the same world", () => {
    expect(applyConfirmation(registration())).toContain("1 台相机写入新位姿");
    expect(applyConfirmation(registration())).toContain("稳定相机的基准位姿保持冻结不变");
  });

  it("surfaces an ambiguous consensus as a choice, never as a majority vote", () => {
    const ambiguous = registration({
      world_continuity_state: "BROKEN",
      reason: "consensus_ambiguous",
      alignment: null,
      consensus: {
        ...registration().consensus,
        stable_cameras: ["cam_06", "cam_07", "cam_12"],
        moved_cameras: ["cam_13", "cam_14", "cam_09"],
        ambiguous: true,
        alternative_clusters: [["cam_09", "cam_13", "cam_14"]],
      },
    });

    expect(needsOperatorChoice(ambiguous)).toBe(true);
    expect(candidateClusters(ambiguous)).toHaveLength(2);
    expect(needsOperatorChoice(registration())).toBe(false);
  });

  it("states the common-mode blind spot instead of staying silent", () => {
    expect(commonModeSummary(registration())).toContain("未被检查");

    const watched = registration({
      common_mode: { observable: true, translation_mm: 120.4, rotation_deg: 1.2, drifted: true },
    });
    expect(commonModeSummary(watched)).toContain("整架相机相对环境移动");
    expect(commonModeSummary(watched)).toContain("120.40 mm");
  });

  it("explains what is missing before anything has been frozen", () => {
    expect(referenceSummary({ exists: false })).toContain("尚未冻结基准世界系");
    expect(
      referenceSummary({
        exists: true,
        world_frame_id: "world_20260819_100000",
        created_utc: "2026-08-19T10:00:00Z",
        cameras: ["cam_06", "cam_07"],
      }),
    ).toContain("2 台相机");
  });

  it("mentions the world graph only once there is more than one world", () => {
    const base = {
      ok: true,
      reference: { exists: true },
      registration: null,
      graph: { worlds: 1, edges: 0, nodes: [] },
    };
    expect(graphSummary(base)).toBe("");
    expect(graphSummary({ ...base, graph: { worlds: 2, edges: 0, nodes: [] } })).toContain("2 个世界系");
  });
});

describe("which evidence chose the stable cameras", () => {
  it("says so, and says how sensitive it is", () => {
    const summary = stableSourceSummary({
      origin: "rig_check",
      cameras: ["cam_06", "cam_07", "cam_12"],
      moved: ["cam_09"],
      generatedUtc: "2026-08-19T02:00:00Z",
    });

    expect(summary).toContain("相机自检");
    expect(summary).toContain("cam_09");
    // The reason the finer measurement is in charge has to be visible, or the
    // per-camera residual below reads as a restatement instead of a check.
    expect(summary).toContain("1.7 mm");
    expect(summary).toContain("独立复核");
  });

  it("explains the coarser floor when geometry is left to decide alone", () => {
    const summary = stableSourceSummary({ origin: "geometry", reason: "没有相机自检结果" });

    expect(summary).toContain("1 cm");
    expect(summary).toContain("没有相机自检结果");
  });

  it("offers the geometry-only fallback only when the self-check is driving", () => {
    expect(canFallBackToGeometry({ origin: "rig_check" })).toBe(true);
    expect(canFallBackToGeometry({ origin: "geometry" })).toBe(false);
    expect(canFallBackToGeometry({ origin: "operator" })).toBe(false);
    expect(canFallBackToGeometry(undefined)).toBe(false);
  });

  it("names the operator when they overrode both", () => {
    expect(stableSourceSummary({ origin: "operator", cameras: ["cam_06", "cam_07"] })).toContain(
      "操作者指定",
    );
  });
});
