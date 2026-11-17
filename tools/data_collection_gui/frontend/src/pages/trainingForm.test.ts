import { describe, expect, it } from "vitest";
import {
  DEFAULT_FORM,
  chunkSeconds,
  cropProblem,
  formProblems,
  outputSize,
  progressFraction,
  startRequest,
  stepHz
} from "./trainingForm";

const sizes: Record<string, [number, number]> = { cam_10: [1920, 1080], cam_06: [1920, 1080] };

describe("training form", () => {
  it("states the action step in time", () => {
    expect(stepHz(60, 2)).toBe(30);
    expect(chunkSeconds(60, 2, 32)).toBeCloseTo(32 / 30);
  });

  it("sizes a cropped camera like the trainer does", () => {
    expect(outputSize([140, 220, 1180, 1000], 320)).toEqual([320, 240]);
    expect(outputSize([0, 0, 1920, 1080], 320)).toEqual([320, 180]);
  });

  it("refuses crops outside the image or too small", () => {
    expect(cropProblem([140, 220, 1180, 1000], [1920, 1080])).toBe("");
    expect(cropProblem([-1, 0, 100, 100], [1920, 1080])).toMatch(/inside/);
    expect(cropProblem([0, 0, 40, 400], [1920, 1080])).toMatch(/64/);
  });

  it("needs exports and cam_10, and known cameras", () => {
    expect(formProblems(DEFAULT_FORM, sizes)).toEqual(["Pick at least one export."]);
    const noWrist = { ...DEFAULT_FORM, exports: ["a"], cameras: [{ name: "cam_06", crop: null }] };
    expect(formProblems(noWrist, sizes)).toContain("cam_10 (on the gripper) is required.");
    const unknown = { ...DEFAULT_FORM, exports: ["a"], cameras: [...DEFAULT_FORM.cameras, { name: "cam_99", crop: null }] };
    expect(formProblems(unknown, sizes)).toContain("cam_99 is not in every chosen export.");
    expect(formProblems({ ...DEFAULT_FORM, exports: ["a"], nActionSteps: 64 }, sizes)[0]).toMatch(/exceed/);
  });

  it("sends the crop per camera and always ACT", () => {
    const body = startRequest({
      ...DEFAULT_FORM,
      exports: ["a"],
      jobName: "  run1 ",
      cameras: [
        { name: "cam_10", crop: null },
        { name: "cam_06", crop: [140, 220, 1180, 1000] }
      ]
    });
    expect(body.policy).toBe("act");
    expect(body.jobName).toBe("run1");
    expect(body.cameras).toEqual([
      { name: "cam_10", crop: null },
      { name: "cam_06", crop: [140, 220, 1180, 1000] }
    ]);
  });

  it("clamps progress", () => {
    expect(progressFraction(50, 100)).toBe(0.5);
    expect(progressFraction(150, 100)).toBe(1);
    expect(progressFraction(5, 0)).toBe(0);
  });
});
