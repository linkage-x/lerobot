import { describe, expect, it } from "vitest";
import {
  DEFAULT_FORM,
  chunkSeconds,
  cropProblem,
  formProblems,
  outputSize,
  parseEpisodes,
  policyDefaults,
  progressFraction,
  startRequest,
  stepHz,
  suggestHoldout
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

  it("sends the crop per camera, ACT by default", () => {
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

  it("asks pi0.5 for a task, sends its settings, the GPUs and the held-out episodes", () => {
    const pi05 = { ...DEFAULT_FORM, ...policyDefaults("pi05"), exports: ["a"], gpus: [0, 1, 2, 3] };
    expect(pi05.steps).toBe(6000);
    expect(formProblems(pi05, sizes)).toContain("pi0.5 needs the task described in words.");
    expect(formProblems({ ...pi05, task: "pick", holdout: "4,x" }, sizes)).toContain(
      "Held-out episodes: whole numbers, comma separated."
    );
    expect(formProblems({ ...pi05, task: "pick", gpus: [] }, sizes)).toContain("Pick at least one GPU.");
    const body = startRequest({ ...pi05, task: " pick up the part ", holdout: "24, 4,14" });
    expect(body).toMatchObject({ policy: "pi05", task: "pick up the part", gpus: [0, 1, 2, 3], holdoutEpisodes: [4, 14, 24] });
    expect(body).not.toHaveProperty("holdout");
    expect(startRequest({ ...DEFAULT_FORM, exports: ["a"], task: "ignored" }).task).toBe("");
  });

  it("holds out about a tenth of the episodes, evenly spaced", () => {
    expect(suggestHoldout(29)).toBe("4,14,24");
    expect(suggestHoldout(3)).toBe("");
    expect(parseEpisodes("")).toEqual([]);
    expect(parseEpisodes("3,1,3")).toEqual([1, 3]);
  });

  it("clamps progress", () => {
    expect(progressFraction(50, 100)).toBe(0.5);
    expect(progressFraction(150, 100)).toBe(1);
    expect(progressFraction(5, 0)).toBe(0);
  });
});
