import { describe, expect, it } from "vitest";
import type { RecordedDataset } from "../types";
import {
  cropSpecsForSelection,
  groupDatasetsByTask,
  selectionFpsProblem,
  summarizeSelection,
  taskBaseName,
  trainingViewName,
  viewFpsProblem
} from "./trainingViewSelection";

function dataset(overrides: Partial<RecordedDataset> & { name: string }): RecordedDataset {
  return {
    path: `/data/${overrides.name}`,
    totalEpisodes: 0,
    totalFrames: 0,
    fps: 60,
    updatedAt: "",
    ...overrides
  } as RecordedDataset;
}

const sideCam = { key: "observation.images.side", width: 640, height: 480 };

describe("taskBaseName", () => {
  it("strips the capture timestamp", () => {
    expect(taskBaseName("pick_and_place_20260819_171756")).toBe("pick_and_place");
  });

  it("strips a same-second disambiguating suffix", () => {
    expect(taskBaseName("pick_and_place_20260819_171756_02")).toBe("pick_and_place");
  });

  it("leaves a name that carries no timestamp alone", () => {
    expect(taskBaseName("fr3_pick_place_ee2ee_v1")).toBe("fr3_pick_place_ee2ee_v1");
  });
});

describe("trainingViewName", () => {
  it("keeps a single source's own name, timestamp included", () => {
    expect(trainingViewName(["pick_and_place_20260819_171756"], "delta_ee_from_prev_cmd")).toBe(
      "pick_and_place_20260819_171756__delta_ee_from_prev_cmd"
    );
  });

  it("collapses sessions of one task to the task name", () => {
    expect(
      trainingViewName(
        ["pick_and_place_20260819_171756", "pick_and_place_20260819_171323"],
        "delta_ee_from_prev_cmd"
      )
    ).toBe("pick_and_place__delta_ee_from_prev_cmd");
  });

  it("is order independent, so the same selection rebuilds into the same directory", () => {
    const a = trainingViewName(["stack_cube_20260819_100000", "pick_and_place_20260819_171756"], "absolute_ee");
    const b = trainingViewName(["pick_and_place_20260819_171756", "stack_cube_20260819_100000"], "absolute_ee");
    expect(a).toBe(b);
    expect(a).toBe("pick_and_place-stack_cube__absolute_ee");
  });

  it("names nothing when nothing is selected", () => {
    expect(trainingViewName([], "absolute_ee")).toBe("");
  });

  it("stays inside the character set a training job name accepts", () => {
    // The name becomes the job name and the `local/<name>` repo id, both of which reject
    // anything outside [A-Za-z0-9._-]. A view that builds but cannot be trained looks ready
    // on this page and fails on the Training page instead.
    const jobNameChars = /^[A-Za-z0-9._-]+$/;
    for (const names of [
      ["pick_and_place_20260819_171756", "fr3_spacemouse_20260813_160401"],
      ["pick and place_20260819_171756"],
      ["cube+stack_20260819_100000", "pick_and_place_20260819_171756"]
    ]) {
      expect(trainingViewName(names, "delta_ee_from_prev_cmd")).toMatch(jobNameChars);
    }
  });
});

describe("viewFpsProblem", () => {
  it("accepts an integer decimation", () => {
    expect(viewFpsProblem(60, 30)).toBe("");
  });

  it("refuses to invent frames", () => {
    expect(viewFpsProblem(30, 60)).toContain("invent frames");
  });

  it("refuses a non-divisor rate and names the divisors", () => {
    expect(viewFpsProblem(60, 25)).toContain("30");
  });

  it("has nothing to say about the source rate", () => {
    expect(viewFpsProblem(60, 0)).toBe("");
  });
});

describe("selectionFpsProblem", () => {
  it("blocks keeping the source rate when the sources disagree", () => {
    const problem = selectionFpsProblem(
      [dataset({ name: "a_20260101_000000", fps: 30 }), dataset({ name: "b_20260101_000000", fps: 60 })],
      0
    );
    expect(problem).toContain("30 and 60");
  });

  it("allows keeping the source rate when they agree", () => {
    expect(
      selectionFpsProblem(
        [dataset({ name: "a_20260101_000000", fps: 60 }), dataset({ name: "b_20260101_000000", fps: 60 })],
        0
      )
    ).toBe("");
  });

  it("lets a mixed selection through at a rate that divides both", () => {
    expect(
      selectionFpsProblem(
        [dataset({ name: "a_20260101_000000", fps: 30 }), dataset({ name: "b_20260101_000000", fps: 60 })],
        30
      )
    ).toBe("");
  });

  it("names the recording that blocks the rate", () => {
    const problem = selectionFpsProblem(
      [dataset({ name: "slow_20260101_000000", fps: 30 }), dataset({ name: "fast_20260101_000000", fps: 60 })],
      60
    );
    expect(problem).toContain("slow_20260101_000000");
  });
});

describe("summarizeSelection", () => {
  it("subtracts the episodes review excluded", () => {
    const summary = summarizeSelection(
      [dataset({ name: "a_20260101_000000", totalEpisodes: 10, totalFrames: 6000, excludedEpisodes: [2, 3] })],
      0
    );
    expect(summary).toMatchObject({ datasets: 1, episodes: 8, excluded: 2 });
  });

  it("decimates the frame estimate to the view rate", () => {
    const summary = summarizeSelection(
      [dataset({ name: "a_20260101_000000", totalEpisodes: 2, totalFrames: 2000, fps: 60 })],
      30
    );
    expect(summary.frames).toBe(1000);
  });

  it("adds up several sources at their own strides", () => {
    const summary = summarizeSelection(
      [
        dataset({ name: "a_20260101_000000", totalEpisodes: 2, totalFrames: 2000, fps: 60 }),
        dataset({ name: "b_20260101_000000", totalEpisodes: 1, totalFrames: 600, fps: 30 })
      ],
      30
    );
    expect(summary).toMatchObject({ datasets: 2, episodes: 3, frames: 1600 });
  });
});

describe("cropSpecsForSelection", () => {
  it("sends nothing while crop is off", () => {
    expect(cropSpecsForSelection([dataset({ name: "a_20260101_000000" })], false, {})).toEqual({});
  });

  it("omits a box that covers the whole frame", () => {
    const result = cropSpecsForSelection(
      [dataset({ name: "a_20260101_000000", cameraFeatures: [sideCam] })],
      true,
      { "observation.images.side": { x: 0, y: 0, w: 640, h: 480 } }
    );
    expect(result.crops).toBeUndefined();
    expect(result.label).toBe("full frame");
  });

  it("sends a real box once per camera key", () => {
    const result = cropSpecsForSelection(
      [
        dataset({ name: "a_20260101_000000", cameraFeatures: [sideCam] }),
        dataset({ name: "b_20260101_000000", cameraFeatures: [sideCam] })
      ],
      true,
      { "observation.images.side": { x: 224, y: 0, w: 416, h: 346 } }
    );
    expect(result.crops).toEqual({ "observation.images.side": [224, 0, 416, 346] });
    expect(result.label).toBe("1 camera crop");
  });

  it("names the source whose camera the box does not fit", () => {
    // The exporter's crop is one box per camera key applied to every source, so a smaller
    // camera of the same name has to be caught before the merge starts writing.
    const result = cropSpecsForSelection(
      [
        dataset({ name: "big_20260101_000000", cameraFeatures: [sideCam] }),
        dataset({
          name: "small_20260101_000000",
          cameraFeatures: [{ key: "observation.images.side", width: 320, height: 240 }]
        })
      ],
      true,
      { "observation.images.side": { x: 224, y: 0, w: 416, h: 346 } }
    );
    expect(result.error).toContain("small_20260101_000000");
  });

  it("rejects an odd box that H.264 cannot encode", () => {
    const result = cropSpecsForSelection(
      [dataset({ name: "a_20260101_000000", cameraFeatures: [sideCam] })],
      true,
      { "observation.images.side": { x: 3, y: 0, w: 416, h: 346 } }
    );
    expect(result.error).toContain("even");
  });
});

describe("groupDatasetsByTask", () => {
  it("puts every session of a task in one group, newest first", () => {
    const groups = groupDatasetsByTask([
      dataset({ name: "pick_and_place_20260819_171323" }),
      dataset({ name: "stack_cube_20260801_120000" }),
      dataset({ name: "pick_and_place_20260819_171756" })
    ]);
    expect(groups.map((group) => group.base)).toEqual(["stack_cube", "pick_and_place"]);
    expect(groups[1].datasets.map((item) => item.name)).toEqual([
      "pick_and_place_20260819_171756",
      "pick_and_place_20260819_171323"
    ]);
  });
});
