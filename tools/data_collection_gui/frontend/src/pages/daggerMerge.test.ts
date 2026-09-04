import { describe, expect, it } from "vitest";
import type { ProcessingItem, RecordedDataset, TrainingView } from "../types";
import {
  daggerMergeCandidates,
  isDaggerDataset,
  mergeBlockedReason,
  mergeFormFingerprint,
  mergedViewName,
  parseEpisodeSelection,
  safeTrainingViewName,
  visibleDaggerCandidates
} from "./daggerMerge";

function dataset(overrides: Partial<RecordedDataset> & { name: string }): RecordedDataset {
  return {
    path: `/data/${overrides.name}`,
    totalEpisodes: 4,
    totalFrames: 400,
    fps: 30,
    updatedAt: "",
    ...overrides
  } as RecordedDataset;
}

function processing(overrides: Partial<ProcessingItem> & { path: string }): ProcessingItem {
  return {
    name: overrides.path.split("/").pop() ?? "",
    status: "qc_pass",
    trajectoryVersion: null,
    qcSummary: "",
    message: "",
    updatedAt: "",
    totalEpisodes: 4,
    totalFrames: 400,
    validFramesPct: null,
    logTail: [],
    ...overrides
  } as ProcessingItem;
}

function trainingView(overrides: Partial<TrainingView> & { name: string }): TrainingView {
  const { name, ...rest } = overrides;
  return {
    name,
    root: `/views/${name}`,
    repoId: `local/${name}`,
    episodes: 48,
    frames: 23971,
    fps: 30,
    actionMode: "delta_ee_from_prev_cmd",
    cameras: [],
    sourceFps: {},
    frameStride: {},
    cameraCrops: {},
    sourceRoots: [],
    excludedEpisodes: {},
    buildId: "",
    sourceDigest: "",
    modifiedAt: "",
    ...rest
  };
}

describe("mergedViewName", () => {
  it("names a single-source merge after the checkpoint step the corrections came from", () => {
    expect(
      mergedViewName(
        "/repo/outputs/exports/training_views/insert__delta_ee_from_prev_cmd__L4_full48",
        ["/repo/outputs/datasets/dagger_L4_full48_holdout22_40_030000"],
        ""
      )
    ).toBe("insert__delta_ee_from_prev_cmd__L4_full48__plus_dagger_030000");
  });

  it("counts the sources when several DAgger datasets are merged at once", () => {
    expect(mergedViewName("base_view", ["/data/dagger_a", "/data/dagger_b"], "")).toBe(
      "base_view__plus2dagger"
    );
  });

  it("falls back to an unnumbered tag when the dataset name carries no step", () => {
    expect(mergedViewName("base_view", ["/data/dagger_corrections"], "")).toBe("base_view__plus_dagger");
  });

  it("prefers the name the operator typed, folded into the safe character set", () => {
    expect(mergedViewName("base_view", ["/data/dagger_a"], " my view/v2 ")).toBe("my-view-v2");
  });
});

describe("safeTrainingViewName", () => {
  it("matches the gateway's folding of unsafe characters and edge punctuation", () => {
    expect(safeTrainingViewName("insert task/v2")).toBe("insert-task-v2");
    expect(safeTrainingViewName("...")).toBe("training_view");
    expect(safeTrainingViewName("-leading.trailing-")).toBe("leading.trailing");
  });
});

describe("parseEpisodeSelection", () => {
  it("reads an empty field as the whole base view", () => {
    expect(parseEpisodeSelection("  ", 48)).toEqual({ episodes: [], error: "" });
  });

  it("expands ranges and de-duplicates, so a holdout split can be typed in one line", () => {
    expect(parseEpisodeSelection("0-3, 2 6", 48)).toEqual({ episodes: [0, 1, 2, 3, 6], error: "" });
  });

  it("refuses an episode the base view does not have", () => {
    const result = parseEpisodeSelection("0-49", 48);
    expect(result.episodes).toEqual([]);
    expect(result.error).toContain("48 episode(s)");
  });

  it("refuses tokens that are not episode numbers", () => {
    expect(parseEpisodeSelection("first,second", 48).error).toContain("Cannot read first");
    expect(parseEpisodeSelection("7-3", 48).error).toContain("backwards");
  });

  it("accepts any index when the base view episode count is unknown", () => {
    expect(parseEpisodeSelection("120", 0)).toEqual({ episodes: [120], error: "" });
  });
});

describe("daggerMergeCandidates", () => {
  const daggerPass = dataset({ name: "dagger_L4_full48_030000" });
  const daggerWarned = dataset({ name: "dagger_L4_full48_020000" });
  const raw = dataset({ name: "insert_20260821_170816" });
  const items = [
    processing({ path: daggerPass.path, status: "qc_pass" }),
    processing({ path: daggerWarned.path, status: "qc_warn" }),
    processing({ path: raw.path, status: "qc_pass" })
  ];

  it("blocks a QC-warned dataset, which a merge refuses even though a view build allows it", () => {
    const [pass, warned] = daggerMergeCandidates([daggerPass, daggerWarned], items);
    expect(pass.blockedReason).toBe("");
    expect(warned.blockedReason).toContain("QC PASS");
  });

  it("blocks a dataset QC has not run on, and an empty one", () => {
    const empty = dataset({ name: "dagger_empty", totalEpisodes: 0 });
    const [unchecked, blank] = daggerMergeCandidates(
      [dataset({ name: "dagger_unchecked" }), empty],
      []
    );
    expect(unchecked.blockedReason).toContain("Run QC");
    expect(blank.blockedReason).toContain("No episodes");
  });

  it("never offers a training view as a correction source", () => {
    const view = dataset({ name: "insert__delta_ee_from_prev_cmd", datasetKind: "training_view" });
    expect(daggerMergeCandidates([view], [])).toEqual([]);
  });
});

describe("visibleDaggerCandidates", () => {
  const daggerPass = dataset({ name: "dagger_L4_full48_030000" });
  const raw = dataset({ name: "insert_20260821_170816" });
  const candidates = daggerMergeCandidates([daggerPass, raw], []);

  it("shows the DAgger shortlist until the operator asks for every recording", () => {
    expect(
      visibleDaggerCandidates(candidates, { includeNonDagger: false, selectedPaths: [] }).map(
        (candidate) => candidate.dataset.name
      )
    ).toEqual([daggerPass.name]);
    expect(
      visibleDaggerCandidates(candidates, { includeNonDagger: true, selectedPaths: [] })
    ).toHaveLength(2);
  });

  it("keeps a ticked recording visible when the shortlist comes back on", () => {
    expect(
      visibleDaggerCandidates(candidates, {
        includeNonDagger: false,
        selectedPaths: [raw.path]
      })
    ).toHaveLength(2);
  });
});

describe("mergeFormFingerprint", () => {
  const form = {
    baseView: "base",
    daggerRoots: ["/data/b", "/data/a"],
    baseEpisodes: [3, 1],
    outputName: "combined"
  };

  it("ignores the order the sources and episodes were ticked in", () => {
    expect(mergeFormFingerprint(form)).toBe(
      mergeFormFingerprint({ ...form, daggerRoots: ["/data/a", "/data/b"], baseEpisodes: [1, 3] })
    );
  });

  it("changes when the check would be answering a different question", () => {
    expect(mergeFormFingerprint({ ...form, baseEpisodes: [1] })).not.toBe(mergeFormFingerprint(form));
    expect(mergeFormFingerprint({ ...form, baseView: "other" })).not.toBe(mergeFormFingerprint(form));
  });
});

describe("mergeBlockedReason", () => {
  const base = trainingView({ name: "base_view" });
  const ready = {
    baseView: base,
    daggerCount: 1,
    episodeError: "",
    keptBaseEpisodes: 48,
    existingView: null,
    overwrite: false
  };

  it("clears once a base view and a correction dataset are picked", () => {
    expect(mergeBlockedReason(ready)).toBe("");
  });

  it("names the missing input first", () => {
    expect(mergeBlockedReason({ ...ready, baseView: null })).toContain("training view");
    expect(mergeBlockedReason({ ...ready, daggerCount: 0 })).toContain("DAgger correction");
    expect(mergeBlockedReason({ ...ready, episodeError: "Cannot read x" })).toBe("Cannot read x");
    expect(mergeBlockedReason({ ...ready, keptBaseEpisodes: 0 })).toContain("keeps nothing");
  });

  it("requires Replace before a merge can land on an existing view", () => {
    const existing = trainingView({ name: "base_view__plus_dagger_030000" });
    expect(mergeBlockedReason({ ...ready, existingView: existing })).toContain("already exists");
    expect(mergeBlockedReason({ ...ready, existingView: existing, overwrite: true })).toBe("");
  });
});

describe("isDaggerDataset", () => {
  it("recognises the prefix the rollout writes corrections under", () => {
    expect(isDaggerDataset(dataset({ name: "dagger_L4_full48_030000" }))).toBe(true);
    expect(isDaggerDataset(dataset({ name: "insert_20260821_170816" }))).toBe(false);
  });
});
