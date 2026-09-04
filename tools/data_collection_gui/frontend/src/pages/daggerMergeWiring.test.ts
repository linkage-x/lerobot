import { describe, expect, it } from "vitest";

import apiSource from "../api.ts?raw";
import pageSource from "./DatasetExportPage.tsx?raw";

/**
 * The merge request is a plain object literal, so every field in it is optional at the type
 * level and a dropped one fails silently on the gateway's side: `baseEpisodes` absent means
 * "merge every base episode", which is exactly the shape of the mistake this field exists to
 * prevent -- the holdout the checkpoint was evaluated on ends up in the set it is retrained on,
 * and nothing anywhere reports an error. `overwrite` absent turns a deliberate replace into a
 * refusal; `copyVideos` absent leaves symlinks where copies were asked for.
 */
describe("DAgger merge wiring", () => {
  it("mounts the merge panel on the Training View page", () => {
    expect(
      pageSource.includes("<DaggerMergePanel"),
      "DatasetExportPage no longer renders DaggerMergePanel; the merge has no entry point in the UI"
    ).toBe(true);
  });

  const start = pageSource.indexOf("const request = {");
  const request = start < 0 ? "" : pageSource.slice(start, pageSource.indexOf("};", start));

  it("builds a merge request", () => {
    expect(start, "The merge panel no longer builds a request object").toBeGreaterThan(-1);
  });

  for (const field of [
    "baseView",
    "daggerRoots",
    "baseEpisodes",
    "outputName",
    "overwrite",
    "copyVideos"
  ]) {
    it(`sends ${field}`, () => {
      expect(
        request.includes(field),
        `The merge request drops ${field}. The control that sets it stays on the page and goes ` +
          "on producing merges that ignore it, with no error anywhere."
      ).toBe(true);
    });
  }

  it("keeps both merge endpoints reachable from the client", () => {
    expect(apiSource).toContain("/api/training/dagger-merge/check");
    expect(apiSource).toContain("/api/training/dagger-merge/start");
  });

  it("gates the merge button on a check that answers the current form", () => {
    expect(
      pageSource.includes("!checkIsCurrent"),
      "The Merge button no longer requires a current compatibility check, so a stale 'compatible' " +
        "can start a merge of different datasets"
    ).toBe(true);
  });
});
