import { describe, expect, it } from "vitest";

import pageSource from "./TrainingPage.tsx?raw";
import { runJobToAdopt } from "./trainingRunForm";

describe("the Training form while a job runs", () => {
  it("takes over a running job once, and never an idle or finished one", () => {
    const run = (state: string, jobName = "view__pi05__20261021_092125") => ({ state, jobName });
    expect(runJobToAdopt(run("running"), "")).toBe("view__pi05__20261021_092125");
    expect(runJobToAdopt(run("starting"), "")).toBe("view__pi05__20261021_092125");
    expect(runJobToAdopt(run("running"), "view__pi05__20261021_092125")).toBeNull();
    expect(runJobToAdopt(run("running", "view__pi05__20261022_010000"), "view__pi05__20261021_092125")).toBe(
      "view__pi05__20261022_010000"
    );
    for (const state of ["idle", "complete", "failed", "stopped"]) expect(runJobToAdopt(run(state), "")).toBeNull();
    expect(runJobToAdopt(null, "")).toBeNull();
  });

  it("fills host, view and every parameter from the job's own history record", () => {
    expect(
      pageSource.includes("setHostId(run.hostId);") && pageSource.includes("setViewName(run.viewName);"),
      "The form no longer takes the running job's host and view; it shows this machine and no " +
        "view beside a run on another host."
    ).toBe(true);
    expect(
      pageSource.includes("item.jobName === job") && pageSource.includes("copyRunParams(entry);"),
      "The running job's parameters no longer come from its history record."
    ).toBe(true);
  });

  it("takes the host from the run even when the job has no history record", () => {
    // R4 was started on Dante outside the page and adopted before any record existed; waiting
    // for one kept the Machine field on the NUC (10-26).
    const hostTaken = pageSource.indexOf("setHostId(run.hostId);");
    const paramsGated = pageSource.indexOf("if (entry) {\n      copyRunParams(entry);");
    expect(hostTaken, "setHostId(run.hostId) is gone").toBeGreaterThan(0);
    expect(paramsGated, "only the parameters may wait for the history record").toBeGreaterThan(hostTaken);
  });
});
