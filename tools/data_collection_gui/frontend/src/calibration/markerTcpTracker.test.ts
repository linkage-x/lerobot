import { describe, expect, it } from "vitest";
import type { MarkerTcpSample, MarkerTcpSession } from "../types";
import { e1pChoices, e1pReadiness, e1pSamples, liveTrackerSessionId, trackerLink } from "./markerTcpTracker";

const BOX = "box1672693301";

function sample(over: Partial<MarkerTcpSample> = {}): MarkerTcpSample {
  return {
    id: "sample_001",
    side: BOX,
    boxId: BOX,
    condition: "e1p_01",
    source: "recording",
    status: "saved",
    datasetRoot: "outputs/datasets/thor_20260923",
    episodeIndex: 3,
    staticTransformPath: "",
    note: "",
    createdAt: "",
    laserTracker: true,
    trackerSessionId: "lt_20260923_101500",
    ...over,
  };
}

function session(samples: MarkerTcpSample[], over: Partial<MarkerTcpSession> = {}): MarkerTcpSession {
  return {
    active: true,
    sessionName: "marker_tcp_20260923",
    sessionRoot: "",
    stage: "capture",
    samples,
    pendingSampleId: "",
    message: "",
    reportPath: "",
    ...over,
  };
}

const DISCONNECTED = { state: "idle" };
const LIVE = {
  state: "armed",
  laserTracker: true,
  laserTrackerReady: true,
  laserTrackerHomed: true,
  laserTrackerDetail: "session lt_20260923_101500 · beam on SMR",
};

describe("e1pSamples", () => {
  it("keeps one BOX, one clamping, saved, with the tracker", () => {
    const picked = e1pSamples(
      [
        sample({ id: "a" }),
        sample({ id: "b", condition: "e1p_02" }),
        sample({ id: "c", status: "discarded" }),
        sample({ id: "d", laserTracker: false }),
        sample({ id: "e", boxId: "box1819152274", side: "box1819152274" }),
      ],
      BOX,
      " e1p_01 ",
    );
    expect(picked.map((s) => s.id)).toEqual(["a"]);
  });
});

describe("trackerLink", () => {
  it("says what to change to make the next sample count for E1p", () => {
    expect(trackerLink(DISCONNECTED).dot).toBe("idle");
    expect(trackerLink({ state: "armed", laserTracker: false }).text).toContain("跟踪仪开关");
    expect(trackerLink({ ...LIVE, laserTrackerReady: false }).dot).toBe("warning");
    expect(trackerLink(LIVE).dot).toBe("running");
    // Locked but never homed (W2): named as such, not as a beam problem.
    const unhomed = trackerLink({ ...LIVE, laserTrackerHomed: false, laserTrackerReady: false });
    expect(unhomed.dot).toBe("warning");
    expect(unhomed.text).toContain("Home");
    const broken = trackerLink({ ...LIVE, laserTrackerBeamBroken: true, laserTrackerReady: false });
    expect(broken.dot).toBe("warning");
    expect(broken.text).toContain("断过光");
  });
});

describe("e1pReadiness", () => {
  const base = { boxId: BOX, condition: "e1p_01", stationPath: "outputs/laser_tracker/station_1.json" };

  it("runs on landed samples with a station", () => {
    const ready = e1pReadiness({ ...base, session: session([sample()]), recording: DISCONNECTED });
    expect(ready.canRun).toBe(true);
    expect(ready.samples).toHaveLength(1);
  });

  it("waits for Disconnect while this Connect's stream is still open", () => {
    const ready = e1pReadiness({ ...base, session: session([sample()]), recording: LIVE });
    expect(ready.canRun).toBe(false);
    expect(ready.reason).toContain("Disconnect");
  });

  it("does not wait on samples whose Connect already ended", () => {
    const older = sample({ trackerSessionId: "lt_20260922_090000" });
    const ready = e1pReadiness({ ...base, session: session([older]), recording: LIVE });
    expect(ready.canRun).toBe(true);
  });

  it("asks for a station, and says the order of capture does not matter", () => {
    const ready = e1pReadiness({ ...base, stationPath: "", session: session([sample()]), recording: DISCONNECTED });
    expect(ready.canRun).toBe(false);
    expect(ready.reason).toContain("station");
    expect(ready.reason).toContain("先录 pivot");
  });

  it("names the clamping that has no tracker samples", () => {
    const ready = e1pReadiness({
      ...base,
      session: session([sample({ laserTracker: false })]),
      recording: DISCONNECTED,
    });
    expect(ready.canRun).toBe(false);
    expect(ready.reason).toContain("e1p_01");
  });

  it("refuses while a sample is recording or a solve is running", () => {
    const recording = e1pReadiness({
      ...base,
      session: session([sample()], { pendingSampleId: "sample_002" }),
      recording: DISCONNECTED,
    });
    const solving = e1pReadiness({ ...base, session: session([sample()], { stage: "solving" }), recording: DISCONNECTED });
    expect(recording.canRun).toBe(false);
    expect(solving.canRun).toBe(false);
  });
});

describe("liveTrackerSessionId", () => {
  it("reads the id out of the device detail", () => {
    expect(liveTrackerSessionId(LIVE.laserTrackerDetail)).toBe("lt_20260923_101500");
    expect(liveTrackerSessionId(undefined)).toBe("");
  });
});

describe("e1pChoices", () => {
  it("lists what was recorded with the tracker, per BOX and clamping", () => {
    const choices = e1pChoices([
      sample({ id: "a" }),
      sample({ id: "b" }),
      sample({ id: "c", condition: "e1p_02" }),
      sample({ id: "d", laserTracker: false, condition: "same_mount_01" }),
      sample({ id: "e", status: "recording" }),
    ]);
    expect(choices).toEqual([
      {
        boxId: BOX,
        conditions: [
          { condition: "e1p_01", samples: 2 },
          { condition: "e1p_02", samples: 1 },
        ],
      },
    ]);
  });
});
