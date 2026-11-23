import { describe, expect, it } from "vitest";
import type { RolloutCheckpoint, RolloutStatus } from "./rolloutApi";
import { cameraBlocker, executeBlockers, recorderConnecting, recorderNotStreaming } from "./rolloutExecute";

const status = (over: Partial<RolloutStatus> = {}): RolloutStatus => ({
  stages: [],
  jobs: [],
  profile: "thor",
  sandbox: null,
  recorderConnected: true,
  ...over
});
const ckpt = { path: "p", hasWorkspace: true } as RolloutCheckpoint;

describe("executeBlockers", () => {
  it("lets a checkpoint whose preview replayed identically run (the dialog then asks for the e-stop)", () => {
    expect(executeBlockers({ status: status(), checkpoint: ckpt, previewReplayed: true })).toEqual([]);
  });
  it("names every missing condition", () => {
    const out = executeBlockers({
      status: status({ recorderConnected: false, sandbox: "hph" }),
      checkpoint: ckpt,
      previewReplayed: false
    });
    expect(out.join("|")).toContain("sandbox");
    expect(out.join("|")).toContain("connect the recorder");
    expect(out.join("|")).toContain("Replay check");
    expect(out.join("|")).not.toContain("locked");
  });
});

describe("cameraBlocker", () => {
  const cam10 = { ...ckpt, cameras: ["cam_10"] } as RolloutCheckpoint;
  it("is silent on a full session and when the inference session has the checkpoint's cameras", () => {
    expect(cameraBlocker(status(), cam10)).toBe("");
    expect(cameraBlocker(status({ recorderCameras: ["cam_10"] }), cam10)).toBe("");
  });
  it("names the camera an inference session lacks, and blocks execution with it", () => {
    const s = status({ recorderCameras: ["cam_06"] });
    expect(cameraBlocker(s, cam10)).toContain("this checkpoint reads cam_10");
    const out = executeBlockers({ status: s, checkpoint: cam10, previewReplayed: true });
    expect(out).toEqual([cameraBlocker(s, cam10)]);
  });
});

describe("recorderConnecting", () => {
  const sensors = [
    { id: "cam_10", kind: "camera", state: "running" },
    { id: "box/gripper", kind: "box_collection", state: "warning" }
  ];
  it("holds Disconnect while the recorder brings its sensors up, counting the ready ones", () => {
    const s = status({ recorder: { state: "connecting", sensors, ready: false, pending: true, failed: false } });
    expect(recorderConnecting(s)).toEqual({
      pending: true,
      stuck: false,
      label: "Connecting… 1/2 sensors ready",
      waiting: ["box/gripper"]
    });
    expect(executeBlockers({ status: s, checkpoint: ckpt, previewReplayed: true })).toEqual([
      "the recorder is still connecting"
    ]);
  });
  it("releases it once every sensor streams, or once one failed (so a broken connect can be torn down)", () => {
    const ready = sensors.map((x) => ({ ...x, state: "running" }));
    expect(recorderConnecting(status({ recorder: { state: "armed", sensors: ready, ready: true, pending: false, failed: false } })).pending).toBe(false);
    expect(recorderConnecting(status({ recorder: { state: "error", sensors, ready: false, pending: false, failed: true } })).pending).toBe(false);
    expect(recorderConnecting(status({ recorderConnected: false })).pending).toBe(false);
  });
  it("lets a connect stuck past the gateway's limit be cancelled, naming what never came up", () => {
    const s = status({ recorder: { state: "connecting", sensors, ready: false, pending: true, failed: false, stuck: true, elapsedS: 61 } });
    expect(recorderConnecting(s)).toMatchObject({ pending: true, stuck: true, waiting: ["box/gripper"] });
  });
});

describe("recorderNotStreaming", () => {
  const cam = { id: "cam_10", kind: "camera", state: "running" };
  const box = (state: string) => ({ id: "box/gripper", kind: "box_collection", state });
  const up = (sensors: { id: string; kind: string; state: string }[], pending = false) =>
    status({ recorderCameras: ["cam_10"], recorder: { state: pending ? "connecting" : "armed", sensors, ready: !pending, pending, failed: false } });
  it("names a BOX the recorder came up without (Box devices: (none) marks its rows error)", () => {
    expect(recorderNotStreaming(up([cam, box("error")]))).toEqual({ boxes: ["box/gripper"], cameras: [], noBox: false });
  });
  it("says when there is no BOX row at all", () => {
    expect(recorderNotStreaming(up([cam])).noBox).toBe(true);
  });
  it("is silent when every sensor streams, while connecting, and when not connected", () => {
    expect(recorderNotStreaming(up([cam, box("running")]))).toEqual({ boxes: [], cameras: [], noBox: false });
    expect(recorderNotStreaming(up([cam, box("warning")], true))).toEqual({ boxes: [], cameras: [], noBox: false });
    expect(recorderNotStreaming(status({ recorderConnected: false })).boxes).toEqual([]);
  });
  it("names a camera that failed", () => {
    expect(recorderNotStreaming(up([{ ...cam, state: "error" }, box("running")])).cameras).toEqual(["cam_10"]);
  });
});
