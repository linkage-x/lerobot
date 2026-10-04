import { afterEach, describe, expect, it, vi } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import type { ConfigSummary, Fr3TeleopStatus, RecordingStatus, TrackerMountSession } from "../types";
import { DataCollectionGuiApi } from "../api";
import { Fr3StatusPanel, RecordingPanel } from "./LiveRecordPage";
import { hasActiveCalibrationCapture, recordingControlAvailability, recordingShortcutAction } from "./recordingControls";

const record = (state: RecordingStatus["state"] = "armed"): RecordingStatus => ({
  state, pid: state === "idle" ? null : 123, datasetRoot: "outputs/test", repoId: "local/test",
  episodeIndex: 0, savedEpisodes: 0, frameIndex: 0, targetFrames: 600, queueDepth: 0, message: "Ready",
});
const fr3 = (state: Fr3TeleopStatus["state"] = "idle"): Fr3TeleopStatus => ({
  enabled: true, state, message: "Ready", telemetry: {},
});
const key = (value: string, extra: Partial<KeyboardEvent> = {}) => ({
  key: value, repeat: false, isComposing: false, defaultPrevented: false, ctrlKey: false, metaKey: false, altKey: false, ...extra,
});

describe("BOX capture and explicit FR3 motion", () => {
  it("preserves C and E for configs and snapshots without FR3", () => {
    expect(recordingShortcutAction(key("C"), recordingControlAvailability(record("idle")))).toBe("connect");
    expect(recordingShortcutAction(key("E"), recordingControlAvailability(record()))).toBe("startEpisode");
    expect(recordingShortcutAction(key("F"), recordingControlAvailability(record()))).toBeNull();
    expect(recordingControlAvailability(record(), { ...fr3("error"), enabled: false }).canStartEpisode).toBe(true);
  });

  it("requires connected cameras and BOX before F, then running motion before E", () => {
    expect(recordingControlAvailability(record("idle"), fr3()).canStartFr3).toBe(false);
    expect(recordingControlAvailability(record("connecting"), fr3()).canStartFr3).toBe(false);
    const connected = recordingControlAvailability(record(), fr3());
    expect(recordingShortcutAction(key("C"), connected)).toBeNull();
    expect(recordingShortcutAction(key("F"), connected)).toBe("startFr3");
    expect(recordingShortcutAction(key("E"), connected)).toBeNull();
    const running = recordingControlAvailability(record(), fr3("running"));
    expect(recordingShortcutAction(key("F"), running)).toBeNull();
    expect(recordingShortcutAction(key("E"), running)).toBe("startEpisode");
    expect(recordingControlAvailability({ ...record(), boxEnabled: false }, fr3()).canStartFr3).toBe(false);
  });

  it.each(["starting", "moving_to_start", "stopping"] as const)("prevents F and E while motion is %s", (state) => {
    const controls = recordingControlAvailability(record(), fr3(state));
    expect(recordingShortcutAction(key("F"), controls)).toBeNull();
    expect(recordingShortcutAction(key("E"), controls)).toBeNull();
  });

  it("offers F after a fault while blocking E until a successful restart", () => {
    const controls = recordingControlAvailability(record(), fr3("error"));
    expect(controls.canStartFr3).toBe(true);
    expect(controls.canStartEpisode).toBe(false);
  });

  it.each(["recording", "review", "saving", "discarding", "error"] as const)("never moves to start while recorder is %s", (state) => {
    expect(recordingControlAvailability(record(state), fr3("error")).canStartFr3).toBe(false);
  });

  it.each(["recording", "review"] as const)("keeps Save and Discard separate from motion in %s", (state) => {
    const controls = recordingControlAvailability(record(state), fr3("error"));
    expect(recordingShortcutAction(key("S"), controls)).toBe("save");
    expect(recordingShortcutAction(key("D"), controls)).toBe("discard");
    expect(recordingShortcutAction(key("Escape"), controls)).toBe("exit");
  });

  it("allows an explicit Stop FR3 during recording", () => {
    expect(recordingControlAvailability(record("recording"), fr3("running")).canStopFr3).toBe(true);
  });

  it("applies tracker and calibration capture gates to the keyboard too", () => {
    expect(recordingShortcutAction(key("E"), recordingControlAvailability({ ...record(), laserTracker: true, laserTrackerReady: false }, fr3("running")))).toBeNull();
    const mount = { active: true, stage: "capture" } as TrackerMountSession;
    expect(recordingShortcutAction(key("F"), recordingControlAvailability(record(), fr3(), mount))).toBeNull();
    expect(recordingShortcutAction(key("E"), recordingControlAvailability(record(), fr3("running"), mount))).toBeNull();
  });

  it.each(["calibrationSession", "trackerMountSession", "markerTcp"] as const)("blocks F during %s capture and releases it afterwards", (sessionKey) => {
    const snapshot = { [sessionKey]: { active: true, stage: "capture" } };
    const held = hasActiveCalibrationCapture(snapshot as Parameters<typeof hasActiveCalibrationCapture>[0]);
    expect(held).toBe(true);
    expect(recordingControlAvailability(record(), fr3(), undefined, false, held).canStartFr3).toBe(false);
    expect(recordingShortcutAction(key("F"), recordingControlAvailability(record(), fr3(), undefined, false, held))).toBeNull();
    expect(hasActiveCalibrationCapture({ [sessionKey]: { active: false, stage: "capture" } } as Parameters<typeof hasActiveCalibrationCapture>[0])).toBe(false);
    expect(hasActiveCalibrationCapture({ [sessionKey]: { active: true, stage: "ready" } } as Parameters<typeof hasActiveCalibrationCapture>[0])).toBe(false);
  });

  it("suppresses all commands while busy and ignores typing, held keys and browser combinations", () => {
    const busy = recordingControlAvailability(record(), fr3(), undefined, true);
    expect(recordingShortcutAction(key("F"), busy)).toBeNull();
    expect(recordingShortcutAction(key("Escape"), busy)).toBeNull();
    const controls = recordingControlAvailability(record(), fr3());
    for (const extra of [{ repeat: true }, { isComposing: true }, { ctrlKey: true }, { metaKey: true }, { altKey: true }, { defaultPrevented: true }]) {
      expect(recordingShortcutAction(key("F", extra), controls)).toBeNull();
    }
    expect(recordingShortcutAction(key("F"), controls, true)).toBeNull();
  });
});

const config: ConfigSummary = {
  configPath: "thor.yaml", repoId: "local/test", root: "outputs/test", fps: 60, episodeTimeS: 10,
  targetFrames: 600, numEpisodes: "unlimited", video: true, streamingEncoding: true, vcodec: "h264",
  softSync: false, rerun: { displayData: false, savePath: "" }, rigType: "gmsl2",
};
function panel(status: Fr3TeleopStatus | undefined, state: RecordingStatus["state"] = "armed") {
  return renderToStaticMarkup(<RecordingPanel status={record(state)} config={config} busy={false}
    fr3Teleop={status} onConnect={() => {}} onStart={() => {}} onStop={() => {}} onStartFr3={() => {}} onStopFr3={() => {}} />);
}
function buttonDisabled(markup: string, text: string) {
  const button = markup.match(new RegExp(`<button([^>]*)>${text}(?:\\s|<)`));
  expect(button, `Missing button ${text}`).not.toBeNull();
  return button?.[1].includes("disabled=");
}

describe("visible recording controls", () => {
  it("renders the same F/E gates as the keyboard and retains legacy buttons", () => {
    const idle = panel(fr3());
    expect(buttonDisabled(idle, "Start FR3")).toBe(false);
    expect(buttonDisabled(idle, "StartEpisode")).toBe(true);
    const running = panel(fr3("running"));
    expect(buttonDisabled(running, "Start FR3")).toBe(true);
    expect(buttonDisabled(running, "StartEpisode")).toBe(false);
    expect(buttonDisabled(panel(fr3("error")), "Start FR3")).toBe(false);
    expect(buttonDisabled(panel(undefined), "StartEpisode")).toBe(false);
    expect(panel(undefined)).not.toContain("Start FR3");
    for (const label of ["Connect", "StartEpisode", "Save", "Discard", "Exit"]) expect(idle).toContain(label);
  });

  it("renders actionable fault recovery and measured telemetry without treating missing values as zero", () => {
    const fault = renderToStaticMarkup(<Fr3StatusPanel status={{ ...fr3("error"), message: "FCI reflex stop", telemetry: {
      q: [0.123], tau_J: [1.234], measured_tcp: [0.4, 0.1, 0.5, 0, 0, 0], control_command_success_rate: 0.999,
    } }} />);
    expect(fault).toContain('role="alert"');
    expect(fault).toContain("FCI reflex stop");
    expect(fault).toContain("press F");
    expect(fault).toContain("Measurements are available while FR3 teleoperation is running");
    expect(fault).toContain("0.123");
    expect(fault).toContain("1.234");
    expect(fault).toContain("0.4000, 0.1000, 0.5000");
    expect(fault).toContain("—");
  });

  it("does not claim stopped motion while worker shutdown is unconfirmed", () => {
    const fault = renderToStaticMarkup(<Fr3StatusPanel status={{ ...fr3("error"), pid: 123 }} />);
    expect(fault).toContain("shutdown is not confirmed");
    expect(fault).toContain("hardware controls");
    expect(fault).not.toContain("motion has stopped");
  });
});

afterEach(() => vi.unstubAllGlobals());
describe("FR3 API acknowledgement", () => {
  it("sends only the selected Thor cameras and BOX choice with C", async () => {
    vi.stubGlobal("window", { setTimeout });
    const fetchMock = vi.fn().mockRejectedValue(new Error("offline"));
    vi.stubGlobal("fetch", fetchMock);
    const api = new DataCollectionGuiApi();
    await api.connectRecording(undefined, false, { cameraIds: [6, 7], boxEnabled: true });
    expect(fetchMock.mock.calls[0][0]).toBe("/api/handheld/record/connect?laser_tracker=0&camera_ids=6%2C7&box=1");
  });

  it("uses the same gateway routes and refuses to simulate a successful start on network loss", async () => {
    vi.stubGlobal("window", { setTimeout });
    const fetchMock = vi.fn().mockRejectedValue(new Error("offline"));
    vi.stubGlobal("fetch", fetchMock);
    const api = new DataCollectionGuiApi();
    const snapshot = await api.startFr3Teleop();
    expect(fetchMock.mock.calls[0][0]).toBe("/api/fr3/teleop/start");
    expect(fetchMock.mock.calls[0][1].method).toBe("POST");
    expect(snapshot.fr3Teleop?.state).toBe("idle");
    expect(api.consumeCommandFailure()?.message).toContain("Gateway unavailable");
    await api.stopFr3Teleop();
    expect(fetchMock.mock.calls.some(([url]) => url === "/api/fr3/teleop/stop")).toBe(true);
    expect(api.consumeCommandFailure()?.message).toContain("not acknowledged");
  });
});
