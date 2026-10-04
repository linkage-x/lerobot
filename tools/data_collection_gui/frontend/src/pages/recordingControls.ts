import type { CalibrationSession, Fr3TeleopStatus, MarkerTcpSession, RecordingStatus, TrackerMountSession } from "../types";

export function hasActiveCalibrationCapture(snapshot: {
  calibrationSession?: CalibrationSession;
  trackerMountSession?: TrackerMountSession;
  markerTcp?: MarkerTcpSession;
}) {
  return [snapshot.calibrationSession, snapshot.trackerMountSession, snapshot.markerTcp]
    .some((session) => session?.active && session.stage === "capture");
}

/** The buttons and keyboard use exactly the same recorder and motion gates. */
export function recordingControlAvailability(
  status: RecordingStatus,
  fr3?: Fr3TeleopStatus,
  mountSession?: TrackerMountSession,
  busy = false,
  calibrationCapture = false,
) {
  const isConnected =
    status.pid != null ||
    ["connecting", "armed", "recording", "review", "saving", "discarding"].includes(status.state);
  const mountHeld = Boolean(mountSession?.active) && mountSession?.stage === "capture";
  const captureHeld = mountHeld || calibrationCapture;
  const trackerBlocking = Boolean(status.laserTracker) && !status.laserTrackerReady;
  return {
    isConnected,
    canConnect: !busy && !isConnected,
    canStartEpisode: !busy && status.state === "armed" && !captureHeld && !trackerBlocking
      && (!fr3?.enabled || fr3.state === "running"),
    canResolveEpisode: !busy && (status.state === "recording" || status.state === "review"),
    canExit: !busy && isConnected,
    canStartFr3: !busy && Boolean(fr3?.enabled) && status.state === "armed" && !captureHeld
      && status.boxEnabled !== false
      && (fr3?.state === "idle" || fr3?.state === "error"),
    canStopFr3: !busy && Boolean(fr3?.enabled)
      && ["starting", "moving_to_start", "running"].includes(fr3?.state ?? "idle"),
  };
}

export type RecordingControlAction = "connect" | "startFr3" | "startEpisode" | "save" | "discard" | "exit";
type ShortcutInput = Pick<KeyboardEvent, "key" | "repeat" | "isComposing" | "defaultPrevented" | "ctrlKey" | "metaKey" | "altKey">;

/** Ignore browser shortcuts and text entry; F is a deliberate motion command. */
export function recordingShortcutAction(
  event: ShortcutInput,
  controls: ReturnType<typeof recordingControlAvailability>,
  editable = false,
): RecordingControlAction | null {
  if (event.defaultPrevented || event.repeat || event.isComposing || event.ctrlKey || event.metaKey || event.altKey || editable) {
    return null;
  }
  const key = event.key.toLowerCase();
  if (key === "c" && controls.canConnect) return "connect";
  if (key === "f" && controls.canStartFr3) return "startFr3";
  if (key === "e" && controls.canStartEpisode) return "startEpisode";
  if (key === "s" && controls.canResolveEpisode) return "save";
  if (key === "d" && controls.canResolveEpisode) return "discard";
  if (key === "escape" && controls.canExit) return "exit";
  return null;
}
