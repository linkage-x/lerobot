import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../apiClient";
import type { GuiSnapshot } from "../api";
import type { BoxPreviewPayload, BoxCaliLog, BoxCaliLogLine, CollectionTask, ConfigSummary, DeviceStatus, EpisodeAnnotation, EventLogItem, ProcessingItem, ProcessingStatus, RecordedDataset, RecordingBackend, RecordingStatus, ReplayStatus, RolloutLandmarks, SceneResetRequest, SubtaskSegment, TaskStatus, DatasetExportStatus, AnnotationOutcome, AnnotationQuality, ReviewStatus } from "../types";
import { StatusDot, Metric, PageHeader, stateLabel, QualityOverview, processingStatusLabel, datasetNamePrefixes, taskDatasetBaseName, processingItemsForTask, taskNeedsQcExportConfirmation } from "../shared/ui";
import { SceneResetPanel } from "./SceneResetPanel";

export function DeviceList({ devices, config }: { devices: DeviceStatus[]; config: ConfigSummary }) {
  const grouped = useMemo(() => {
    return devices.reduce<Record<string, DeviceStatus[]>>((acc, device) => {
      acc[device.kind] = [...(acc[device.kind] ?? []), device];
      return acc;
    }, {});
  }, [devices]);

  const cameraCount = grouped["camera"]?.length ?? 0;
  const runningCameras = grouped["camera"]?.filter((d) => d.state === "running").length ?? 0;
  const errorCameras = grouped["camera"]?.filter((d) => d.state === "error").length ?? 0;

  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>Devices</h2>
        <span>{devices.length} streams</span>
      </div>
      {Object.entries(grouped).map(([kind, items]) => {
        const kindLabel = kind === "camera" && config.rigType === "gmsl2"
          ? `GMSL2 cameras`
          : kind === "box_collection"
            ? "BOX sensors"
            : kind.replace("_", " ");
        const kindSummary = kind === "camera" && config.rigType === "gmsl2"
          ? `${runningCameras}/${cameraCount} running${errorCameras ? `, ${errorCameras} error` : ""}`
          : `${items.length} devices`;
        return (
          <div className="device-group" key={kind}>
            <div className="device-group-header">
              <h3>{kindLabel}</h3>
              <small>{kindSummary}</small>
            </div>
            {items.map((device) => (
              <div className="device-row" key={device.id}>
                <div>
                  <div className="row-title">
                    <StatusDot state={device.state} />
                    <strong>{device.id}</strong>
                  </div>
                  <p>{device.label}</p>
                </div>
                <div className="device-stats">
                  <span>{device.fps} fps</span>
                  <span>{device.latencyMs} ms</span>
                  <small>{device.detail}</small>
                </div>
              </div>
            ))}
          </div>
        );
      })}
    </section>
  );
}

export function HardwareSyncBadge({ config }: { config: ConfigSummary }) {
  const hw = config.hardwareSync;
  if (!hw) return null;
  const trigLabel = hw.trigMode === 1 ? "PWM slave" : hw.trigMode === 0 ? "free-run" : `trig ${hw.trigMode}`;
  return (
    <div className={`hw-sync-badge ${hw.enabled ? "hw-sync-on" : "hw-sync-off"}`}>
      <span className="hw-sync-icon">{hw.enabled ? "◉" : "○"}</span>
      <span>HW Sync {hw.enabled ? "ON" : "OFF"}</span>
      {hw.enabled && <small>{hw.fps} Hz {trigLabel}{hw.pwmChip ? ` · ${hw.pwmChip}` : ""}</small>}
    </div>
  );
}

export function CameraEncodingInfo({ config }: { config: ConfigSummary }) {
  const cam = config.cameraDefaults;
  if (!cam || !cam.codec) return null;
  const res = cam.width && cam.height ? `${cam.width}x${cam.height}` : "";
  const bitrate = cam.bitrateKbps ? `${cam.bitrateKbps} kbps` : "";
  const exposure = cam.exposureUs ? `exp ${cam.exposureUs} us` : "";
  const gain = cam.gain ? `gain ${cam.gain}` : "";
  return (
    <div className="encoding-info">
      <Metric label="Codec" value={`${cam.codec.toUpperCase()} / ${cam.container || "mkv"}`} />
      <Metric label="Resolution" value={res || "—"} />
      <Metric label="Bitrate" value={bitrate || "—"} />
      <Metric label="Pipeline" value={cam.pipeline || "—"} />
      <Metric label="Exposure" value={exposure || "auto"} />
      <Metric label="Gain" value={gain || "auto"} />
    </div>
  );
}

export function RecorderLogStream({ lines }: { lines: string[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  // Stop auto-scrolling once the user has manually scrolled up; resume once
  // they scroll back to within the bottom threshold.
  const stickToBottomRef = useRef(true);

  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    stickToBottomRef.current = distanceFromBottom < 24;
  };

  useEffect(() => {
    const el = containerRef.current;
    if (el && stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [lines]);

  if (lines.length === 0) return null;
  return (
    <div
      className="process-output-log"
      ref={containerRef}
      onScroll={handleScroll}
      role="log"
      aria-live="polite"
    >
      {lines.map((line, i) => (
        <div className="process-output-line" key={`${i}-${line}`}>{line}</div>
      ))}
    </div>
  );
}

// Single source of truth for which record controls are available in a given
// recorder state. Shared by the RecordingPanel buttons AND the LiveRecord
// keyboard shortcuts, so a shortcut can never fire an action that a disabled
// button wouldn't.
export function recordingControlAvailability(status: RecordingStatus) {
  const isConnected =
    status.pid != null ||
    ["connecting", "armed", "recording", "review", "saving", "discarding"].includes(status.state);
  return {
    isConnected,
    canConnect: !isConnected,
    canStartEpisode: status.state === "armed",
    canResolveEpisode: status.state === "recording" || status.state === "review",
    canSetStartPose: status.state === "armed" || status.state === "recording" || status.state === "review",
    canExit: isConnected,
  };
}

const syncStatusLabels: Record<string, string> = {
  unknown: "not measured yet",
  pass: "aligned",
  fail: "out of budget",
  unavailable: "audit unavailable"
};

/** Per-episode capture-timestamp verdict, surfaced while the rig is still set up. */
export function SyncAuditPanel({ status }: { status: RecordingStatus }) {
  const syncStatus = status.syncStatus ?? "unknown";
  if (syncStatus === "unknown" && !status.syncSummary) return null;
  const dotState = syncStatus === "pass" ? "running" : syncStatus === "fail" ? "error" : "warning";
  const warnings = status.syncWarnings ?? [];
  return (
    <div className={`sync-audit sync-audit-${syncStatus}`}>
      <div className="sync-audit-heading">
        <StatusDot state={dotState} />
        <strong>Timestamp sync</strong>
        <span>{syncStatusLabels[syncStatus] ?? syncStatus}</span>
      </div>
      {status.syncSummary ? <code className="sync-audit-summary">{status.syncSummary}</code> : null}
      {warnings.length > 0 ? (
        <ul className="sync-audit-warnings">
          {warnings.map((warning, index) => (
            <li key={`${index}-${warning}`}>{warning}</li>
          ))}
        </ul>
      ) : null}
      {status.syncReportPath ? <small>report: {status.syncReportPath}</small> : null}
    </div>
  );
}

export function RecordingPanel({
  status,
  config,
  busy,
  onConnect,
  onStart,
  onStop,
  onSetStartPose,
  onResetStartPose,
  logLines,
  backendPicker,
  episodeDurationControl,
  episodeDurationValid = true,
  showStartPoseControl = false
}: {
  status: RecordingStatus;
  config: ConfigSummary;
  busy: boolean;
  onConnect: () => void;
  onStart: () => void;
  onStop: (action: "save" | "discard" | "exit") => void;
  onSetStartPose?: () => void;
  onResetStartPose?: () => void;
  logLines?: string[];
  backendPicker?: React.ReactNode;
  episodeDurationControl?: React.ReactNode;
  episodeDurationValid?: boolean;
  showStartPoseControl?: boolean;
}) {
  const progress = Math.round((status.frameIndex / Math.max(status.targetFrames, 1)) * 100);
  const { isConnected, canStartEpisode, canResolveEpisode, canSetStartPose, canExit } =
    recordingControlAvailability(status);
  const isGmsl = config.rigType === "gmsl2";
  const panelTitle = backendPicker ? "FR3 Record" : isGmsl ? "GMSL2 Record" : "Handheld Record";

  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>{panelTitle}</h2>
        <span className="state-pill">
          <StatusDot state={status.state} />
          {stateLabel(status.state)}
        </span>
      </div>
      {backendPicker}
      {episodeDurationControl}
      {isGmsl && <HardwareSyncBadge config={config} />}
      <div className="config-grid">
        <Metric label="Config" value={config.configPath} />
        <Metric label="Repo" value={status.repoId} />
        <Metric label="Root" value={status.datasetRoot} />
        <Metric label="FPS" value={config.fps} />
        <Metric label="Episode" value={`${config.episodeTimeS}s / ${status.targetFrames} frames`} />
        <Metric label="Encoding" value={`${config.vcodec || "raw"}${config.streamingEncoding ? ", streaming" : ""}`} />
      </div>
      {isGmsl && <CameraEncodingInfo config={config} />}
      <div className="progress">
        <div className="progress-bar" style={{ width: `${progress}%` }} />
      </div>
      <div className="control-row">
        <button disabled={busy || isConnected || !episodeDurationValid} onClick={onConnect} title="Shortcut: C">Connect <kbd>C</kbd></button>
        <button disabled={busy || !canStartEpisode} onClick={onStart} title="Shortcut: E">StartEpisode <kbd>E</kbd></button>
        <button disabled={busy || !canResolveEpisode} onClick={() => onStop("save")} title="Shortcut: S">Save <kbd>S</kbd></button>
        <button disabled={busy || !canResolveEpisode} onClick={() => onStop("discard")} title="Shortcut: D">Discard <kbd>D</kbd></button>
        {showStartPoseControl ? (
          <button
            disabled={busy || !canSetStartPose || !onSetStartPose}
            onClick={onSetStartPose}
            title="Use current FR3 joints as the next return pose"
          >
            Set Home
          </button>
        ) : null}
        {showStartPoseControl ? (
          <button
            className="ghost"
            disabled={busy || !canSetStartPose || !onResetStartPose}
            onClick={onResetStartPose}
            title="Return pose back to the one this recorder was launched with"
          >
            Reset Home
          </button>
        ) : null}
        <button disabled={busy || !canExit} onClick={() => onStop("exit")} title="Shortcut: Esc">Exit <kbd>Esc</kbd></button>
      </div>
      <div className="summary-grid">
        <Metric label="Frame" value={`${status.frameIndex}/${status.targetFrames}`} />
        <Metric label="Queue" value={status.queueDepth} />
        <Metric label="Saved" value={status.savedEpisodes} />
        <Metric label="PID" value={status.pid ?? "none"} />
      </div>
      <p className="panel-note">{status.message}</p>
      <SyncAuditPanel status={status} />
      {logLines && logLines.length > 0
        ? <RecorderLogStream lines={logLines} />
        : status.lastOutput
          ? <p className="process-output">{status.lastOutput}</p>
          : null}
    </section>
  );
}



export function LiveRecordPage({
  snapshot,
  busy,
  onConnect,
  onStart,
  onStop,
  onSetStartPose,
  onResetStartPose,
  onSceneReset,
  onOpenInReplay,
  onQueueTrajGen,
  onGoToProcessing,
  onClearActiveTask
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onConnect: (
    backend?: RecordingBackend,
    episodeTimeS?: number,
    fps?: number,
    numEpisodes?: number
  ) => void;
  onStart: () => void;
  onStop: (action: "save" | "discard" | "exit") => void;
  onSetStartPose: () => void;
  onResetStartPose: () => void;
  onSceneReset: (request: SceneResetRequest) => Promise<boolean>;
  onOpenInReplay: () => void;
  onQueueTrajGen: () => void;
  onGoToProcessing: () => void;
  onClearActiveTask: () => void;
}) {
  const showSavedBanner = snapshot.recording.savedEpisodes > 0;
  // Only the FR3 workstation has two robots behind one recorder; Thor's rig is singular and
  // must keep sending Connect with no backend at all.
  const workstationProfile = snapshot.deployment?.profile === "workstation";
  const supportsBackendChoice = workstationProfile;
  const supportsTrajectoryGeneration = !workstationProfile;
  const [sceneResetLandmarks, setSceneResetLandmarks] = useState<RolloutLandmarks>({});
  const [selectedBackend, setSelectedBackend] = useState<RecordingBackend>(
    snapshot.recording.backend ?? "real"
  );
  const [episodeTimeInput, setEpisodeTimeInput] = useState(() => String(snapshot.configSummary.episodeTimeS));
  const [fpsInput, setFpsInput] = useState(() => String(snapshot.configSummary.fps));
  // `numEpisodes` is `number | "unlimited"`: the handheld rigs run until the operator stops, the
  // FR3 recorder stops itself after `dataset.num_episodes`. Only a number is editable, so the
  // control below hides itself rather than offering an edit the recorder would ignore.
  const configuredNumEpisodes = snapshot.configSummary.numEpisodes;
  const supportsNumEpisodes = typeof configuredNumEpisodes === "number";
  const [numEpisodesInput, setNumEpisodesInput] = useState(() =>
    supportsNumEpisodes ? String(configuredNumEpisodes) : ""
  );
  const activeTask = snapshot.activeTaskId
    ? snapshot.tasks.find((t) => t.id === snapshot.activeTaskId) ?? null
    : null;
  const recorderConnected = ["connecting", "armed", "recording", "review", "saving", "discarding"].includes(
    snapshot.recording.state
  );
  // The backend keeps a per-session ring buffer (RecordingStatus.recentOutput)
  // and clears it when the operator clicks Connect, so we can render it
  // directly. The pre-PR6 approach of accumulating `lastOutput` lost any
  // line that didn't land at the top of a snapshot poll window.
  const logLines = snapshot.recording.recentOutput ?? [];
  const parsedEpisodeTimeS = Number(episodeTimeInput);
  const parsedFps = Number(fpsInput);
  const parsedNumEpisodes = Number(numEpisodesInput);
  const canUseEpisodeTimeInput = Number.isFinite(parsedEpisodeTimeS) && parsedEpisodeTimeS >= 1 && parsedEpisodeTimeS <= 600;
  const canUseFpsInput = Number.isInteger(parsedFps) && parsedFps >= 1 && parsedFps <= 120;
  // Bounds mirror _parse_num_episodes_override in gateway.py; keep them in step or the field
  // accepts a value Connect then rejects.
  const canUseNumEpisodesInput =
    !supportsNumEpisodes ||
    (Number.isInteger(parsedNumEpisodes) && parsedNumEpisodes >= 1 && parsedNumEpisodes <= 1000);
  const requestedEpisodeTimeS = canUseEpisodeTimeInput ? parsedEpisodeTimeS : snapshot.configSummary.episodeTimeS;
  const requestedFps = canUseFpsInput ? parsedFps : snapshot.configSummary.fps;
  const requestedNumEpisodes =
    supportsNumEpisodes && Number.isInteger(parsedNumEpisodes) ? parsedNumEpisodes : undefined;
  const requestedTargetFrames = Math.max(1, Math.round(requestedFps * requestedEpisodeTimeS));

  // Keyboard shortcuts for the record controls. This component is mounted only
  // while activePage === "live-record", so the window listener is naturally
  // scoped to this page. Each key mirrors the matching button's enabled gating
  // exactly (recordingControlAvailability), is suppressed while busy, ignores
  // modifier combos (so Ctrl/Cmd+S etc. stay with the browser), and never fires
  // while the operator is typing in an input/textarea/select.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.defaultPrevented || event.repeat || event.isComposing) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const el = document.activeElement as HTMLElement | null;
      const tag = el?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el?.isContentEditable) return;
      if (busy) return;
      const controls = recordingControlAvailability(snapshot.recording);
      const key = event.key.toLowerCase();
      if (
        key === "c" &&
        controls.canConnect &&
        canUseEpisodeTimeInput &&
        canUseFpsInput &&
        canUseNumEpisodesInput
      ) {
        event.preventDefault();
        onConnect(
          supportsBackendChoice ? selectedBackend : undefined,
          requestedEpisodeTimeS,
          requestedFps,
          requestedNumEpisodes
        );
      } else if (key === "e" && controls.canStartEpisode) {
        event.preventDefault();
        onStart();
      } else if (key === "s" && controls.canResolveEpisode) {
        event.preventDefault();
        onStop("save");
      } else if (key === "d" && controls.canResolveEpisode) {
        event.preventDefault();
        onStop("discard");
      } else if (event.key === "Escape" && controls.canExit) {
        event.preventDefault();
        onStop("exit");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, snapshot.recording, onConnect, onStart, onStop, supportsBackendChoice, selectedBackend, requestedEpisodeTimeS, requestedFps, requestedNumEpisodes, canUseEpisodeTimeInput, canUseFpsInput, canUseNumEpisodesInput]);

  // Once a session is live the backend is fixed by the running recorder process; showing the
  // operator's stale pick instead of the actual one would misreport what is being recorded.
  useEffect(() => {
    if (recorderConnected && snapshot.recording.backend) {
      setSelectedBackend(snapshot.recording.backend);
    }
  }, [recorderConnected, snapshot.recording.backend]);

  useEffect(() => {
    if (!workstationProfile) return;
    let cancelled = false;
    void (async () => {
      const payload = await api.fetchRolloutLandmarks(snapshot.recording.datasetRoot);
      if (!cancelled) setSceneResetLandmarks(payload);
    })();
    return () => {
      cancelled = true;
    };
  }, [workstationProfile, snapshot.recording.datasetRoot]);

  useEffect(() => {
    if (!recorderConnected) {
      setEpisodeTimeInput(String(snapshot.configSummary.episodeTimeS));
      setFpsInput(String(snapshot.configSummary.fps));
      setNumEpisodesInput(typeof configuredNumEpisodes === "number" ? String(configuredNumEpisodes) : "");
    }
  }, [recorderConnected, snapshot.configSummary.episodeTimeS, snapshot.configSummary.fps, configuredNumEpisodes]);

  const episodeDurationControl = (
    <div className="recording-session-controls">
      <label className={`episode-duration-control ${canUseEpisodeTimeInput ? "" : "episode-duration-invalid"}`}>
        <span>Episode seconds</span>
        <input
          type="number"
          min={1}
          max={600}
          step={1}
          disabled={busy || recorderConnected}
          value={episodeTimeInput}
          onChange={(event) => setEpisodeTimeInput(event.target.value)}
        />
        <small>{canUseEpisodeTimeInput ? `${requestedTargetFrames} frames` : "1-600s"}</small>
      </label>
      <label className={`episode-duration-control ${canUseFpsInput ? "" : "episode-duration-invalid"}`}>
        <span>Recording FPS</span>
        <input
          type="number"
          min={1}
          max={120}
          step={1}
          disabled={busy || recorderConnected}
          value={fpsInput}
          onChange={(event) => setFpsInput(event.target.value)}
        />
        <small>{canUseFpsInput ? `${requestedFps} Hz` : "1-120"}</small>
      </label>
      {supportsNumEpisodes ? (
        <label className={`episode-duration-control ${canUseNumEpisodesInput ? "" : "episode-duration-invalid"}`}>
          <span>Episodes</span>
          <input
            type="number"
            min={1}
            max={1000}
            step={1}
            disabled={busy || recorderConnected}
            value={numEpisodesInput}
            onChange={(event) => setNumEpisodesInput(event.target.value)}
          />
          <small>
            {canUseNumEpisodesInput
              ? `stops after ${requestedNumEpisodes ?? configuredNumEpisodes}`
              : "1-1000"}
          </small>
        </label>
      ) : null}
    </div>
  );

  const backendPicker = supportsBackendChoice ? (
    <div className="mujoco-mode-picker" role="group" aria-label="Recording backend">
      <button
        className={selectedBackend === "real" ? "active" : ""}
        disabled={busy || recorderConnected}
        onClick={() => setSelectedBackend("real")}
        type="button"
      >
        Real FR3
      </button>
      <button
        className={selectedBackend === "sim" ? "active" : ""}
        disabled={busy || recorderConnected}
        onClick={() => setSelectedBackend("sim")}
        type="button"
      >
        MuJoCo Sim
      </button>
    </div>
  ) : undefined;

  return (
    <div className="page-stack">
      <PageHeader
        title="Live Record"
        subtitle={supportsBackendChoice
          ? `FR3 SpaceMouse capture on the ${selectedBackend === "sim" ? "MuJoCo twin" : "real arm"}; both write the same dataset schema`
          : snapshot.configSummary.rigType === "gmsl2"
            ? `GMSL2 ${snapshot.devices.filter((d) => d.kind === "camera").length}-camera capture with${snapshot.configSummary.hardwareSync?.enabled ? "" : "out"} hardware sync`
            : "capture raw multi-camera handheld data; post-processing lives on the Processing page"}
      />
      {workstationProfile ? (
        <section className="panel capture-readiness-banner" aria-label="FR3 capture readiness">
          <div>
            <strong>Before collecting: lock camera exposure and white balance</strong>
            <span>Keep lighting, camera mounts, hole pose, and table background stable for this batch.</span>
          </div>
          <div className="capture-readiness-checks">
            <span>Exposure locked</span>
            <span>White balance locked</span>
            <span>Hole visible in EE and side views</span>
          </div>
        </section>
      ) : null}
      {activeTask && (
        <section className="panel task-binding-banner">
          <div className="panel-heading">
            <h2>Recording for task: {activeTask.name}</h2>
            <button disabled={busy || recorderConnected} onClick={onClearActiveTask}>Unbind</button>
          </div>
          <p className="panel-note">
            Episodes save into <strong>{activeTask.datasetRepoId}</strong> and count toward this task ({activeTask.completedEpisodes}/{activeTask.targetEpisodes}).
            {recorderConnected ? " Disconnect to unbind or switch tasks." : " Binding applies on the next Connect."}
          </p>
        </section>
      )}
      <div className="split-layout">
        <RecordingPanel
          status={snapshot.recording}
          config={snapshot.configSummary}
          busy={busy}
          onConnect={() =>
            onConnect(
              supportsBackendChoice ? selectedBackend : undefined,
              requestedEpisodeTimeS,
              requestedFps,
              requestedNumEpisodes
            )
          }
          onStart={onStart}
          onStop={onStop}
          onSetStartPose={onSetStartPose}
          onResetStartPose={onResetStartPose}
          logLines={logLines}
          backendPicker={backendPicker}
          episodeDurationControl={episodeDurationControl}
          episodeDurationValid={canUseEpisodeTimeInput && canUseFpsInput && canUseNumEpisodesInput}
          showStartPoseControl={workstationProfile}
        />
        <DeviceList devices={snapshot.devices} config={snapshot.configSummary} />
      </div>
      {workstationProfile ? (
        <SceneResetPanel
          title="Record scene reset"
          landmarks={sceneResetLandmarks}
          backgroundImageUrl={recorderConnected ? api.cameraSnapshotUrl("side") : undefined}
          backgroundLabel="side camera"
          busy={busy}
          disabled={!(["armed", "review"].includes(snapshot.recording.state))}
          disabledReason={
            ["armed", "review"].includes(snapshot.recording.state)
              ? ""
              : "Connect the FR3 recorder and wait between episodes before resetting the scene."
          }
          onReset={async (request) => {
            const ok = await onSceneReset(request);
            return { ok, error: ok ? undefined : "Scene reset command was rejected." };
          }}
        />
      ) : null}
      {showSavedBanner ? (
        <section className="panel saved-cta">
          <div className="panel-heading">
            <h2>Raw dataset saved</h2>
            <span>{snapshot.recording.savedEpisodes} episodes this session</span>
          </div>
          <p className="panel-note">{snapshot.recording.datasetRoot}</p>
          <div className="control-row">
            <button disabled={busy} onClick={onOpenInReplay}>Open in Replay</button>
            {supportsTrajectoryGeneration ? (
              <button disabled={busy} onClick={onQueueTrajGen}>Queue Traj Gen</button>
            ) : null}
            <button disabled={busy} onClick={onGoToProcessing}>Go to Processing</button>
          </div>
        </section>
      ) : null}
    </div>
  );
}

