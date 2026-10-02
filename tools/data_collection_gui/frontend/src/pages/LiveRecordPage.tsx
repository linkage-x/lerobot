import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GuiSnapshot } from "../api";
import type { BoxPreviewPayload, BoxCaliLog, BoxCaliLogLine, CollectionTask, ConfigSummary, DeviceStatus, EpisodeAnnotation, EventLogItem, ProcessingItem, ProcessingStatus, RecordedDataset, RecordingBackend, RecordingStatus, ReplayStatus, SubtaskSegment, TaskStatus, DatasetExportStatus, AnnotationOutcome, AnnotationQuality, ReviewStatus, TrackerMountSession, Fr3TeleopStatus } from "../types";
import { StatusDot, Metric, PageHeader, stateLabel, QualityOverview, processingStatusLabel, datasetNamePrefixes, taskDatasetBaseName, processingItemsForTask, taskNeedsQcExportConfirmation } from "../shared/ui";
import { hasActiveCalibrationCapture, recordingControlAvailability, recordingShortcutAction } from "./recordingControls";
export { recordingControlAvailability } from "./recordingControls";

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

function telemetryVector(telemetry: Record<string, unknown>, key: string): unknown[] {
  return Array.isArray(telemetry[key]) ? telemetry[key] as unknown[] : [];
}

function telemetryNumber(value: unknown, precision = 3): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(precision) : "—";
}

export function Fr3StatusPanel({ status }: { status: Fr3TeleopStatus }) {
  const telemetry = status.telemetry ?? {};
  const jointColumns = ["q", "dq", "tau_J", "tau_ext_hat_filtered"].map((key) => telemetryVector(telemetry, key));
  const tcp = telemetryVector(telemetry, "measured_tcp").length
    ? telemetryVector(telemetry, "measured_tcp") : telemetryVector(telemetry, "tcp");
  const wrench = telemetryVector(telemetry, "O_F_ext_hat_K");
  const dot = status.state === "running" ? "running" : status.state === "error" ? "error"
    : status.state === "idle" ? "idle" : "warning";
  return (
    <div className="fr3-status-panel">
      <div className="fr3-status-heading">
        <strong>FR3 SpaceMouse teleoperation</strong>
        <span className="state-pill"><StatusDot state={dot} />{stateLabel(status.state)}</span>
      </div>
      <p className={status.state === "error" ? "fr3-fault-message" : "panel-note"} role={status.state === "error" ? "alert" : "status"}>
        {status.message}
      </p>
      {status.state === "error" && (
        <p className="fr3-fault-message">{status.pid != null
          ? "FR3 worker shutdown is not confirmed. Stop the robot using its hardware controls and resolve the remaining worker process before retrying F."
          : "Fix the reported problem and clear any robot fault. Release the SpaceMouse, then press F to move to start and resume."}</p>
      )}
      <details className="fr3-telemetry">
        <summary>Measured joints, torque and end effector</summary>
        {status.state !== "running" && <p className="panel-note">Measurements are available while FR3 teleoperation is running.</p>}
        <div className="fr3-table-scroll">
          <table>
            <thead><tr><th>Joint</th><th>q (rad)</th><th>dq (rad/s)</th><th>Torque (Nm)</th><th>External torque (Nm)</th></tr></thead>
            <tbody>{Array.from({ length: 7 }, (_, i) => (
              <tr key={i}><th>{i + 1}</th>{jointColumns.map((values, column) => <td key={column}>{telemetryNumber(values[i])}</td>)}</tr>
            ))}</tbody>
          </table>
        </div>
        <div className="fr3-vector-grid">
          <Metric label="Measured TCP xyz (m)" value={[0, 1, 2].map((i) => telemetryNumber(tcp[i], 4)).join(", ")} />
          <Metric label="Measured TCP rotation vector (rad)" value={[3, 4, 5].map((i) => telemetryNumber(tcp[i], 4)).join(", ")} />
          <Metric label="External force xyz (N)" value={[0, 1, 2].map((i) => telemetryNumber(wrench[i])).join(", ")} />
          <Metric label="External torque xyz (Nm)" value={[3, 4, 5].map((i) => telemetryNumber(wrench[i])).join(", ")} />
          <Metric label="FCI command success rate" value={telemetryNumber(telemetry.control_command_success_rate, 5)} />
          <Metric label="Measured gripper opening (m)" value={telemetryNumber(telemetry.gripper_measured_m, 4)} />
        </div>
      </details>
    </div>
  );
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
  logLines,
  backendPicker,
  laserTrackerToggle,
  mountSession,
  fr3Teleop,
  onStartFr3,
  onStopFr3,
  calibrationCapture = false,
}: {
  status: RecordingStatus;
  config: ConfigSummary;
  busy: boolean;
  onConnect: () => void;
  onStart: () => void;
  onStop: (action: "save" | "discard" | "exit") => void;
  logLines?: string[];
  backendPicker?: React.ReactNode;
  laserTrackerToggle?: React.ReactNode;
  /** A calibration capture holding the recorder; see TrackerMountSession. */
  mountSession?: TrackerMountSession;
  fr3Teleop?: Fr3TeleopStatus;
  onStartFr3?: () => void;
  onStopFr3?: () => void;
  calibrationCapture?: boolean;
}) {
  const progress = Math.round((status.frameIndex / Math.max(status.targetFrames, 1)) * 100);
  // Only while the tracker is actually switched on for this session: an episode
  // recorded with a blind tracker looks complete and measures nothing, and the
  // tracker re-homes on a timer, so this clears itself once the SMR is in place.
  // Ready also means homed: a locked beam without a Home measures every distance
  // against a stale reference, so "not homed" is named on its own.
  const trackerBlocking = Boolean(status.laserTracker) && !status.laserTrackerReady;
  const trackerNotHomed = Boolean(status.laserTracker) && !status.laserTrackerHomed;
  const trackerBeamBroken = Boolean(status.laserTracker) && Boolean(status.laserTrackerBeamBroken);
  const trackerBlockReason = trackerNotHomed
    ? `激光跟踪仪还没 Home 成功，不能开录：把 SMR 放进 home 窝等待自动 Home（${status.laserTrackerDetail || "等待中"}）`
    : trackerBeamBroken
      ? "激光跟踪仪断过光，当前距离不是绝对的，不能开录：请把 SMR 放回 home 窝，会自动重新 Home"
      : status.laserTrackerDetail || "激光跟踪仪尚未锁定 SMR";
  // The gateway refuses StartEpisode while a mount capture owns the recorder,
  // because a task episode there does two invisible kinds of damage: it clears
  // the calibration redirect and lands a stationary rig in the training set, and
  // it puts motion into the middle of the tracker stream the mount fit will cut
  // parked poses out of. Shown here so the refusal is not a surprise.
  const mountHeld = Boolean(mountSession?.active) && mountSession?.stage === "capture";
  const mountBlockReason = mountHeld
    ? `跟踪仪站位采集 ${mountSession?.sessionName ?? ""} 正在占用录制器`
    : "";
  const { isConnected, canConnect, canStartEpisode, canResolveEpisode, canExit, canStartFr3, canStopFr3 } =
    recordingControlAvailability(status, fr3Teleop, mountSession, busy, calibrationCapture);
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
      {fr3Teleop?.enabled && <Fr3StatusPanel status={fr3Teleop} />}
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
      {laserTrackerToggle}
      {status.laserTracker && isConnected && (
        <p
          className="tracker-home-state"
          data-homed={status.laserTrackerHomed && !status.laserTrackerBeamBroken ? "yes" : "no"}
        >
          跟踪仪 Home：
          {!status.laserTrackerHomed
            ? "❌ 未 Home"
            : status.laserTrackerBeamBroken
              ? "⚠️ 断过光，请把 SMR 放回窝里重新 Home（放回后自动 Home）"
              : "✅ 已 Home（有绝对距离）"}
        </p>
      )}
      {trackerBlocking && (
        <p className="tracker-wait-banner">
          ⏳ {trackerBlockReason}
        </p>
      )}
      {mountHeld && (
        <p className="tracker-wait-banner">
          🔒 {mountBlockReason}（已落盘 {mountSession?.dwellsOnDisk ?? 0} 段）。
          要录任务数据，先到「标定」页结束这次站位采集——已经录下的停驻段不会被删。
        </p>
      )}
      {calibrationCapture && !mountHeld && (
        <p className="tracker-wait-banner">A calibration capture owns this recorder. Finish it on the Calibration page before starting FR3 or recording a task episode.</p>
      )}
      {fr3Teleop?.enabled && fr3Teleop.state !== "running" && status.state === "armed" && (
        <p className="tracker-wait-banner">Press F to move FR3 to the configured start pose and enable SpaceMouse motion. Press E afterwards to record.</p>
      )}
      <div className="progress">
        <div className="progress-bar" style={{ width: `${progress}%` }} />
      </div>
      <div className="control-row">
        <button disabled={!canConnect} onClick={onConnect} title="Shortcut: C">Connect <kbd>C</kbd></button>
        {fr3Teleop?.enabled && (
          <button disabled={!canStartFr3} onClick={onStartFr3} title="F: move FR3 to start, then enable SpaceMouse motion">Start FR3 <kbd>F</kbd></button>
        )}
        <button
          disabled={!canStartEpisode}
          onClick={onStart}
          title={calibrationCapture ? "Finish the active calibration capture first" : mountHeld ? mountBlockReason : trackerBlocking ? trackerBlockReason : "Shortcut: E"}
        >
          StartEpisode <kbd>E</kbd>
        </button>
        <button disabled={!canResolveEpisode} onClick={() => onStop("save")} title="Shortcut: S">Save <kbd>S</kbd></button>
        <button disabled={!canResolveEpisode} onClick={() => onStop("discard")} title="Shortcut: D">Discard <kbd>D</kbd></button>
        {fr3Teleop?.enabled && <button disabled={!canStopFr3} onClick={onStopFr3} title="Stop FR3 motion; discard any active episode">Stop FR3</button>}
        <button disabled={!canExit} onClick={() => onStop("exit")} title="Shortcut: Esc">Exit <kbd>Esc</kbd></button>
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
  onStartFr3,
  onStopFr3,
  onOpenInReplay,
  onQueueTrajGen,
  onGoToProcessing,
  onClearActiveTask
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onConnect: (backend?: RecordingBackend, laserTracker?: boolean) => void;
  onStart: () => void;
  onStop: (action: "save" | "discard" | "exit") => void;
  onStartFr3: () => void;
  onStopFr3: () => void;
  onOpenInReplay: () => void;
  onQueueTrajGen: () => void;
  onGoToProcessing: () => void;
  onClearActiveTask: () => void;
}) {
  const showSavedBanner = snapshot.recording.savedEpisodes > 0;
  // The row exists whenever the rig *can* carry the tracker; the toggle decides
  // whether this session does. Separate concepts: on the days the instrument is
  // booked by someone else the row is still there and the answer is still no.
  const hasLaserTracker = snapshot.devices.some((d) => d.kind === "laser_tracker");
  const [laserTracker, setLaserTracker] = useState(false);
  // Only the FR3 workstation has two robots behind one recorder; Thor's rig is singular and
  // must keep sending Connect with no backend at all.
  const supportsBackendChoice = snapshot.deployment?.profile === "workstation";
  const [selectedBackend, setSelectedBackend] = useState<RecordingBackend>(
    snapshot.recording.backend ?? "real"
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

  // Keyboard shortcuts for the record controls. This component is mounted only
  // while activePage === "live-record", so the window listener is naturally
  // scoped to this page. Each key mirrors the matching button's enabled gating
  // exactly (recordingControlAvailability), is suppressed while busy, ignores
  // modifier combos (so Ctrl/Cmd+S etc. stay with the browser), and never fires
  // while the operator is typing in an input/textarea/select.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const el = document.activeElement as HTMLElement | null;
      const tag = el?.tagName;
      const editable = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || Boolean(el?.isContentEditable);
      const controls = recordingControlAvailability(snapshot.recording, snapshot.fr3Teleop, snapshot.trackerMountSession, busy, hasActiveCalibrationCapture(snapshot));
      const action = recordingShortcutAction(event, controls, editable);
      if (!action) return;
      event.preventDefault();
      if (action === "connect") {
        onConnect(
          supportsBackendChoice ? selectedBackend : undefined,
          hasLaserTracker ? laserTracker : undefined,
        );
      } else if (action === "startFr3") {
        onStartFr3();
      } else if (action === "startEpisode") {
        onStart();
      } else if (action === "save") {
        onStop("save");
      } else if (action === "discard") {
        onStop("discard");
      } else if (action === "exit") {
        onStop("exit");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, snapshot.recording, snapshot.fr3Teleop, snapshot.trackerMountSession, snapshot.calibrationSession, snapshot.markerTcp, onConnect, onStart, onStop, onStartFr3, supportsBackendChoice, selectedBackend, hasLaserTracker, laserTracker]);

  // Once a session is live the backend is fixed by the running recorder process; showing the
  // operator's stale pick instead of the actual one would misreport what is being recorded.
  useEffect(() => {
    if (recorderConnected && snapshot.recording.backend) {
      setSelectedBackend(snapshot.recording.backend);
    }
  }, [recorderConnected, snapshot.recording.backend]);

  // Same reasoning as the backend picker: once a session is live the choice is
  // fixed by the running recorder, and showing the operator's stale pick would
  // misreport what is being recorded.
  useEffect(() => {
    if (recorderConnected) {
      setLaserTracker(Boolean(snapshot.recording.laserTracker));
    }
  }, [recorderConnected, snapshot.recording.laserTracker]);

  const laserTrackerToggle = hasLaserTracker ? (
    <label className="laser-tracker-toggle">
      <input
        type="checkbox"
        checked={laserTracker}
        disabled={busy || recorderConnected}
        onChange={(event) => setLaserTracker(event.target.checked)}
      />
      <span>激光跟踪仪</span>
      <small>
        {recorderConnected
          ? snapshot.recording.laserTrackerDevice
            || snapshot.recording.laserTrackerDetail
            || (laserTracker ? "本次会话已启用" : "本次会话未启用")
          : "共享设备，同一时刻只能一个客户端；连不上不会阻塞录制"}
      </small>
    </label>
  ) : undefined;

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
        subtitle={snapshot.fr3Teleop?.enabled
          ? "C connects Sengyun cameras and BOX sensors · F moves FR3 to start and enables SpaceMouse · E records · S saves · D discards"
          : supportsBackendChoice
          ? `FR3 SpaceMouse capture on the ${selectedBackend === "sim" ? "MuJoCo twin" : "real arm"}; both write the same dataset schema`
          : snapshot.configSummary.rigType === "gmsl2"
            ? `GMSL2 ${snapshot.devices.filter((d) => d.kind === "camera").length}-camera capture with${snapshot.configSummary.hardwareSync?.enabled ? "" : "out"} hardware sync`
            : "capture raw multi-camera handheld data; post-processing lives on the Processing page"}
      />
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
              hasLaserTracker ? laserTracker : undefined,
            )
          }
          laserTrackerToggle={laserTrackerToggle}
          onStart={onStart}
          onStop={onStop}
          fr3Teleop={snapshot.fr3Teleop}
          onStartFr3={onStartFr3}
          onStopFr3={onStopFr3}
          calibrationCapture={hasActiveCalibrationCapture(snapshot)}
          logLines={logLines}
          backendPicker={backendPicker}
          mountSession={snapshot.trackerMountSession}
        />
        <DeviceList devices={snapshot.devices} config={snapshot.configSummary} />
      </div>
      {showSavedBanner ? (
        <section className="panel saved-cta">
          <div className="panel-heading">
            <h2>Raw dataset saved</h2>
            <span>{snapshot.recording.savedEpisodes} episodes this session</span>
          </div>
          <p className="panel-note">{snapshot.recording.datasetRoot}</p>
          <div className="control-row">
            <button disabled={busy} onClick={onOpenInReplay}>Open in Replay</button>
            <button disabled={busy} onClick={onQueueTrajGen}>Queue Traj Gen</button>
            <button disabled={busy} onClick={onGoToProcessing}>Go to Processing</button>
          </div>
        </section>
      ) : null}
    </div>
  );
}
