import { useEffect, useMemo, useRef, useState } from "react";
import { DataCollectionGuiApi, type GuiSnapshot } from "./api";
import { ReplayInspector } from "./ReplayInspector";
import type {
  BoxPreviewPayload,
  CollectionTask,
  ConfigSummary,
  DeviceStatus,
  EpisodeAnnotation,
  EventLogItem,
  ProcessingItem,
  ProcessingStatus,
  RecordedDataset,
  RecordingStatus,
  ReplayStatus,
  SubtaskSegment,
  TaskStatus
} from "./types";
import "./styles.css";

const api = new DataCollectionGuiApi();

function stateLabel(state: string) {
  return state.replace("_", " ");
}

function datasetNamePrefixes(name: string): Set<string> {
  const prefixes = new Set([name]);
  const match = name.match(/^(?<base>.+)_\d{8}_\d{6}(?:_\d{2})?$/);
  if (match?.groups?.base) {
    prefixes.add(match.groups.base);
  }
  return prefixes;
}

function taskDatasetBaseName(task: CollectionTask): string {
  return task.datasetRepoId.split("/").pop()?.trim() ?? "";
}

function processingItemsForTask(task: CollectionTask, processing: ProcessingItem[]): ProcessingItem[] {
  const baseName = taskDatasetBaseName(task);
  if (!baseName) {
    return [];
  }
  return processing.filter((item) => datasetNamePrefixes(item.name).has(baseName));
}

function taskNeedsQcExportConfirmation(task: CollectionTask, processing: ProcessingItem[]) {
  const taskItems = processingItemsForTask(task, processing);
  const notQcPassed = taskItems.filter((item) => item.status !== "qc_pass");
  const processingEpisodes = taskItems.reduce((total, item) => total + item.totalEpisodes, 0);
  const missingEpisodeCount = Math.max(task.completedEpisodes - processingEpisodes, 0);
  return { taskItems, notQcPassed, missingEpisodeCount };
}

function StatusDot({ state }: { state: string }) {
  return <span className={`status-dot status-${state}`} aria-hidden="true" />;
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DeviceList({ devices, config }: { devices: DeviceStatus[]; config: ConfigSummary }) {
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

function HardwareSyncBadge({ config }: { config: ConfigSummary }) {
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

function CameraEncodingInfo({ config }: { config: ConfigSummary }) {
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

function RecorderLogStream({ lines }: { lines: string[] }) {
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
function recordingControlAvailability(status: RecordingStatus) {
  const isConnected =
    status.pid != null ||
    ["connecting", "armed", "recording", "review", "saving", "discarding"].includes(status.state);
  return {
    isConnected,
    canConnect: !isConnected,
    canStartEpisode: status.state === "armed",
    canResolveEpisode: status.state === "recording" || status.state === "review",
    canExit: isConnected,
  };
}

function RecordingPanel({
  status,
  config,
  busy,
  onConnect,
  onStart,
  onStop,
  logLines
}: {
  status: RecordingStatus;
  config: ConfigSummary;
  busy: boolean;
  onConnect: () => void;
  onStart: () => void;
  onStop: (action: "save" | "discard" | "exit") => void;
  logLines?: string[];
}) {
  const progress = Math.round((status.frameIndex / Math.max(status.targetFrames, 1)) * 100);
  const { isConnected, canStartEpisode, canResolveEpisode, canExit } =
    recordingControlAvailability(status);
  const isGmsl = config.rigType === "gmsl2";
  const panelTitle = isGmsl ? "GMSL2 Record" : "Handheld Record";

  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>{panelTitle}</h2>
        <span className="state-pill">
          <StatusDot state={status.state} />
          {stateLabel(status.state)}
        </span>
      </div>
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
        <button disabled={busy || isConnected} onClick={onConnect} title="Shortcut: C">Connect <kbd>C</kbd></button>
        <button disabled={busy || !canStartEpisode} onClick={onStart} title="Shortcut: E">StartEpisode <kbd>E</kbd></button>
        <button disabled={busy || !canResolveEpisode} onClick={() => onStop("save")} title="Shortcut: S">Save <kbd>S</kbd></button>
        <button disabled={busy || !canResolveEpisode} onClick={() => onStop("discard")} title="Shortcut: D">Discard <kbd>D</kbd></button>
        <button disabled={busy || !canExit} onClick={() => onStop("exit")} title="Shortcut: Esc">Exit <kbd>Esc</kbd></button>
      </div>
      <div className="summary-grid">
        <Metric label="Frame" value={`${status.frameIndex}/${status.targetFrames}`} />
        <Metric label="Queue" value={status.queueDepth} />
        <Metric label="Saved" value={status.savedEpisodes} />
        <Metric label="PID" value={status.pid ?? "none"} />
      </div>
      <p className="panel-note">{status.message}</p>
      {logLines && logLines.length > 0
        ? <RecorderLogStream lines={logLines} />
        : status.lastOutput
          ? <p className="process-output">{status.lastOutput}</p>
          : null}
    </section>
  );
}


function ReplayPanel({
  status,
  busy,
  onPreflight,
  onReplay,
  onMujocoReplay,
  onAbort
}: {
  status: ReplayStatus;
  busy: boolean;
  onPreflight: () => void;
  onReplay: (realRobot: boolean) => void;
  onMujocoReplay: () => void;
  onAbort: () => void;
}) {
  const isActive = status.state === "dry_run" || status.state === "sim_replay" || status.state === "replaying";
  const canReplayData = status.dataStatus === "loaded" && (status.recordedFrames ?? status.totalFrames) > 0;
  const validation = status.mujocoValidation;
  const mujocoPassed = validation?.status === "passed" && validation.isCurrentForSelection === true;
  const validationLabel =
    validation?.status === "passed"
      ? "passed"
      : validation?.status === "failed"
        ? "failed"
        : validation?.status === "running"
          ? "running"
          : "recommended";
  const validationMetric =
    validation?.maxPositionErrorMm != null && validation?.maxRotationErrorDeg != null
      ? `Max ${validation.maxPositionErrorMm.toFixed(2)}mm / ${validation.maxRotationErrorDeg.toFixed(2)}deg; limits ${validation.maxPositionThresholdMm.toFixed(2)}mm / ${validation.maxRotationThresholdDeg.toFixed(2)}deg`
      : "No completed MuJoCo metrics yet";
  const validationGuidance = mujocoPassed
    ? "MuJoCo replay is current for this episode."
    : "Strongly recommended before Preflight/Dry Run; required before Real Robot.";
  const realRobotDisabled = busy || isActive || status.safety !== "ready" || !mujocoPassed;
  return (
    <section className="panel replay-panel">
      <div className="panel-heading">
        <h2>Replay Controls</h2>
        <span className="state-pill">
          <StatusDot state={status.state} />
          {stateLabel(status.state)}
        </span>
      </div>
      <div className="control-row">
        <button disabled={busy || isActive || !canReplayData} onClick={onPreflight}>Preflight</button>
        <button disabled={busy || isActive || status.safety !== "ready"} onClick={() => onReplay(false)}>Dry Run</button>
        <button disabled={busy || isActive || !canReplayData} onClick={onMujocoReplay}>MuJoCo</button>
        <button className="danger" disabled={realRobotDisabled} onClick={() => onReplay(true)}>Real Robot</button>
        <button disabled={busy || !isActive} onClick={onAbort}>Abort</button>
      </div>
      <p className="panel-note">Safety {status.safety} · {status.message}</p>
      <p className={`validation-note validation-${validationLabel}`}>
        MuJoCo replay {validationLabel}: {validationGuidance} {validation?.message ?? "Run MuJoCo replay before real-robot replay."} · {validationMetric}
      </p>
      {status.lastOutput ? <p className="process-output">{status.lastOutput}</p> : null}
      {status.diagnostics?.length ? (
        <div className="diagnostic-box">
          {status.diagnostics.slice(0, 2).map((diagnostic, index) => (
            <p key={index}>{diagnostic}</p>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function EventLog({ events }: { events: EventLogItem[] }) {
  return (
    <section className="panel event-panel">
      <div className="panel-heading">
        <h2>Events</h2>
        <span>audit stream</span>
      </div>
      <div className="event-list">
        {events.map((event) => (
          <div className={`event-row level-${event.level}`} key={event.id}>
            <time>{event.time}</time>
            <span>{event.level}</span>
            <p>{event.message}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function RecordedDatasetList({
  datasets,
  activePath,
  busy,
  onSelect
}: {
  datasets: RecordedDataset[];
  activePath: string;
  busy: boolean;
  onSelect: (path: string) => void;
}) {
  const latest = datasets[0];
  return (
    <section className="panel dataset-list-panel">
      <div className="panel-heading">
        <h2>Recorded Datasets</h2>
        <span>{datasets.length} found</span>
      </div>
      {latest ? (
        <div className="latest-dataset">
          <span>Default replay target</span>
          <strong>{latest.name}</strong>
          <small>{latest.path}</small>
        </div>
      ) : (
        <div className="empty-dataset-list">No recorded datasets found under the configured dataset root.</div>
      )}
      <div className="dataset-list">
        {datasets.map((dataset) => {
          const isActive = dataset.path === activePath || (!activePath && dataset.isLatest);
          return (
            <button
              className={isActive ? "dataset-row active" : "dataset-row"}
              disabled={busy}
              key={dataset.path}
              onClick={() => onSelect(dataset.path)}
            >
              <div>
                <div className="row-title">
                  <StatusDot state={dataset.dataStatus === "loaded" ? "running" : "warning"} />
                  <strong>{dataset.name}</strong>
                  {dataset.isLatest ? <em>latest</em> : null}
                </div>
                <p>{dataset.path}</p>
              </div>
              <div className="dataset-stats">
                <span>{dataset.totalEpisodes} episodes</span>
                <span>{dataset.totalFrames} frames</span>
                <small>{dataset.updatedAt || "unknown time"}</small>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}

type PageId =
  | "live-record"
  | "dataset-processing"
  | "episode-replay"
  | "dataset-export"
  | "dashboard"
  | "qc-report"
  | "model-evaluation"
  | "device-manager"
  | "task-library"
  | "annotation-audit";

type PageMeta = { id: PageId; label: string; kind: "mvp" | "deferred" };

const mvpPages: PageMeta[] = [
  { id: "live-record", label: "Live Record", kind: "mvp" },
  { id: "dataset-processing", label: "Dataset Processing", kind: "mvp" },
  { id: "episode-replay", label: "Episode Replay", kind: "mvp" },
  { id: "dataset-export", label: "Dataset Export", kind: "mvp" },
  { id: "task-library", label: "Task Library", kind: "mvp" },
  { id: "device-manager", label: "Device Manager", kind: "mvp" }
];

const deferredPages: PageMeta[] = [
  { id: "dashboard", label: "Dashboard", kind: "deferred" },
  { id: "qc-report", label: "QC Report", kind: "deferred" },
  { id: "model-evaluation", label: "Model Evaluation", kind: "deferred" },
  { id: "annotation-audit", label: "Annotation & Audit", kind: "deferred" }
];

const pages: PageMeta[] = [...mvpPages, ...deferredPages];

function pageFromHash(): PageId {
  const hash = window.location.hash.replace(/^#\/?/, "") as PageId;
  return pages.some((page) => page.id === hash) ? hash : "live-record";
}

function PageHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="page-header">
      <div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
    </div>
  );
}

function QualityOverview({ snapshot }: { snapshot: GuiSnapshot }) {
  const points = snapshot.trajectory;
  const maxSkew = Math.max(0, ...points.map((point) => point.skewMs));
  const eventCount = points.filter((point) => point.event).length;
  const runningDevices = snapshot.devices.filter((device) => device.state === "running").length;
  const score = Math.max(
    0,
    100 -
      eventCount * 7 -
      (maxSkew > 50 ? 12 : 0) -
      (snapshot.replay.dataStatus === "loaded" ? 0 : 10) -
      (snapshot.configSummary.softSync ? 0 : 4)
  );

  const issues = [
    snapshot.replay.dataStatus === "loaded" ? null : "recorded trajectory is not loaded yet",
    eventCount > 0 ? `${eventCount} timeline markers need review` : null,
    maxSkew > 50 ? `timestamp skew max ${maxSkew.toFixed(1)} ms` : null,
    snapshot.configSummary.softSync ? null : "soft sync is disabled"
  ].filter(Boolean);

  return (
    <section className="panel quality-score-panel">
      <div className="panel-heading">
        <h2>Episode Quality Score</h2>
        <span>auto QC</span>
      </div>
      <div className="quality-score">
        <strong>{score}</strong>
        <span>/ 100</span>
      </div>
      <div className="summary-grid">
        <Metric label="Camera fps" value={`${snapshot.configSummary.fps} target`} />
        <Metric label="Running devices" value={`${runningDevices}/${snapshot.devices.length}`} />
        <Metric label="Frame drop markers" value={eventCount} />
        <Metric label="Timestamp skew" value={`${maxSkew.toFixed(1)} ms`} />
        <Metric label="Episode completeness" value={`${snapshot.recording.frameIndex}/${snapshot.recording.targetFrames}`} />
        <Metric label="Schema check" value={snapshot.replay.dataStatus === "loaded" ? "loaded" : "pending"} />
      </div>
      <div className="issue-list">
        {(issues.length ? issues : ["no blocking QC issues in the current snapshot"]).map((issue) => (
          <p key={issue}>{issue}</p>
        ))}
      </div>
    </section>
  );
}

function CalibrationPanel({
  status,
  busy,
  onRun
}: {
  status: GuiSnapshot["calibration"];
  busy: boolean;
  onRun: () => void;
}) {
  const cameraCount = status.cameras.length;
  return (
    <section className="panel calibration-panel">
      <div className="panel-heading">
        <h2>Multi-Camera Calibration</h2>
        <span className="state-pill">
          <StatusDot state={status.state === "complete" ? "running" : status.state === "failed" ? "error" : status.state === "running" ? "warning" : "idle"} />
          {stateLabel(status.state)}
        </span>
      </div>
      <div className="control-row">
        <button disabled={busy || status.state === "running"} onClick={onRun}>
          {status.state === "complete" || status.state === "failed" ? "Re-run Calibration" : "Run Calibration"}
        </button>
        <span className="calibration-pattern">pattern: {status.pattern}</span>
      </div>
      <p className="panel-note">{status.message}</p>
      {cameraCount > 0 ? (
        <div className="check-table calibration-table">
          {status.cameras.map((camera) => (
            <div className="check-row" key={camera.id}>
              <strong>{camera.id}</strong>
              <span>repro {camera.reprojectionMm.toFixed(3)} mm · baseline {camera.baselineMm.toFixed(1)} mm</span>
              <em>{camera.status}</em>
            </div>
          ))}
        </div>
      ) : null}
      <div className="summary-grid">
        <Metric label="Last run" value={status.lastRunAt || "—"} />
        <Metric label="Output" value={status.outputPath || "—"} />
      </div>
    </section>
  );
}

function LiveRecordPage({
  snapshot,
  busy,
  onConnect,
  onStart,
  onStop,
  onOpenInReplay,
  onQueueTrajGen,
  onGoToProcessing,
  onRunCalibration,
  onClearActiveTask
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onConnect: () => void;
  onStart: () => void;
  onStop: (action: "save" | "discard" | "exit") => void;
  onOpenInReplay: () => void;
  onQueueTrajGen: () => void;
  onGoToProcessing: () => void;
  onRunCalibration: () => void;
  onClearActiveTask: () => void;
}) {
  const showSavedBanner = snapshot.recording.savedEpisodes > 0;
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
      if (event.defaultPrevented || event.repeat || event.isComposing) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const el = document.activeElement as HTMLElement | null;
      const tag = el?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el?.isContentEditable) return;
      if (busy) return;
      const controls = recordingControlAvailability(snapshot.recording);
      const key = event.key.toLowerCase();
      if (key === "c" && controls.canConnect) {
        event.preventDefault();
        onConnect();
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
  }, [busy, snapshot.recording, onConnect, onStart, onStop]);

  return (
    <div className="page-stack">
      <PageHeader
        title="Live Record"
        subtitle={snapshot.configSummary.rigType === "gmsl2"
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
        <RecordingPanel status={snapshot.recording} config={snapshot.configSummary} busy={busy} onConnect={onConnect} onStart={onStart} onStop={onStop} logLines={logLines} />
        <DeviceList devices={snapshot.devices} config={snapshot.configSummary} />
      </div>
      <CalibrationPanel status={snapshot.calibration} busy={busy} onRun={onRunCalibration} />
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

function ReplayReadinessCard({
  status,
  processing,
  busy,
  onGenerate,
  onOpenProcessing
}: {
  status: ReplayStatus;
  processing: ProcessingItem | null;
  busy: boolean;
  onGenerate: () => void;
  onOpenProcessing: () => void;
}) {
  const trajectoryReady =
    status.dataStatus === "loaded" &&
    (status.trajectoryKind === "pose" || status.trajectoryKind === "gripper_width") &&
    (!processing || processing.status === "qc_pass" || processing.status === "pose_ready");
  const qcLabel = !processing ? "—" : processing.status === "qc_pass" ? "Pass" : processing.status === "qc_failed" ? "Fail" : "Pending";
  const validFrames = processing?.validFramesPct != null ? `${processing.validFramesPct}%` : "—";

  if (!trajectoryReady) {
    return (
      <section className="panel readiness-card readiness-missing">
        <div className="panel-heading">
          <h2>Trajectory: Missing</h2>
          <span>EE trajectory not generated</span>
        </div>
        <p className="panel-note">Generate EE trajectory before replaying this dataset. Replay is disabled until the trajectory is ready.</p>
        <div className="control-row">
          <button disabled={busy || !processing} onClick={onGenerate}>Generate EE Trajectory</button>
          <button disabled={busy} onClick={onOpenProcessing}>Open Processing</button>
        </div>
      </section>
    );
  }

  return (
    <section className="panel readiness-card readiness-ready">
      <div className="panel-heading">
        <h2>Trajectory: Ready</h2>
        <span>{processing?.trajectoryVersion ?? "active"}</span>
      </div>
      <div className="summary-grid">
        <Metric label="QC" value={qcLabel} />
        <Metric label="Frame" value="base_link" />
        <Metric label="Valid frames" value={validFrames} />
        <Metric label="Source" value={status.trajectoryKind ?? "pose"} />
      </div>
    </section>
  );
}

function SubtaskSegmentEditor({
  segments,
  totalFrames,
  onChange
}: {
  segments: SubtaskSegment[];
  totalFrames: number;
  onChange: (segments: SubtaskSegment[]) => void;
}) {
  const colors = ["#2563eb", "#0d9488", "#d97706", "#dc2626", "#7c3aed", "#059669"];

  const addSegment = () => {
    const lastEnd = segments.length > 0 ? segments[segments.length - 1].endFrame : 0;
    const newSeg: SubtaskSegment = {
      id: `seg-${Date.now()}`,
      startFrame: lastEnd,
      endFrame: Math.max(lastEnd, totalFrames - 1),
      description: ""
    };
    onChange([...segments, newSeg]);
  };

  const updateSegment = (id: string, patch: Partial<SubtaskSegment>) => {
    onChange(segments.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  };

  const removeSegment = (id: string) => {
    onChange(segments.filter((s) => s.id !== id));
  };

  return (
    <section className="panel segment-panel">
      <div className="panel-heading">
        <h2>Subtask Segments</h2>
        <span>{segments.length} segments</span>
      </div>
      {totalFrames > 0 && segments.length > 0 && (
        <div className="segment-timeline">
          {segments.map((seg, i) => {
            const left = (seg.startFrame / Math.max(totalFrames - 1, 1)) * 100;
            const width = ((seg.endFrame - seg.startFrame) / Math.max(totalFrames - 1, 1)) * 100;
            return (
              <div
                key={seg.id}
                className="segment-bar"
                title={seg.description || `Segment ${i + 1}`}
                style={{
                  left: `${left}%`,
                  width: `${Math.max(width, 0.5)}%`,
                  backgroundColor: colors[i % colors.length]
                }}
              />
            );
          })}
        </div>
      )}
      <div className="segment-list">
        {segments.map((seg, i) => (
          <div className="segment-row" key={seg.id}>
            <span className="segment-index" style={{ color: colors[i % colors.length] }}>#{i + 1}</span>
            <input
              type="number"
              min={0}
              max={totalFrames - 1}
              value={seg.startFrame}
              onChange={(e) => updateSegment(seg.id, { startFrame: Math.max(0, Number(e.target.value)) })}
              aria-label="Start frame"
              className="segment-frame-input"
            />
            <span>-</span>
            <input
              type="number"
              min={0}
              max={totalFrames - 1}
              value={seg.endFrame}
              onChange={(e) => updateSegment(seg.id, { endFrame: Math.max(0, Number(e.target.value)) })}
              aria-label="End frame"
              className="segment-frame-input"
            />
            <input
              value={seg.description}
              onChange={(e) => updateSegment(seg.id, { description: e.target.value })}
              placeholder="Subtask description"
              className="segment-desc-input"
            />
            <button onClick={() => removeSegment(seg.id)} className="segment-remove">x</button>
          </div>
        ))}
      </div>
      <div className="control-row">
        <button onClick={addSegment}>Add Segment</button>
      </div>
    </section>
  );
}

function EpisodeAnnotationPanel({
  annotation,
  datasetPath,
  totalFrames,
  busy,
  onSave
}: {
  annotation: EpisodeAnnotation;
  datasetPath: string;
  totalFrames: number;
  busy: boolean;
  onSave: (annotation: EpisodeAnnotation) => void;
}) {
  const [draft, setDraft] = useState<EpisodeAnnotation>(annotation);
  const [tagsText, setTagsText] = useState(annotation.tags.join(", "));
  const annotationIdentityRef = useRef(`${annotation.datasetRoot}:${annotation.episode}:${annotation.updatedAt}:${annotation.source}`);

  useEffect(() => {
    const nextIdentity = `${annotation.datasetRoot}:${annotation.episode}:${annotation.updatedAt}:${annotation.source}`;
    if (annotationIdentityRef.current !== nextIdentity) {
      annotationIdentityRef.current = nextIdentity;
      setDraft(annotation);
      setTagsText(annotation.tags.join(", "));
    }
  }, [annotation]);

  const normalizedTags = tagsText
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean)
    .slice(0, 12);
  const canSave = draft.taskPrompt.trim().length > 0;

  const buildSavePayload = (overrides?: Partial<EpisodeAnnotation>): EpisodeAnnotation => ({
    ...draft,
    datasetRoot: datasetPath,
    tags: normalizedTags,
    taskPrompt: draft.taskPrompt.trim(),
    annotator: draft.annotator.trim(),
    notes: draft.notes.trim(),
    ...overrides
  });

  const markFailure = (tag: string) => {
    const existingTags = new Set(normalizedTags);
    existingTags.add(tag);
    onSave(buildSavePayload({
      outcome: "failure",
      quality: "bad",
      includeInTraining: false,
      tags: [...existingTags]
    }));
  };

  const reviewBorder = draft.reviewStatus === "rejected" ? "2px solid #dc2626" : undefined;

  return (
    <>
      <section className="panel annotation-panel" style={{ borderLeft: reviewBorder }}>
        <div className="panel-heading">
          <h2>Episode Annotation</h2>
          <span>
            {draft.reviewStatus !== "pending" && (
              <span className={`review-badge review-${draft.reviewStatus}`}>{draft.reviewStatus}</span>
            )}
            {draft.updatedAt ? ` saved ${new Date(draft.updatedAt).toLocaleString()}` : " not saved"}
          </span>
        </div>
        <div className="annotation-grid">
          <label className="annotation-field annotation-field-wide">
            <span>Task Prompt</span>
            <textarea
              value={draft.taskPrompt}
              onChange={(event) => setDraft({ ...draft, taskPrompt: event.target.value })}
              placeholder="Pick up the red cube and place it into the fixture"
            />
          </label>
          <label className="annotation-field">
            <span>Outcome</span>
            <select
              value={draft.outcome}
              onChange={(event) => setDraft({ ...draft, outcome: event.target.value as EpisodeAnnotation["outcome"] })}
            >
              <option value="unreviewed">Unreviewed</option>
              <option value="success">Success</option>
              <option value="partial">Partial</option>
              <option value="failure">Failure</option>
            </select>
          </label>
          <label className="annotation-field">
            <span>Quality</span>
            <select
              value={draft.quality}
              onChange={(event) => setDraft({ ...draft, quality: event.target.value as EpisodeAnnotation["quality"] })}
            >
              <option value="unreviewed">Unreviewed</option>
              <option value="good">Good</option>
              <option value="needs_review">Needs review</option>
              <option value="bad">Bad</option>
            </select>
          </label>
          <label className="annotation-field annotation-toggle">
            <input
              checked={draft.includeInTraining}
              type="checkbox"
              onChange={(event) => setDraft({ ...draft, includeInTraining: event.target.checked })}
            />
            <span>Use for training</span>
          </label>
          <label className="annotation-field">
            <span>Annotator</span>
            <input
              value={draft.annotator}
              onChange={(event) => setDraft({ ...draft, annotator: event.target.value })}
              placeholder="operator"
            />
          </label>
          <label className="annotation-field annotation-field-wide">
            <span>Tags</span>
            <input
              value={tagsText}
              onChange={(event) => setTagsText(event.target.value)}
              placeholder="occlusion, collision, retry, object-slip"
            />
          </label>
          <label className="annotation-field annotation-field-wide">
            <span>Notes</span>
            <textarea
              value={draft.notes}
              onChange={(event) => setDraft({ ...draft, notes: event.target.value })}
              placeholder="Short review notes, failure reason, or scene details"
            />
          </label>
          <label className="annotation-field">
            <span>Review Status</span>
            <select
              value={draft.reviewStatus}
              onChange={(event) => setDraft({ ...draft, reviewStatus: event.target.value as EpisodeAnnotation["reviewStatus"] })}
            >
              <option value="pending">Pending</option>
              <option value="approved">Approved</option>
              <option value="rejected">Rejected</option>
            </select>
          </label>
          <label className="annotation-field annotation-field-wide">
            <span>Review Comment</span>
            <textarea
              value={draft.reviewComment}
              onChange={(event) => setDraft({ ...draft, reviewComment: event.target.value })}
              placeholder="Reason for rejection or review notes"
              className={draft.reviewStatus === "rejected" ? "review-rejected-input" : ""}
            />
          </label>
        </div>
        <div className="control-row">
          <button
            disabled={busy || !datasetPath || !canSave}
            onClick={() => onSave(buildSavePayload())}
          >
            Save Annotation
          </button>
          <button
            className="danger"
            disabled={busy || !datasetPath}
            onClick={() => markFailure("collision")}
          >
            Mark Collision
          </button>
          <button
            className="danger"
            disabled={busy || !datasetPath}
            onClick={() => markFailure("abandoned")}
          >
            Mark Abandoned
          </button>
          <span className="annotation-hint">Episode {draft.episode} · {draft.source}</span>
        </div>
      </section>
      <SubtaskSegmentEditor
        segments={draft.segments ?? []}
        totalFrames={totalFrames}
        onChange={(segments) => setDraft({ ...draft, segments })}
      />
    </>
  );
}

function EpisodeSelector({
  status,
  annotation,
  busy,
  onSelectEpisode
}: {
  status: ReplayStatus;
  annotation?: EpisodeAnnotation;
  busy: boolean;
  onSelectEpisode: (episode: number) => void;
}) {
  const [pendingEpisode, setPendingEpisode] = useState<number | null>(null);
  const [episodeInput, setEpisodeInput] = useState(String(status.episode ?? 0));
  const options = status.episodeOptions?.length
    ? status.episodeOptions
    : Array.from({ length: status.totalEpisodes ?? 0 }, (_item, index) => index);
  const current = status.episode ?? 0;
  const currentIndex = options.indexOf(current);
  const hasPrevious = currentIndex > 0;
  const hasNext = currentIndex >= 0 && currentIndex < options.length - 1;
  const switching = pendingEpisode != null && (busy || pendingEpisode !== current);
  const inputEpisode = episodeInput.trim() === "" ? NaN : Number(episodeInput);
  const inputIsInteger = Number.isInteger(inputEpisode);
  const inputInOptions = inputIsInteger && options.includes(inputEpisode);
  const canGoToInput = !busy && inputInOptions && inputEpisode !== current;

  useEffect(() => {
    if (!busy && pendingEpisode === current) {
      setPendingEpisode(null);
    }
  }, [busy, current, pendingEpisode]);

  useEffect(() => {
    if (!switching) {
      setEpisodeInput(String(current));
    }
  }, [current, switching]);

  const selectEpisode = (episode: number) => {
    setEpisodeInput(String(episode));
    setPendingEpisode(episode);
    onSelectEpisode(episode);
  };

  const submitEpisodeInput = () => {
    if (canGoToInput) {
      selectEpisode(inputEpisode);
    }
  };

  return (
    <section className="panel episode-selector-panel">
      <div className="panel-heading">
        <h2>Episode</h2>
        <span>{options.length ? `${options.length} available` : "no episode index"}</span>
      </div>
      <div className="episode-selector-row">
        <button disabled={busy || !hasPrevious} onClick={() => selectEpisode(options[currentIndex - 1])}>Previous</button>
        <select
          disabled={busy || options.length === 0}
          value={current}
          onChange={(event) => selectEpisode(Number(event.target.value))}
        >
          {options.length ? (
            options.map((episode) => (
              <option value={episode} key={episode}>
                Episode {episode}
              </option>
            ))
          ) : (
            <option value={current}>Episode {current}</option>
          )}
        </select>
        <div className="episode-input-group">
          <input
            aria-label="Episode number"
            disabled={busy || options.length === 0}
            inputMode="numeric"
            min={options[0] ?? 0}
            max={options[options.length - 1] ?? current}
            step={1}
            type="number"
            value={episodeInput}
            onChange={(event) => setEpisodeInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                submitEpisodeInput();
              }
            }}
          />
          <button disabled={!canGoToInput} onClick={submitEpisodeInput}>Go</button>
        </div>
        <button disabled={busy || !hasNext} onClick={() => selectEpisode(options[currentIndex + 1])}>Next</button>
      </div>
      <div className="episode-badges">
        {annotation?.outcome && annotation.outcome !== "unreviewed" && (
          <span className={`outcome-badge outcome-${annotation.outcome}`}>{annotation.outcome}</span>
        )}
        {annotation?.reviewStatus && annotation.reviewStatus !== "pending" && (
          <span className={`review-badge review-${annotation.reviewStatus}`}>{annotation.reviewStatus}</span>
        )}
      </div>
      <p className="panel-note">
        {switching
          ? `Switching to episode ${pendingEpisode}: reading episode metadata and preparing video/timeline resources.`
          : "Inspector, annotation, and replay commands follow the selected episode."}
      </p>
    </section>
  );
}

function EpisodeReplayPage({
  snapshot,
  busy,
  onPreflight,
  onReplay,
  onMujocoReplay,
  onAbort,
  onSelectDataset,
  onSelectEpisode,
  onGenerateForActive,
  onOpenProcessing,
  onSaveAnnotation
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onPreflight: () => void;
  onReplay: (realRobot: boolean) => void;
  onMujocoReplay: () => void;
  onAbort: () => void;
  onSelectDataset: (path: string) => void;
  onSelectEpisode: (episode: number) => void;
  onGenerateForActive: () => void;
  onOpenProcessing: () => void;
  onSaveAnnotation: (annotation: EpisodeAnnotation) => void;
}) {
  const activePath = snapshot.replay.datasetRoot ?? snapshot.replay.dataset;
  const matchingProcessing =
    snapshot.processing.find((item) => item.path === activePath) ?? snapshot.processing[0] ?? null;
  return (
    <div className="page-stack">
      <PageHeader title="Episode Replay" subtitle="consume processed trajectories: timeline review, safety preflight, dry-run, and real-robot replay" />
      <ReplayReadinessCard
        status={snapshot.replay}
        processing={matchingProcessing}
        busy={busy}
        onGenerate={onGenerateForActive}
        onOpenProcessing={onOpenProcessing}
      />
      <div className="replay-workspace">
        <RecordedDatasetList
          datasets={snapshot.recordedDatasets}
          activePath={activePath}
          busy={busy}
          onSelect={onSelectDataset}
        />
        <ReplayPanel
          status={snapshot.replay}
          busy={busy}
          onPreflight={onPreflight}
          onReplay={onReplay}
          onMujocoReplay={onMujocoReplay}
          onAbort={onAbort}
        />
      </div>
      <EpisodeSelector status={snapshot.replay} annotation={snapshot.annotation} busy={busy} onSelectEpisode={onSelectEpisode} />
      <ReplayInspector api={api} datasetPath={activePath} episode={snapshot.replay.episode} fallbackFps={snapshot.replay.fps} />
      <EpisodeAnnotationPanel
        annotation={snapshot.annotation}
        datasetPath={activePath}
        totalFrames={snapshot.replay.totalFrames}
        busy={busy}
        onSave={onSaveAnnotation}
      />
      <EventLog events={snapshot.events} />
    </div>
  );
}

const processingStatusLabel: Record<ProcessingStatus, string> = {
  pose_missing: "Pose missing",
  queued: "Queued",
  running: "Running",
  pose_ready: "Pose ready",
  qc_pass: "QC pass",
  qc_failed: "QC failed",
  error: "Error"
};

const processingStatusDot: Record<ProcessingStatus, string> = {
  pose_missing: "warning",
  queued: "connecting",
  running: "running",
  pose_ready: "armed",
  qc_pass: "running",
  qc_failed: "error",
  error: "error"
};

function ProcessingRow({
  item,
  active,
  busy,
  onSelect,
  onGenerate,
  onRunQc,
  onOpenReplay
}: {
  item: ProcessingItem;
  active: boolean;
  busy: boolean;
  onSelect: () => void;
  onGenerate: () => void;
  onRunQc: () => void;
  onOpenReplay: () => void;
}) {
  const primary =
    item.status === "pose_missing" ? (
      <button disabled={busy} onClick={onGenerate}>Generate EE Trajectory</button>
    ) : item.status === "queued" || item.status === "running" ? (
      <button disabled={busy} onClick={onSelect}>View Log</button>
    ) : item.status === "qc_pass" ? (
      <button disabled={busy} onClick={onOpenReplay}>Open Replay</button>
    ) : item.status === "error" && !item.trajectoryVersion ? (
      // trajectory generation failed: no trajectory to QC, offer to regenerate instead
      <button disabled={busy} onClick={onGenerate}>Regenerate Trajectory</button>
    ) : item.status === "qc_failed" || item.status === "error" ? (
      <button disabled={busy} onClick={onRunQc}>Re-run QC</button>
    ) : (
      <button disabled={busy} onClick={onRunQc}>Run QC</button>
    );

  return (
    <div className={active ? "processing-row active" : "processing-row"}>
      <button className="processing-row-main" onClick={onSelect} disabled={busy}>
        <div>
          <div className="row-title">
            <StatusDot state={processingStatusDot[item.status]} />
            <strong>{item.name}</strong>
            {item.trajectoryVersion ? <em>{item.trajectoryVersion}</em> : null}
          </div>
          <p>{item.message}</p>
        </div>
        <div className="processing-stats">
          <span>{processingStatusLabel[item.status]}</span>
          <small>{item.updatedAt}</small>
        </div>
      </button>
      <div className="processing-actions">{primary}</div>
    </div>
  );
}

function DatasetProcessingPage({
  snapshot,
  busy,
  onGenerate,
  onRunQc,
  onOpenReplay,
  onSetDatasetsRoot
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onGenerate: (path: string) => void;
  onRunQc: (path: string) => void;
  onOpenReplay: (path: string) => void;
  onSetDatasetsRoot: (path: string) => void;
}) {
  const items = snapshot.processing;
  const [selectedPath, setSelectedPath] = useState<string>(items[0]?.path ?? "");
  const selected = items.find((item) => item.path === selectedPath) ?? items[0];
  const runningCount = items.filter((item) => item.status === "running" || item.status === "queued").length;
  const failedCount = items.filter((item) => item.status === "qc_failed" || item.status === "error").length;
  const readyCount = items.filter((item) => item.status === "qc_pass").length;
  const currentRoot = snapshot.gateway.datasetsRoot ?? "";
  const [rootInput, setRootInput] = useState<string>(currentRoot);
  useEffect(() => {
    setRootInput(currentRoot);
  }, [currentRoot]);
  const rootDirty = rootInput.trim() !== currentRoot;

  return (
    <div className="page-stack">
      <PageHeader title="Dataset Processing" subtitle="EE trajectory generation, QC, and post-processing job queue for recorded datasets" />
      <section className="panel">
        <div className="panel-heading">
          <h2>Datasets Root</h2>
          <span>scan path</span>
        </div>
        <div className="datasets-root-row">
          <input
            className="datasets-root-input"
            value={rootInput}
            onChange={(event) => setRootInput(event.target.value)}
            placeholder="outputs/datasets"
            spellCheck={false}
          />
          <button
            disabled={busy || !rootDirty || !rootInput.trim()}
            onClick={() => onSetDatasetsRoot(rootInput.trim())}
          >
            Save
          </button>
          <button
            disabled={busy || rootInput.trim() === currentRoot}
            onClick={() => setRootInput(currentRoot)}
          >
            Reset
          </button>
        </div>
        <p className="panel-note">Absolute or relative to the gateway repo root. Default: outputs/datasets.</p>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>Queue Overview</h2>
          <span>{items.length} datasets</span>
        </div>
        <div className="summary-grid">
          <Metric label="In flight" value={runningCount} />
          <Metric label="QC passed" value={readyCount} />
          <Metric label="Failed" value={failedCount} />
          <Metric label="Total" value={items.length} />
        </div>
      </section>
      <div className="processing-workspace">
        <section className="panel processing-list-panel">
          <div className="panel-heading">
            <h2>Datasets</h2>
            <span>job queue</span>
          </div>
          {items.length === 0 ? (
            snapshot.gateway.state !== "online" ? (
              <div className="empty-dataset-list">
                <p>Gateway not connected (state: {snapshot.gateway.state}).</p>
                <p>Ensure the gateway is running on Thor:</p>
                <code>ssh nvidia@192.168.111.122</code>
                <code>bash ~/lerobot/run/restart_gateway.sh</code>
                <p>Then set the Vite proxy target:</p>
                <code>GUI_API_TARGET=http://192.168.111.122:8765 npm run dev</code>
              </div>
            ) : (
              <div className="empty-dataset-list">
                No datasets found under <code>{currentRoot || "(unset)"}</code>. Update the path above or
                record an episode first.
              </div>
            )
          ) : (
            <div className="processing-list">
              {items.map((item) => (
                <ProcessingRow
                  key={item.path}
                  item={item}
                  active={item.path === selected?.path}
                  busy={busy}
                  onSelect={() => setSelectedPath(item.path)}
                  onGenerate={() => onGenerate(item.path)}
                  onRunQc={() => onRunQc(item.path)}
                  onOpenReplay={() => onOpenReplay(item.path)}
                />
              ))}
            </div>
          )}
        </section>
        {selected ? (
          <section className="panel processing-detail-panel">
            <div className="panel-heading">
              <h2>{selected.name}</h2>
              <span className="state-pill">
                <StatusDot state={processingStatusDot[selected.status]} />
                {processingStatusLabel[selected.status]}
              </span>
            </div>
            <div className="summary-grid">
              <Metric label="Episodes" value={selected.totalEpisodes} />
              <Metric label="Frames" value={selected.totalFrames} />
              <Metric label="Trajectory" value={selected.trajectoryVersion ?? "—"} />
              <Metric label="Valid frames" value={selected.validFramesPct != null ? `${selected.validFramesPct}%` : "—"} />
            </div>
            <div className="control-row">
              <button
                disabled={busy || selected.status === "running" || selected.status === "queued"}
                onClick={() => onGenerate(selected.path)}
              >
                {selected.trajectoryVersion ? "Regenerate Trajectory" : "Generate EE Trajectory"}
              </button>
              <button
                disabled={
                  busy ||
                  !["pose_ready", "qc_pass", "qc_failed", "error"].includes(selected.status) ||
                  // trajectory generation failed: nothing to QC until a trajectory exists
                  (selected.status === "error" && !selected.trajectoryVersion)
                }
                onClick={() => onRunQc(selected.path)}
              >
                {selected.status === "qc_failed" || (selected.status === "error" && selected.trajectoryVersion)
                  ? "Re-run QC"
                  : "Run QC"}
              </button>
              <button
                disabled={busy || !["pose_ready", "qc_pass"].includes(selected.status)}
                onClick={() => onOpenReplay(selected.path)}
              >
                Open Replay
              </button>
            </div>
            <div className="qc-block">
              <h3>QC summary</h3>
              <p>{selected.qcSummary}</p>
            </div>
            <div className="log-block">
              <h3>Log</h3>
              {selected.logTail.length === 0 ? (
                <p className="panel-note">No log entries yet.</p>
              ) : (
                <pre className="log-tail">{selected.logTail.join("\n")}</pre>
              )}
            </div>
          </section>
        ) : null}
      </div>
      <QualityOverview snapshot={snapshot} />
    </div>
  );
}

function QcReportPage({ snapshot }: { snapshot: GuiSnapshot }) {
  return (
    <div className="page-stack">
      <PageHeader title="QC Report" subtitle="episode-level data quality checks for sync, completeness, device health, and schema readiness" />
      <QualityOverview snapshot={snapshot} />
      <section className="panel">
        <div className="panel-heading">
          <h2>Checks</h2>
          <span>minimum viable QC</span>
        </div>
        <div className="check-table">
          {[
            ["camera fps", `${snapshot.configSummary.fps} fps target`, "pass"],
            ["frame drop", `${snapshot.trajectory.filter((point) => point.event === "gap").length} gaps`, "review"],
            ["timestamp gap", `${Math.max(0, ...snapshot.trajectory.map((point) => point.skewMs)).toFixed(1)} ms max`, "review"],
            ["action latency", `${snapshot.replay.trackingErrorMm.toFixed(1)} mm replay error`, "pass"],
            ["LeRobot schema", snapshot.replay.dataStatus === "loaded" ? "trajectory loaded" : "pending dataset", "review"]
          ].map(([name, value, state]) => (
            <div className="check-row" key={name}>
              <strong>{name}</strong>
              <span>{value}</span>
              <em>{state}</em>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function DatasetExportPage({
  snapshot,
  busy,
  onPrepare,
  onExport,
  onExportTask,
  onOpenProcessing
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onPrepare: (target: GuiSnapshot["datasetExport"]["target"]) => void;
  onExport: () => void;
  onExportTask: (id: string) => void;
  onOpenProcessing: () => void;
}) {
  const exportStatus = snapshot.datasetExport;
  const eligible = snapshot.processing.filter((item) => item.status === "qc_pass");
  const hasEligible = eligible.length > 0;
  const exportableTasks = (snapshot.tasks ?? []).filter((t) => t.datasetRepoId);
  const exporting = exportStatus.state === "exporting";
  return (
    <div className="page-stack">
      <PageHeader title="Dataset Export" subtitle="package QC-approved datasets into LeRobot, MCAP, or Parquet bundles" />
      <section className="panel">
        <div className="panel-heading">
          <h2>Consolidate a Task</h2>
          <span>{exportableTasks.length} exportable</span>
        </div>
        <p className="panel-note">
          Merge every recorded session of a task into one LeRobot v3 dataset under the exports root. Raw sessions are left untouched; re-run any time.
        </p>
        {exportableTasks.length === 0 ? (
          <div className="empty-dataset-list">No tasks with a dataset repo id. Create one in Task Library first.</div>
        ) : (
          <div className="processing-list">
            {exportableTasks.map((task) => (
              <div className="processing-row" key={task.id}>
                <div className="processing-row-main static">
                  <div>
                    <div className="row-title">
                      <StatusDot state={taskStatusDot[task.status]} />
                      <strong>{task.name}</strong>
                      <em>{task.datasetRepoId}</em>
                    </div>
                    <p>{task.completedEpisodes} episode(s) recorded across its sessions</p>
                  </div>
                  <div className="processing-stats">
                    <button
                      disabled={busy || exporting}
                      onClick={() => onExportTask(task.id)}
                    >
                      {exporting && exportStatus.taskId === task.id ? "Exporting…" : "Export v3"}
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
        {exportStatus.state !== "idle" && (
          <div className="summary-grid">
            <Metric label="Export state" value={stateLabel(exportStatus.state)} />
            <Metric label="Output" value={exportStatus.outputPath || "—"} />
            <Metric label="Episodes" value={exportStatus.selectedEpisodes} />
            <Metric label="Frames" value={exportStatus.totalFrames} />
            <Metric label="Message" value={exportStatus.message} />
          </div>
        )}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>Approved Datasets</h2>
          <span>{eligible.length} ready</span>
        </div>
        {hasEligible ? (
          <div className="processing-list">
            {eligible.map((item) => (
              <div className="processing-row" key={item.path}>
                <div className="processing-row-main static">
                  <div>
                    <div className="row-title">
                      <StatusDot state="running" />
                      <strong>{item.name}</strong>
                      {item.trajectoryVersion ? <em>{item.trajectoryVersion}</em> : null}
                    </div>
                    <p>{item.qcSummary}</p>
                  </div>
                  <div className="processing-stats">
                    <span>{item.totalEpisodes} ep · {item.totalFrames} fr</span>
                    <small>{item.updatedAt}</small>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-dataset-list">
            No QC-passed datasets yet. Run QC in Dataset Processing first.
            <div className="control-row">
              <button disabled={busy} onClick={onOpenProcessing}>Open Processing</button>
            </div>
          </div>
        )}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>Export Plan</h2>
          <span className="state-pill">
            <StatusDot state={exportStatus.state} />
            {stateLabel(exportStatus.state)}
          </span>
        </div>
        <div className="control-row">
          <button disabled={busy || !hasEligible} onClick={() => onPrepare("lerobot_v3")}>LeRobot v3</button>
          <button disabled={busy || !hasEligible} onClick={() => onPrepare("mcap")}>MCAP</button>
          <button disabled={busy || !hasEligible} onClick={() => onPrepare("parquet")}>Parquet</button>
          <button disabled={busy || exportStatus.state !== "ready" || !hasEligible} onClick={onExport}>Run Export</button>
        </div>
        <div className="summary-grid">
          <Metric label="Target" value={exportStatus.target} />
          <Metric label="Dataset root" value={exportStatus.datasetRoot} />
          <Metric label="Output" value={exportStatus.outputPath} />
          <Metric label="Episodes" value={exportStatus.selectedEpisodes} />
          <Metric label="Frames" value={exportStatus.totalFrames} />
          <Metric label="Message" value={exportStatus.message} />
        </div>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>Layer Manifest</h2>
          <span>Raw / Debug / Training</span>
        </div>
        <div className="layer-grid">
          {[
            ["Raw Layer", exportStatus.includeRaw, "video, robot state, controller state, sidecar jsonl"],
            ["Debug Layer", exportStatus.includeDebug, "MCAP, Rerun log, timeline index"],
            ["Training Layer", exportStatus.includeTraining, "LeRobot v3, Parquet, dataset card"]
          ].map(([label, enabled, detail]) => (
            <div className="layer-card" key={String(label)}>
              <strong>{label}</strong>
              <span>{enabled ? "included" : "excluded"}</span>
              <p>{detail}</p>
            </div>
          ))}
        </div>
        <ul className="manifest-list">
          {exportStatus.manifest.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}

const taskStatusLabel: Record<TaskStatus, string> = {
  pending: "Pending",
  in_progress: "In Progress",
  completed: "Completed",
  paused: "Paused"
};

const taskStatusDot: Record<TaskStatus, string> = {
  pending: "idle",
  in_progress: "running",
  completed: "completed",
  paused: "warning"
};

function TaskLibraryPage({
  snapshot,
  busy,
  onCreate,
  onUpdate,
  onDelete,
  onActivate,
  onNavigate
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onCreate: (task: Partial<CollectionTask>) => void;
  onUpdate: (task: Partial<CollectionTask>) => void;
  onDelete: (id: string) => void;
  onActivate: (id: string) => void;
  onNavigate: (page: PageId) => void;
}) {
  const tasks = snapshot.tasks ?? [];
  const [showForm, setShowForm] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [formName, setFormName] = useState("");
  const [formDesc, setFormDesc] = useState("");
  const [formTarget, setFormTarget] = useState(100);
  const [formAssignee, setFormAssignee] = useState("");
  const [formRepoId, setFormRepoId] = useState("");

  const selected = tasks.find((t) => t.id === selectedId) ?? null;
  const selectedTargetReached =
    selected != null && selected.targetEpisodes > 0 && selected.completedEpisodes >= selected.targetEpisodes;

  const resetForm = () => {
    setFormName("");
    setFormDesc("");
    setFormTarget(100);
    setFormAssignee("");
    setFormRepoId("");
  };

  const submitCreate = () => {
    if (!formName.trim()) return;
    onCreate({
      name: formName.trim(),
      description: formDesc.trim(),
      targetEpisodes: formTarget,
      assignee: formAssignee.trim(),
      datasetRepoId: formRepoId.trim()
    });
    resetForm();
    setShowForm(false);
  };

  const inProgress = tasks.filter((t) => t.status === "in_progress").length;
  const completed = tasks.filter((t) => t.status === "completed").length;

  return (
    <div className="page-stack">
      <PageHeader title="Task Library" subtitle="create and manage data collection tasks with target episode counts and progress tracking" />
      <section className="panel">
        <div className="panel-heading">
          <h2>Overview</h2>
          <span>{tasks.length} tasks</span>
        </div>
        <div className="summary-grid">
          <Metric label="Total" value={tasks.length} />
          <Metric label="In Progress" value={inProgress} />
          <Metric label="Completed" value={completed} />
        </div>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>Tasks</h2>
          <button disabled={busy} onClick={() => setShowForm(!showForm)}>{showForm ? "Cancel" : "New Task"}</button>
        </div>
        {showForm && (
          <div className="task-form">
            <label className="annotation-field">
              <span>Name</span>
              <input value={formName} onChange={(e) => setFormName(e.target.value)} placeholder="Task name" />
            </label>
            <label className="annotation-field annotation-field-wide">
              <span>Description</span>
              <textarea value={formDesc} onChange={(e) => setFormDesc(e.target.value)} placeholder="What to collect" />
            </label>
            <label className="annotation-field">
              <span>Target Episodes</span>
              <input type="number" min={1} value={formTarget} onChange={(e) => setFormTarget(Number(e.target.value))} />
            </label>
            <label className="annotation-field">
              <span>Assignee</span>
              <input value={formAssignee} onChange={(e) => setFormAssignee(e.target.value)} placeholder="operator" />
            </label>
            <label className="annotation-field">
              <span>Dataset Repo ID</span>
              <input value={formRepoId} onChange={(e) => setFormRepoId(e.target.value)} placeholder="local/my_task" />
            </label>
            <div className="control-row">
              <button disabled={busy || !formName.trim()} onClick={submitCreate}>Create Task</button>
            </div>
          </div>
        )}
        {tasks.length === 0 && !showForm ? (
          <div className="empty-dataset-list">No tasks yet. Click "New Task" to create one.</div>
        ) : (
          <div className="processing-list">
            {tasks.map((task) => {
              const progress = task.targetEpisodes > 0
                ? Math.min(100, Math.round((task.completedEpisodes / task.targetEpisodes) * 100))
                : 0;
              return (
                <div
                  className={task.id === selectedId ? "processing-row active" : "processing-row"}
                  key={task.id}
                >
                  <button className="processing-row-main" onClick={() => setSelectedId(task.id)} disabled={busy}>
                    <div>
                      <div className="row-title">
                        <StatusDot state={taskStatusDot[task.status]} />
                        <strong>{task.name}</strong>
                        <em>{taskStatusLabel[task.status]}</em>
                      </div>
                      <p>{task.description || "No description"}</p>
                    </div>
                    <div className="processing-stats">
                      <span>{task.completedEpisodes} / {task.targetEpisodes} episodes</span>
                      <div className="progress" style={{ width: 80, height: 6 }}>
                        <div className="progress-bar" style={{ width: `${progress}%` }} />
                      </div>
                      <small>{task.assignee || "unassigned"}</small>
                    </div>
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </section>
      {selected && (
        <section className="panel">
          <div className="panel-heading">
            <h2>{selected.name}</h2>
            <span className="state-pill">
              <StatusDot state={taskStatusDot[selected.status]} />
              {taskStatusLabel[selected.status]}
            </span>
          </div>
          <div className="summary-grid">
            <Metric label="Progress" value={`${selected.completedEpisodes} / ${selected.targetEpisodes}`} />
            <Metric label="Assignee" value={selected.assignee || "—"} />
            <Metric label="Dataset" value={selected.datasetRepoId || "—"} />
            <Metric label="Created" value={selected.createdAt ? new Date(selected.createdAt).toLocaleString() : "—"} />
            <Metric label="Updated" value={selected.updatedAt ? new Date(selected.updatedAt).toLocaleString() : "—"} />
          </div>
          {!selected.datasetRepoId && (
            <p className="panel-note">No dataset linked — progress can't be tracked. Set a Dataset Repo ID so recorded episodes count toward this task.</p>
          )}
          {selectedTargetReached && selected.status !== "completed" && (
            <p className="panel-note">Target reached ({selected.completedEpisodes}/{selected.targetEpisodes}). Mark the task Complete when collection is done.</p>
          )}
          <div className="control-row">
            <button
              disabled={busy || selected.status === "in_progress"}
              onClick={() => onUpdate({ id: selected.id, status: "in_progress" })}
            >
              {selected.status === "paused" ? "Resume" : selected.status === "completed" ? "Reopen" : "Start"}
            </button>
            <button
              disabled={busy || selected.status !== "in_progress"}
              onClick={() => onUpdate({ id: selected.id, status: "paused" })}
            >
              Pause
            </button>
            <button
              disabled={busy || selected.status === "completed"}
              onClick={() => onUpdate({ id: selected.id, status: "completed" })}
            >
              Complete
            </button>
            <button
              disabled={busy || selected.status === "completed" || !selected.datasetRepoId}
              title={selected.datasetRepoId ? "Bind recording to this task's dataset and open Live Record" : "Set a Dataset Repo ID first"}
              onClick={() => {
                onActivate(selected.id);
                if (selected.status !== "in_progress") {
                  onUpdate({ id: selected.id, status: "in_progress" });
                }
                onNavigate("live-record");
              }}
            >
              Go to Record
            </button>
            <button
              className="danger"
              disabled={busy}
              onClick={() => {
                if (window.confirm(`Delete task "${selected.name}"? This removes the task entry only — recorded datasets are not touched.`)) {
                  onDelete(selected.id);
                  setSelectedId(null);
                }
              }}
            >
              Delete
            </button>
          </div>
        </section>
      )}
    </div>
  );
}

const DEVICE_TOUCH_ROW_LENGTHS = [13, 13, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 13, 13];
const DEVICE_TOUCH_COLUMNS = 17;

function interpolatePreviewChannel(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

function previewTouchColor(value: number, scaleMax: number): string {
  const stops = [
    [17, 24, 39],
    [37, 99, 235],
    [20, 184, 166],
    [250, 204, 21],
    [239, 68, 68]
  ];
  const normalized = Math.max(0, Math.min(1, value / Math.max(scaleMax, 1)));
  const scaled = normalized * (stops.length - 1);
  const index = Math.min(Math.floor(scaled), stops.length - 2);
  const t = scaled - index;
  const a = stops[index];
  const b = stops[index + 1];
  return `rgb(${interpolatePreviewChannel(a[0], b[0], t)}, ${interpolatePreviewChannel(a[1], b[1], t)}, ${interpolatePreviewChannel(a[2], b[2], t)})`;
}

function numberArray(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => Number(item)).filter((item) => Number.isFinite(item));
}

function numberValue(value: unknown): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function DeviceTouchPreview({ sensor }: { sensor?: Record<string, unknown> | null }) {
  const values = numberArray(sensor?.fz_0p1N);
  const hasData = values.length >= 239;
  const scaleMax = hasData ? Math.max(1, ...values.map((value) => Math.abs(value))) : 1;
  const localMax = hasData ? Math.max(...values.map((value) => Math.abs(value))) : 0;
  const activePoints = values.filter((value) => Math.abs(value) > 0).length;
  let cursor = 0;

  return (
    <div className="touch-map device-touch-map">
      <div className="touch-map-heading">
        <strong>tactile</strong>
        <span>max {localMax.toFixed(1)} · active {activePoints}</span>
      </div>
      {hasData ? (
        <div className="touch-grid" aria-label="live tactile preview">
          {DEVICE_TOUCH_ROW_LENGTHS.map((length, rowIndex) => {
            const offset = Math.floor((DEVICE_TOUCH_COLUMNS - length) / 2);
            const row = values.slice(cursor, cursor + length);
            const startIndex = cursor;
            cursor += length;
            return (
              <div className="touch-row" key={rowIndex}>
                {Array.from({ length: offset }).map((_, index) => (
                  <span className="touch-cell touch-cell-empty" key={`pre-${index}`} />
                ))}
                {row.map((value, index) => {
                  const pointIndex = startIndex + index + 1;
                  return (
                    <span
                      className="touch-cell"
                      key={pointIndex}
                      title={`#${pointIndex} fz=${value.toFixed(1)} (0.1N)`}
                      style={{ backgroundColor: previewTouchColor(Math.abs(value), scaleMax) }}
                    />
                  );
                })}
                {Array.from({ length: DEVICE_TOUCH_COLUMNS - length - offset }).map((_, index) => (
                  <span className="touch-cell touch-cell-empty" key={`post-${index}`} />
                ))}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="touch-empty">no touch sample</div>
      )}
      <div className="touch-map-footer">
        <span>ts {String(sensor?.timestamp ?? "-")}</span>
        <span>fz 0.1N</span>
      </div>
    </div>
  );
}

function DeviceForcePreview({ sensor }: { sensor?: Record<string, unknown> | null }) {
  const values = numberArray(sensor?.fxyz_mxyz);
  const labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"];
  const maxAbs = Math.max(1, ...values.slice(0, 6).map((value) => Math.abs(value)));
  if (values.length < 6) {
    return <div className="preview-empty">no force sample</div>;
  }
  return (
    <div className="force-preview-grid">
      {labels.map((label, index) => {
        const value = values[index] ?? 0;
        return (
          <div className="force-preview-row" key={label}>
            <span>{label}</span>
            <div className="force-preview-track">
              <i style={{ width: `${Math.min(100, Math.abs(value) / maxAbs * 100)}%` }} />
            </div>
            <strong>{value.toFixed(2)}</strong>
          </div>
        );
      })}
      <small>ts {String(sensor?.timestamp ?? "-")}</small>
    </div>
  );
}

function RawSensorPreview({ sensor }: { sensor?: Record<string, unknown> | null }) {
  const entries = Object.entries(sensor ?? {}).filter(([, value]) => typeof value !== "object").slice(0, 12);
  if (entries.length === 0) {
    return <div className="preview-empty">no live sample</div>;
  }
  return (
    <div className="device-config-detail device-preview-kv">
      {entries.map(([key, value]) => (
        <div className="device-config-row" key={key}>
          <span className="device-config-key">{key}</span>
          <span className="device-config-value">{String(value)}</span>
        </div>
      ))}
    </div>
  );
}

function BoxLivePreview({ device }: { device: DeviceStatus }) {
  const [preview, setPreview] = useState<BoxPreviewPayload | null>(null);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      const next = await api.fetchBoxPreview(device.id);
      if (mounted) {
        setPreview(next);
      }
    };
    load();
    const timer = window.setInterval(load, 100);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [device.id]);

  const sensor = preview?.sensor ?? null;
  const staleS = preview?.staleS == null ? null : preview.staleS;
  const isTouch = device.id.startsWith("box_touch") || Array.isArray(sensor?.fz_0p1N);
  const isForce = device.id === "box_six_d_force" || Array.isArray(sensor?.fxyz_mxyz);
  const sampleAge = staleS == null ? "-" : `${staleS.toFixed(1)}s`;
  const queueSize = numberValue(preview?.status?.queue_size);

  return (
    <div className="device-preview-live">
      <div className="device-live-stats">
        <Metric label="Live" value={preview?.active ? "yes" : "no"} />
        <Metric label="Age" value={sampleAge} />
        <Metric label="Queue" value={queueSize ?? "-"} />
      </div>
      {isTouch ? <DeviceTouchPreview sensor={sensor} /> : isForce ? <DeviceForcePreview sensor={sensor} /> : <RawSensorPreview sensor={sensor} />}
    </div>
  );
}

function DeviceInlinePreview({ device }: { device: DeviceStatus }) {
  // Cameras render their live snapshot through CameraTile in the grid; this
  // inline preview only covers the non-camera expandable rows.
  return (
    <div className="device-inline-preview">
      {device.kind === "box_collection" ? (
        <BoxLivePreview device={device} />
      ) : (
        <div className="preview-empty">no preview stream</div>
      )}
    </div>
  );
}

// BOX device ids are namespaced as `<box_id>/<sensor>` when more than one BOX
// is configured; strip the prefix so we match on the bare sensor name.
function boxSensorSuffix(deviceId: string): string {
  const slash = deviceId.lastIndexOf("/");
  return slash >= 0 ? deviceId.slice(slash + 1) : deviceId;
}

// The Paxini touch pads and the 6D force sensor carry array payloads that read
// far better as a full-frame visualization (like a camera tile) than as the
// scalar key/value rows the gripper/IMU/trigger use.
function isVisualBoxSensor(deviceId: string): boolean {
  const sid = boxSensorSuffix(deviceId);
  return sid.startsWith("box_touch") || sid === "box_six_d_force";
}

// Full-frame tactile heatmap: same point layout as DeviceTouchPreview but sized
// to fill the tile media (height-bound, centered) instead of a fixed map.
function BoxTouchTileView({ sensor }: { sensor?: Record<string, unknown> | null }) {
  const values = numberArray(sensor?.fz_0p1N);
  const hasData = values.length >= 239;
  if (!hasData) {
    return <div className="camera-tile-empty">no touch sample</div>;
  }
  const scaleMax = Math.max(1, ...values.map((value) => Math.abs(value)));
  let cursor = 0;
  return (
    <div className="box-touch-fill" aria-label="live tactile preview">
      {DEVICE_TOUCH_ROW_LENGTHS.map((length, rowIndex) => {
        const offset = Math.floor((DEVICE_TOUCH_COLUMNS - length) / 2);
        const row = values.slice(cursor, cursor + length);
        const startIndex = cursor;
        cursor += length;
        return (
          <div className="touch-row" key={rowIndex}>
            {Array.from({ length: offset }).map((_, index) => (
              <span className="touch-cell touch-cell-empty" key={`pre-${index}`} />
            ))}
            {row.map((value, index) => {
              const pointIndex = startIndex + index + 1;
              return (
                <span
                  className="touch-cell"
                  key={pointIndex}
                  title={`#${pointIndex} fz=${value.toFixed(1)} (0.1N)`}
                  style={{ backgroundColor: previewTouchColor(Math.abs(value), scaleMax) }}
                />
              );
            })}
            {Array.from({ length: DEVICE_TOUCH_COLUMNS - length - offset }).map((_, index) => (
              <span className="touch-cell touch-cell-empty" key={`post-${index}`} />
            ))}
          </div>
        );
      })}
    </div>
  );
}

const FORCE_TILE_CHANNELS: { label: string; unit: string }[] = [
  { label: "Fx", unit: "N" },
  { label: "Fy", unit: "N" },
  { label: "Fz", unit: "N" },
  { label: "Mx", unit: "N·m" },
  { label: "My", unit: "N·m" },
  { label: "Mz", unit: "N·m" },
];

// Full-frame 6D force/torque: bipolar bars centered on zero (force can push or
// pull on every axis), forces and moments scaled independently so neither
// dwarfs the other.
function BoxForceTileView({ sensor }: { sensor?: Record<string, unknown> | null }) {
  const values = numberArray(sensor?.fxyz_mxyz);
  if (values.length < 6) {
    return <div className="camera-tile-empty">no force sample</div>;
  }
  const forceMax = Math.max(1, ...values.slice(0, 3).map((value) => Math.abs(value)));
  const momentMax = Math.max(0.1, ...values.slice(3, 6).map((value) => Math.abs(value)));
  return (
    <div className="box-force-fill" aria-label="live 6D force preview">
      {FORCE_TILE_CHANNELS.map((channel, index) => {
        const value = values[index] ?? 0;
        const max = index < 3 ? forceMax : momentMax;
        const ratio = Math.max(-1, Math.min(1, value / max));
        const pct = Math.abs(ratio) * 50;
        const positive = ratio >= 0;
        return (
          <div className="force-bipolar-row" key={channel.label}>
            <span className="force-bipolar-label">{channel.label}</span>
            <div className="force-bipolar-track">
              <i
                className={positive ? "pos" : "neg"}
                style={positive ? { left: "50%", width: `${pct}%` } : { right: "50%", width: `${pct}%` }}
              />
            </div>
            <strong className="force-bipolar-value">
              {value.toFixed(2)}
              <small>{channel.unit}</small>
            </strong>
          </div>
        );
      })}
    </div>
  );
}

// Camera-tile lookalike for the array BOX sensors: the live visualization fills
// the media, and the scalar config / liveness stats live in the hover overlay,
// mirroring CameraTile so the Device Manager grid stays visually uniform.
function BoxSensorTile({ device }: { device: DeviceStatus }) {
  const [preview, setPreview] = useState<BoxPreviewPayload | null>(null);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      const next = await api.fetchBoxPreview(device.id);
      if (mounted) {
        setPreview(next);
      }
    };
    load();
    const timer = window.setInterval(load, 100);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [device.id]);

  const sensor = preview?.sensor ?? null;
  const isForce = boxSensorSuffix(device.id) === "box_six_d_force" || Array.isArray(sensor?.fxyz_mxyz);
  const staleS = preview?.staleS == null ? null : preview.staleS;
  const sampleAge = staleS == null ? "-" : `${staleS.toFixed(1)}s`;
  const queueSize = numberValue(preview?.status?.queue_size);
  const config = device.config ?? {};
  const configEntries = Object.entries(config).filter(
    ([, v]) => v != null && typeof v !== "object"
  );

  return (
    <div className="camera-tile">
      <div className="camera-tile-media box-sensor-media">
        {isForce ? <BoxForceTileView sensor={sensor} /> : <BoxTouchTileView sensor={sensor} />}
        <div className="camera-tile-overlay">
          <div className="device-live-stats">
            <Metric label="Live" value={preview?.active ? "yes" : "no"} />
            <Metric label="Age" value={sampleAge} />
            <Metric label="Queue" value={queueSize ?? "-"} />
          </div>
          <div className="device-config-grid">
            <div className="device-config-row">
              <span className="device-config-key">label</span>
              <span className="device-config-value">{device.label}</span>
            </div>
            <div className="device-config-row">
              <span className="device-config-key">timestamp</span>
              <span className="device-config-value">{String(sensor?.timestamp ?? "-")}</span>
            </div>
            {device.detail && (
              <div className="device-config-row">
                <span className="device-config-key">detail</span>
                <span className="device-config-value">{device.detail}</span>
              </div>
            )}
            {configEntries.map(([key, value]) => (
              <div className="device-config-row" key={key}>
                <span className="device-config-key">{key}</span>
                <span className="device-config-value">{String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="camera-tile-label">
        <StatusDot state={device.state} />
        <strong>{device.id}</strong>
        <span className="camera-tile-stat">{device.fps} Hz</span>
      </div>
    </div>
  );
}

// Cold-spawn (Argus open + AWB settle) can 503 for a few seconds before the
// first frame; keep retrying with backoff this many times (~15s) before
// surfacing "preview unavailable", then keep probing in case it recovers.
const PREVIEW_MAX_FAILURES = 30;

function CameraTile({ device, snapshot }: { device: DeviceStatus; snapshot: GuiSnapshot }) {
  // Idle snapshots use the gateway's temporary preview pipeline; while the
  // recorder owns the cameras, snapshots come from recorder-owned tmpfs JPEGs.
  const previewable = device.state !== "error";
  const config = device.config ?? {};
  const configEntries = Object.entries(config).filter(
    ([, v]) => v != null && typeof v !== "object"
  );

  // Snapshot polling: each request is short, so 11 tiles don't exhaust the
  // browser's ~6-connections-per-origin limit the way 11 live MJPEG streams
  // would. The loop is self-throttled off onLoad/onError, so a slow camera
  // never piles up overlapping requests.
  const [src, setSrc] = useState("");
  const [loaded, setLoaded] = useState(false);
  const failuresRef = useRef(0);
  const activeRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  const refresh = () => {
    if (activeRef.current) setSrc(`${api.cameraSnapshotUrl(device.id)}&t=${Date.now()}`);
  };
  const scheduleNext = (delayMs: number) => {
    if (timerRef.current != null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(refresh, delayMs);
  };

  useEffect(() => {
    if (!previewable) {
      activeRef.current = false;
      if (timerRef.current != null) window.clearTimeout(timerRef.current);
      setSrc("");
      setLoaded(false);
      failuresRef.current = 0;
      return;
    }
    activeRef.current = true;
    failuresRef.current = 0;
    setLoaded(false);
    refresh();
    return () => {
      activeRef.current = false;
      if (timerRef.current != null) window.clearTimeout(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewable, device.id]);

  const handleLoad = () => {
    failuresRef.current = 0;
    setLoaded(true);
    scheduleNext(200); // ~5 fps poll
  };
  const handleError = () => {
    failuresRef.current += 1;
    setLoaded(false);
    scheduleNext(failuresRef.current >= PREVIEW_MAX_FAILURES ? 2000 : 500);
  };

  const placeholder = device.state === "error"
    ? "no signal"
    : failuresRef.current >= PREVIEW_MAX_FAILURES
      ? "preview unavailable"
      : device.state;

  return (
    <div className="camera-tile">
      <div className="camera-tile-media">
        {previewable && src ? (
          <img
            className="camera-tile-img"
            src={src}
            alt={`${device.id} preview`}
            style={{ display: loaded ? "block" : "none" }}
            onLoad={handleLoad}
            onError={handleError}
          />
        ) : null}
        {(!previewable || !loaded) && (
          <div className="camera-tile-empty">{placeholder}</div>
        )}
        <div className="camera-tile-overlay">
          <div className="device-config-grid">
            <div className="device-config-row">
              <span className="device-config-key">fps</span>
              <span className="device-config-value">{device.fps}</span>
            </div>
            <div className="device-config-row">
              <span className="device-config-key">latency</span>
              <span className="device-config-value">{device.latencyMs} ms</span>
            </div>
            {device.detail && (
              <div className="device-config-row">
                <span className="device-config-key">detail</span>
                <span className="device-config-value">{device.detail}</span>
              </div>
            )}
            {configEntries.map(([key, value]) => (
              <div className="device-config-row" key={key}>
                <span className="device-config-key">{key}</span>
                <span className="device-config-value">{String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="camera-tile-label">
        <StatusDot state={device.state} />
        <strong>{device.id}</strong>
        <span className="camera-tile-stat">{device.fps} fps</span>
      </div>
    </div>
  );
}

// Expandable scalar-sensor row used for every non-camera device (and the BOX
// gripper/IMU/trigger). Factored out so the box_collection section can mix tile
// and row rendering without duplicating the markup.
function DeviceRow({
  device,
  isExpanded,
  onToggle,
}: {
  device: DeviceStatus;
  isExpanded: boolean;
  onToggle: (id: string) => void;
}) {
  const config = device.config ?? {};
  const configEntries = Object.entries(config).filter(
    ([, v]) => v != null && typeof v !== "object"
  );
  return (
    <div className={`device-manager-row ${isExpanded ? "device-manager-row-active" : ""}`}>
      <button className="device-manager-header" onClick={() => onToggle(device.id)}>
        <div className="row-title">
          <StatusDot state={device.state} />
          <strong>{device.id}</strong>
        </div>
        <div className="device-stats">
          <span>{device.fps} fps</span>
          <span>{device.latencyMs} ms</span>
          <small>{device.detail}</small>
          <small>{isExpanded ? "close" : "open"}</small>
        </div>
      </button>
      {isExpanded && (
        <div className="device-config-detail device-config-detail-expanded">
          {configEntries.length > 0 && (
            <div className="device-config-grid">
              {configEntries.map(([key, value]) => (
                <div className="device-config-row" key={key}>
                  <span className="device-config-key">{key}</span>
                  <span className="device-config-value">{String(value)}</span>
                </div>
              ))}
            </div>
          )}
          <DeviceInlinePreview device={device} />
        </div>
      )}
    </div>
  );
}

function DeviceManagerPage({ snapshot }: { snapshot: GuiSnapshot }) {
  const [hideErrors, setHideErrors] = useState(false);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  const devices = hideErrors
    ? snapshot.devices.filter((d) => d.state !== "error")
    : snapshot.devices;

  const toggleExpanded = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const grouped = useMemo(() => {
    return devices.reduce<Record<string, typeof devices>>((acc, d) => {
      acc[d.kind] = [...(acc[d.kind] ?? []), d];
      return acc;
    }, {});
  }, [devices]);

  const onlineCount = snapshot.devices.filter((d) => d.state === "running").length;
  const errorCount = snapshot.devices.filter((d) => d.state === "error").length;

  const kindLabel = (kind: string): string => {
    if (kind === "camera") return snapshot.configSummary.rigType === "gmsl2" ? "GMSL2 Cameras" : "Cameras";
    if (kind === "box_collection") return "BOX Sensors";
    if (kind === "tactile") return "Tactile Sensors";
    if (kind === "handheld_gripper") return "Handheld Grippers";
    return kind.replace("_", " ");
  };

  return (
    <div className="page-stack">
      <PageHeader title="Device Manager" subtitle="view connected hardware devices, configuration details, and connection status" />
      <section className="panel">
        <div className="panel-heading">
          <h2>Summary</h2>
          <span>{snapshot.devices.length} devices</span>
        </div>
        <div className="summary-grid">
          <Metric label="Total" value={snapshot.devices.length} />
          <Metric label="Online" value={onlineCount} />
          <Metric label="Error" value={errorCount} />
        </div>
        <div className="control-row">
          <label className="annotation-field annotation-toggle">
            <input type="checkbox" checked={hideErrors} onChange={(e) => setHideErrors(e.target.checked)} />
            <span>Hide error devices</span>
          </label>
        </div>
      </section>
      {Object.entries(grouped).map(([kind, items]) => (
        <section className="panel" key={kind}>
          <div className="panel-heading">
            <h2>{kindLabel(kind)}</h2>
            <span>{items.filter((d) => d.state === "running").length}/{items.length} online</span>
          </div>
          {kind === "camera" ? (
            <div className="camera-grid">
              {items.map((device) => (
                <CameraTile key={device.id} device={device} snapshot={snapshot} />
              ))}
            </div>
          ) : kind === "box_collection" ? (
            <>
              {items.some((d) => isVisualBoxSensor(d.id)) && (
                <div className="camera-grid box-sensor-grid">
                  {items
                    .filter((d) => isVisualBoxSensor(d.id))
                    .map((device) => (
                      <BoxSensorTile key={device.id} device={device} />
                    ))}
                </div>
              )}
              {items
                .filter((d) => !isVisualBoxSensor(d.id))
                .map((device) => (
                  <DeviceRow
                    key={device.id}
                    device={device}
                    isExpanded={expandedIds.has(device.id)}
                    onToggle={toggleExpanded}
                  />
                ))}
            </>
          ) : (
            items.map((device) => (
              <DeviceRow
                key={device.id}
                device={device}
                isExpanded={expandedIds.has(device.id)}
                onToggle={toggleExpanded}
              />
            ))
          )}
        </section>
      ))}
    </div>
  );
}

function PlaceholderPage({ title }: { title: string }) {
  return (
    <div className="page-stack">
      <PageHeader title={title} subtitle="placeholder page reserved for the data factory workflow" />
      <section className="panel placeholder-panel">
        <h2>{title}</h2>
        <p>This page is intentionally a placeholder in this pass.</p>
      </section>
    </div>
  );
}

function App() {
  const [snapshot, setSnapshot] = useState<GuiSnapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const [activePage, setActivePage] = useState<PageId>(() => pageFromHash());
  const loadingRef = useRef(false);

  useEffect(() => {
    const onHashChange = () => setActivePage(pageFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    loadingRef.current = true;
    api.getSnapshot().then((next) => {
      if (loadingRef.current) {
        setSnapshot(next);
      }
    });
    return () => {
      loadingRef.current = false;
    };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => setSnapshot(api.tick()), 250);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const timer = window.setInterval(async () => {
      const next = await api.getSnapshot();
      if (!cancelled) {
        setSnapshot(next);
      }
    }, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  async function run(command: () => Promise<GuiSnapshot>) {
    setBusy(true);
    try {
      setSnapshot(await command());
    } finally {
      setBusy(false);
    }
  }

  if (!snapshot) {
    return <main className="loading">Loading GUI shell</main>;
  }

  function navigate(page: PageId) {
    window.location.hash = page;
    setActivePage(page);
  }

  const activeMeta = pages.find((page) => page.id === activePage) ?? mvpPages[0];

  async function selectAndOpenReplay(path: string) {
    await run(() => api.selectRecordedDataset(path));
    navigate("episode-replay");
  }

  async function queueTrajGenAndOpenProcessing(path: string) {
    await run(() => api.queueTrajGen(path));
    navigate("dataset-processing");
  }

  const latestRecordedPath =
    snapshot.recordedDatasets.find((dataset) => dataset.isLatest)?.path ??
    snapshot.recordedDatasets[0]?.path ??
    snapshot.recording.datasetRoot;
  const firstMissingPath =
    snapshot.processing.find((item) => item.status === "pose_missing")?.path ?? latestRecordedPath;

  const activeReplayPath = snapshot.replay.datasetRoot ?? snapshot.replay.dataset;
  const replayMatch =
    snapshot.processing.find((item) => item.path === activeReplayPath) ?? snapshot.processing[0];
  const startReplay = (realRobot: boolean) => {
    if (realRobot) {
      const ok = window.confirm(
        `Start real-robot replay for episode ${snapshot.replay.episode}? MuJoCo validation is current for this dataset.`
      );
      if (!ok) {
        return;
      }
    }
    run(() => api.startReplay(realRobot));
  };

  const exportTaskWithQcGuard = (taskId: string) => {
    const task = (snapshot.tasks ?? []).find((item) => item.id === taskId);
    if (!task) {
      run(() => api.exportTask(taskId));
      return;
    }
    const { notQcPassed, missingEpisodeCount } = taskNeedsQcExportConfirmation(task, snapshot.processing);
    if (missingEpisodeCount > 0 || notQcPassed.length > 0) {
      const statusLines = notQcPassed
        .slice(0, 4)
        .map((item) => `${item.name}: ${processingStatusLabel[item.status]}`);
      const moreCount = Math.max(notQcPassed.length - 4, 0);
      const confirmLines = [
        `Task "${task.name}" has dataset sessions that are not QC-passed.`,
        "",
        missingEpisodeCount > 0 ? `${missingEpisodeCount} recorded episode(s) have no QC record.` : "",
        ...statusLines,
        moreCount > 0 ? `...and ${moreCount} more` : "",
        "",
        "Continue exporting LeRobot v3?"
      ].filter((line) => line !== "");
      const ok = window.confirm(confirmLines.join("\n"));
      if (!ok) {
        return;
      }
    }
    run(() => api.exportTask(taskId));
  };

  const pageNode =
    activePage === "live-record" ? (
      <LiveRecordPage
        snapshot={snapshot}
        busy={busy}
        onConnect={() => run(() => api.connectRecording())}
        onStart={() => run(() => api.startRecording())}
        onStop={(action) => run(() => api.stopRecording(action))}
        onOpenInReplay={() => selectAndOpenReplay(latestRecordedPath)}
        onQueueTrajGen={() => queueTrajGenAndOpenProcessing(firstMissingPath)}
        onGoToProcessing={() => navigate("dataset-processing")}
        onRunCalibration={() => run(() => api.runCalibration())}
        onClearActiveTask={() => run(() => api.setActiveTask(""))}
      />
    ) : activePage === "dataset-processing" ? (
      <DatasetProcessingPage
        snapshot={snapshot}
        busy={busy}
        onGenerate={(path) => run(() => api.queueTrajGen(path))}
        onRunQc={(path) => run(() => api.runQc(path))}
        onOpenReplay={(path) => selectAndOpenReplay(path)}
        onSetDatasetsRoot={(path) => run(() => api.setDatasetsRoot(path))}
      />
    ) : activePage === "episode-replay" ? (
      <EpisodeReplayPage
        snapshot={snapshot}
        busy={busy}
        onPreflight={() => run(() => api.preflightReplay())}
        onReplay={startReplay}
        onMujocoReplay={() => run(() => api.startMujocoReplay())}
        onAbort={() => run(() => api.abortReplay())}
        onSelectDataset={(path) => run(() => api.selectRecordedDataset(path))}
        onSelectEpisode={(episode) => run(() => api.selectReplayEpisode(episode))}
        onGenerateForActive={() => replayMatch && queueTrajGenAndOpenProcessing(replayMatch.path)}
        onOpenProcessing={() => navigate("dataset-processing")}
        onSaveAnnotation={(annotation) => run(() => api.saveEpisodeAnnotation(annotation))}
      />
    ) : activePage === "qc-report" ? (
      <QcReportPage snapshot={snapshot} />
    ) : activePage === "dataset-export" ? (
      <DatasetExportPage
        snapshot={snapshot}
        busy={busy}
        onPrepare={(target) => run(() => api.prepareDatasetExport(target))}
        onExport={() => run(() => api.startDatasetExport())}
        onExportTask={exportTaskWithQcGuard}
        onOpenProcessing={() => navigate("dataset-processing")}
      />
    ) : activePage === "task-library" ? (
      <TaskLibraryPage
        snapshot={snapshot}
        busy={busy}
        onCreate={(task) => run(() => api.createTask(task))}
        onUpdate={(task) => run(() => api.updateTask(task))}
        onDelete={(id) => run(() => api.deleteTask(id))}
        onActivate={(id) => run(() => api.setActiveTask(id))}
        onNavigate={navigate}
      />
    ) : activePage === "device-manager" ? (
      <DeviceManagerPage snapshot={snapshot} />
    ) : (
      <PlaceholderPage title={activeMeta.label} />
    );

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Robot Data Factory</h1>
          <p>
            {snapshot.configSummary.rigType === "gmsl2"
              ? `Thor GMSL2 · ${snapshot.devices.filter((d) => d.kind === "camera").length} cameras · ${snapshot.configSummary.fps} fps`
              : "Live collection, processing, replay, and export for LeRobot data workflows"}
          </p>
        </div>
        <div className="topbar-status">
          <span><StatusDot state={snapshot.gateway.state === "online" ? "running" : "warning"} /> Gateway {snapshot.gateway.state}</span>
          <span><StatusDot state={snapshot.recording.state} /> Recorder</span>
          {snapshot.configSummary.hardwareSync && (
            <span>
              <StatusDot state={snapshot.configSummary.hardwareSync.enabled ? "running" : "warning"} />
              {" "}HW Sync {snapshot.configSummary.hardwareSync.enabled ? "ON" : "OFF"}
            </span>
          )}
          <span><StatusDot state={snapshot.replay.safety === "fault" ? "error" : snapshot.replay.safety === "active" ? "running" : "idle"} /> Replay safety {snapshot.replay.safety}</span>
        </div>
      </header>
      <div className="factory-layout">
        <SidebarNav activePage={activePage} onNavigate={navigate} />
        <section className="page-content">{pageNode}</section>
      </div>
    </main>
  );
}

function SidebarNav({ activePage, onNavigate }: { activePage: PageId; onNavigate: (page: PageId) => void }) {
  const [showDeferred, setShowDeferred] = useState(false);
  return (
    <nav className="sidebar" aria-label="Robot data factory pages">
      <div className="nav-group">
        <small className="nav-group-label">MVP</small>
        {mvpPages.map((page) => (
          <button
            className={page.id === activePage ? "nav-item active" : "nav-item"}
            key={page.id}
            onClick={() => onNavigate(page.id)}
          >
            <span>{page.label}</span>
          </button>
        ))}
      </div>
      <div className="nav-group">
        <button className="nav-toggle" onClick={() => setShowDeferred((value) => !value)}>
          <span>Deferred</span>
          <small>{showDeferred ? "hide" : `${deferredPages.length} hidden`}</small>
        </button>
        {showDeferred
          ? deferredPages.map((page) => (
              <button
                className={page.id === activePage ? "nav-item active deferred" : "nav-item deferred"}
                key={page.id}
                onClick={() => onNavigate(page.id)}
              >
                <span>{page.label}</span>
                <small>deferred</small>
              </button>
            ))
          : null}
      </div>
    </nav>
  );
}

export default App;
