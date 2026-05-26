import { useEffect, useMemo, useRef, useState } from "react";
import { DataCollectionGuiApi, type GuiSnapshot } from "./api";
import { ReplayInspector } from "./ReplayInspector";
import type {
  ConfigSummary,
  DeviceStatus,
  EpisodeAnnotation,
  EventLogItem,
  ProcessingItem,
  ProcessingStatus,
  RecordedDataset,
  RecordingStatus,
  ReplayStatus
} from "./types";
import "./styles.css";

const api = new DataCollectionGuiApi();

function stateLabel(state: string) {
  return state.replace("_", " ");
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

function RecordingPanel({
  status,
  config,
  busy,
  onConnect,
  onStart,
  onStop
}: {
  status: RecordingStatus;
  config: ConfigSummary;
  busy: boolean;
  onConnect: () => void;
  onStart: () => void;
  onStop: (action: "save" | "discard" | "exit") => void;
}) {
  const progress = Math.round((status.frameIndex / Math.max(status.targetFrames, 1)) * 100);
  const isConnected = status.pid != null || ["connecting", "armed", "recording", "review", "saving", "discarding"].includes(status.state);
  const canStartEpisode = status.state === "armed";
  const canResolveEpisode = status.state === "recording" || status.state === "review";
  const canExit = isConnected;
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
        <button disabled={busy || isConnected} onClick={onConnect}>Connect</button>
        <button disabled={busy || !canStartEpisode} onClick={onStart}>StartEpisode</button>
        <button disabled={busy || !canResolveEpisode} onClick={() => onStop("save")}>Save</button>
        <button disabled={busy || !canResolveEpisode} onClick={() => onStop("discard")}>Discard</button>
        <button disabled={busy || !canExit} onClick={() => onStop("exit")}>Exit</button>
      </div>
      <div className="summary-grid">
        <Metric label="Frame" value={`${status.frameIndex}/${status.targetFrames}`} />
        <Metric label="Queue" value={status.queueDepth} />
        <Metric label="Saved" value={status.savedEpisodes} />
        <Metric label="PID" value={status.pid ?? "none"} />
      </div>
      <p className="panel-note">{status.message}</p>
      {status.lastOutput ? <p className="process-output">{status.lastOutput}</p> : null}
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
  { id: "dataset-export", label: "Dataset Export", kind: "mvp" }
];

const deferredPages: PageMeta[] = [
  { id: "dashboard", label: "Dashboard", kind: "deferred" },
  { id: "qc-report", label: "QC Report", kind: "deferred" },
  { id: "model-evaluation", label: "Model Evaluation", kind: "deferred" },
  { id: "device-manager", label: "Device Manager", kind: "deferred" },
  { id: "task-library", label: "Task Library", kind: "deferred" },
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
  onRunCalibration
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
}) {
  const showSavedBanner = snapshot.recording.savedEpisodes > 0;
  return (
    <div className="page-stack">
      <PageHeader
        title="Live Record"
        subtitle={snapshot.configSummary.rigType === "gmsl2"
          ? `GMSL2 ${snapshot.devices.filter((d) => d.kind === "camera").length}-camera capture with${snapshot.configSummary.hardwareSync?.enabled ? "" : "out"} hardware sync`
          : "capture raw multi-camera handheld data; post-processing lives on the Processing page"}
      />
      <div className="split-layout">
        <RecordingPanel status={snapshot.recording} config={snapshot.configSummary} busy={busy} onConnect={onConnect} onStart={onStart} onStop={onStop} />
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

function EpisodeAnnotationPanel({
  annotation,
  datasetPath,
  busy,
  onSave
}: {
  annotation: EpisodeAnnotation;
  datasetPath: string;
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

  return (
    <section className="panel annotation-panel">
      <div className="panel-heading">
        <h2>Episode Annotation</h2>
        <span>{draft.updatedAt ? `saved ${new Date(draft.updatedAt).toLocaleString()}` : "not saved"}</span>
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
      </div>
      <div className="control-row">
        <button
          disabled={busy || !datasetPath || !canSave}
          onClick={() =>
            onSave({
              ...draft,
              datasetRoot: datasetPath,
              tags: normalizedTags,
              taskPrompt: draft.taskPrompt.trim(),
              annotator: draft.annotator.trim(),
              notes: draft.notes.trim()
            })
          }
        >
          Save Annotation
        </button>
        <span className="annotation-hint">Episode {draft.episode} · {draft.source}</span>
      </div>
    </section>
  );
}

function EpisodeSelector({
  status,
  busy,
  onSelectEpisode
}: {
  status: ReplayStatus;
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
      <EpisodeSelector status={snapshot.replay} busy={busy} onSelectEpisode={onSelectEpisode} />
      <ReplayInspector api={api} datasetPath={activePath} episode={snapshot.replay.episode} fallbackFps={snapshot.replay.fps} />
      <EpisodeAnnotationPanel
        annotation={snapshot.annotation}
        datasetPath={activePath}
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
    ) : item.status === "qc_failed" || item.status === "error" ? (
      <button disabled={busy} onClick={onRunQc}>Run QC</button>
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
                Gateway not connected (state: {snapshot.gateway.state}). Start
                <code> python -m tools.data_collection_gui.gateway</code>
                and point the Vite proxy at it via <code>GUI_API_TARGET</code>.
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
                disabled={busy || !["pose_ready", "qc_pass", "qc_failed", "error"].includes(selected.status)}
                onClick={() => onRunQc(selected.path)}
              >
                {selected.status === "qc_failed" || selected.status === "error" ? "Re-run QC" : "Run QC"}
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
  onOpenProcessing
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onPrepare: (target: GuiSnapshot["datasetExport"]["target"]) => void;
  onExport: () => void;
  onOpenProcessing: () => void;
}) {
  const exportStatus = snapshot.datasetExport;
  const eligible = snapshot.processing.filter((item) => item.status === "qc_pass");
  const hasEligible = eligible.length > 0;
  return (
    <div className="page-stack">
      <PageHeader title="Dataset Export" subtitle="package QC-approved datasets into LeRobot, MCAP, or Parquet bundles" />
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
        onOpenProcessing={() => navigate("dataset-processing")}
      />
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
