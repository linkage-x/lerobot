import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GuiSnapshot } from "../api";
import type { BoxPreviewPayload, BoxCaliLog, BoxCaliLogLine, CollectionTask, ConfigSummary, DeviceStatus, EpisodeAnnotation, EventLogItem, ProcessingItem, ProcessingStatus, RecordedDataset, RecordingStatus, ReplayStatus, SubtaskSegment, TaskStatus, DatasetExportStatus, AnnotationOutcome, AnnotationQuality, ReviewStatus, MujocoCubeMode, RealCubeMode, RealEndEffectorMode, RealSensePreviewStatus } from "../types";
import { StatusDot, Metric, PageHeader, stateLabel, QualityOverview, processingStatusLabel, datasetNamePrefixes, taskDatasetBaseName, processingItemsForTask, taskNeedsQcExportConfirmation } from "../shared/ui";
import { ReplayInspector } from "../ReplayInspector";
import { api } from "../apiClient";

export function ReplayPanel({
  status,
  busy,
  onPreflight,
  onReplay,
  mujocoMode,
  onAbort
}: {
  status: ReplayStatus;
  busy: boolean;
  onPreflight: () => void;
  onReplay: (realRobot: boolean) => void;
  mujocoMode: MujocoCubeMode;
  onAbort: () => void;
}) {
  const isActive = status.state === "dry_run" || status.state === "sim_replay" || status.state === "replaying";
  const canReplayData = status.dataStatus === "loaded" && (status.recordedFrames ?? status.totalFrames) > 0;
  const validation = status.mujocoValidation;
  const validationMatchesMode = (validation?.cubeMode ?? "left") === mujocoMode;
  const mujocoPassed =
    validation?.status === "passed" &&
    validation.isCurrentForSelection === true &&
    validationMatchesMode;
  const validationLabel =
    validation?.status === "passed" && !validationMatchesMode
      ? "recommended"
      : validation?.status === "passed"
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
    ? `MuJoCo ${mujocoMode} replay is current for this episode.`
    : validation?.status === "passed" && !validationMatchesMode
      ? `The saved result is for ${validation.cubeMode ?? "left"}; run ${mujocoMode} before Real Robot.`
      : "Strongly recommended before Preflight/Dry Run; required before Real Robot.";
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

function validIpv4(value: string): boolean {
  const parts = value.trim().split(".");
  return parts.length === 4 && parts.every((part) => /^\d+$/.test(part) && Number(part) >= 0 && Number(part) <= 255);
}

export function RealRobotReplayPanel({
  status,
  busy,
  onStart
}: {
  status: ReplayStatus;
  busy: boolean;
  onStart: (mode: RealCubeMode, robotIp: string, endEffectorMode: RealEndEffectorMode) => void;
}) {
  const [mode, setMode] = useState<RealCubeMode>(status.realCubeMode ?? "right");
  const [robotIp, setRobotIp] = useState(status.realRobotIp ?? "");
  const [endEffectorMode, setEndEffectorMode] = useState<RealEndEffectorMode>(
    status.realEndEffectorMode ?? "corenetic_gripper_ee"
  );
  const [monitorRequested, setMonitorRequested] = useState(status.state === "replaying");
  const [cameraStatus, setCameraStatus] = useState<RealSensePreviewStatus | null>(null);
  const [frameKey, setFrameKey] = useState(0);
  const validation = status.mujocoValidation;
  const validationMatches =
    validation?.status === "passed" &&
    validation.isCurrentForSelection === true &&
    (validation.cubeMode ?? status.mujocoCubeMode) === mode;
  const ipValid = validIpv4(robotIp);
  const active = status.state === "replaying" || status.state === "sim_replay" || status.state === "dry_run";
  const disabled = busy || active || status.datasetKind === "exported" || !validationMatches || !ipValid;

  useEffect(() => {
    if (!monitorRequested && status.state !== "replaying") return;
    let mounted = true;
    const poll = async () => {
      const next = await api.fetchRealSenseStatus();
      if (mounted && next) setCameraStatus(next);
    };
    poll();
    const statusTimer = window.setInterval(poll, 700);
    const frameTimer = window.setInterval(() => setFrameKey(Date.now()), 250);
    return () => {
      mounted = false;
      window.clearInterval(statusTimer);
      window.clearInterval(frameTimer);
    };
  }, [monitorRequested, status.state]);

  const run = () => {
    const confirmed = window.confirm(
      `Start REAL ROBOT replay?\n\n` +
      `Dataset: ${status.datasetRoot ?? status.dataset}\n` +
      `Episode: ${status.episode}\n` +
      `Cube: ${mode}\n` +
      `Robot IP: ${robotIp.trim()}\n\n` +
      `Tracked frame: ${endEffectorMode}\n\n` +
      `MuJoCo validation must match this exact selection. Keep an operator at the robot and be ready to use the physical emergency stop.`
    );
    if (!confirmed) return;
    setMonitorRequested(true);
    setCameraStatus({ available: null, running: true, error: "Connecting to the first available RealSense…" });
    onStart(mode, robotIp.trim(), endEffectorMode);
  };

  return (
    <section className="panel real-robot-panel">
      <div className="panel-heading">
        <h2>Real Robot Replay</h2>
        <span>single-cube opencv_kalibr replay · automatic RealSense monitor</span>
      </div>
      <div className="real-robot-layout">
        <div className="real-robot-settings">
          <div className="mujoco-mode-picker" role="group" aria-label="Real robot cube trajectory">
            {(["left", "right"] as RealCubeMode[]).map((candidate) => (
              <button
                key={candidate}
                className={mode === candidate ? "active" : ""}
                disabled={busy || active}
                onClick={() => setMode(candidate)}
                type="button"
              >
                {`${candidate[0].toUpperCase()}${candidate.slice(1)} cube`}
              </button>
            ))}
          </div>
          <div className="mujoco-mode-picker" role="group" aria-label="Real robot end-effector frame">
            <button
              className={endEffectorMode === "corenetic_gripper_ee" ? "active" : ""}
              disabled={busy || active}
              onClick={() => setEndEffectorMode("corenetic_gripper_ee")}
              type="button"
            >
              Corenetic gripper EE
            </button>
            <button
              className={endEffectorMode === "fr3_ee" ? "active" : ""}
              disabled={busy || active}
              onClick={() => setEndEffectorMode("fr3_ee")}
              type="button"
            >
              Bare FR3 · fr3_ee
            </button>
          </div>
          <label className="real-robot-ip-field">
            <span>Robot IP for {mode} trajectory</span>
            <input value={robotIp} onChange={(event) => setRobotIp(event.target.value)} placeholder="192.168.x.x" />
          </label>
          <button className="danger real-robot-run" disabled={disabled} onClick={run} type="button">
            {status.state === "replaying" ? "Real-robot replay running…" : "Run real-robot replay"}
          </button>
          <p className="panel-note">
            {status.datasetKind === "exported"
              ? "Real robot replay is disabled for exported datasets."
              : !validationMatches
                ? `Run and pass MuJoCo ${mode} for this dataset and episode first.`
                : !ipValid
                  ? "Enter a valid robot IPv4 address."
                  : `The gateway preflights this IP, moves ${endEffectorMode} to frame 0, then streams the trajectory.`}
          </p>
        </div>
        <div className="realsense-monitor">
          <div className="realsense-monitor-heading">
            <strong>RealSense live monitor</strong>
            <span>{cameraStatus?.serial ? `S/N ${cameraStatus.serial}` : "first available camera"}</span>
          </div>
          {cameraStatus?.available && cameraStatus.running ? (
            <img src={api.realSenseSnapshotUrl(frameKey)} alt="RealSense live view during real robot replay" />
          ) : (
            <div className="realsense-monitor-empty">
              {cameraStatus?.error || "The camera connects automatically when real-robot replay starts."}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export function EventLog({ events }: { events: EventLogItem[] }) {
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

export function RecordedDatasetList({
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
        <h2>Replay Datasets</h2>
        <span>{datasets.length} found</span>
      </div>
      {latest ? (
        <div className="latest-dataset">
          <span>Default replay target</span>
          <strong>{latest.name}</strong>
          <small>{latest.path}</small>
        </div>
      ) : (
        <div className="empty-dataset-list">No replay datasets found under the configured dataset or exports roots.</div>
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
                  {dataset.datasetKind === "exported" ? <em>exported</em> : null}
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


export function ReplayReadinessCard({
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

export function SubtaskSegmentEditor({
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

export function EpisodeAnnotationPanel({
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

export function EpisodeSelector({
  status,
  annotation,
  busy,
  onSelectEpisode,
  onDeleteEpisode
}: {
  status: ReplayStatus;
  annotation?: EpisodeAnnotation;
  busy: boolean;
  onSelectEpisode: (episode: number) => void;
  onDeleteEpisode: (episode: number) => void;
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
  // Guard against wiping the dataset: the backend also refuses, but hide the
  // action entirely when only one episode is left so it can't be attempted.
  const canDelete = !busy && currentIndex >= 0 && options.length > 1;

  const requestDelete = () => {
    if (!canDelete) {
      return;
    }
    const confirmed = window.confirm(
      `Delete episode ${current}? This permanently removes its frames and videos from disk and ` +
        `renumbers the remaining ${options.length - 1} episode(s). This cannot be undone.`
    );
    if (confirmed) {
      // Don't touch pendingEpisode here: that state tracks episode *switches* and
      // deleting the tail lands on a different index, which would leave it stuck.
      // The busy flag already disables the controls while the delete runs.
      onDeleteEpisode(current);
    }
  };
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
        <button
          className="danger"
          disabled={!canDelete}
          title="Delete this episode from disk and reindex the rest"
          onClick={requestDelete}
        >
          Delete
        </button>
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

export function EpisodeReplayPage({
  snapshot,
  busy,
  onPreflight,
  onReplay,
  onMujocoReplay,
  onApproveMujoco,
  onRealReplay,
  onAbort,
  onSelectDataset,
  onSelectEpisode,
  onDeleteEpisode,
  onGenerateForActive,
  onOpenProcessing,
  onSaveAnnotation
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onPreflight: () => void;
  onReplay: (realRobot: boolean) => void;
  onMujocoReplay: (mode: MujocoCubeMode) => void;
  onApproveMujoco: (mode: MujocoCubeMode) => void;
  onRealReplay: (mode: RealCubeMode, robotIp: string, endEffectorMode: RealEndEffectorMode) => void;
  onAbort: () => void;
  onSelectDataset: (path: string) => void;
  onSelectEpisode: (episode: number) => void;
  onDeleteEpisode: (episode: number) => void;
  onGenerateForActive: () => void;
  onOpenProcessing: () => void;
  onSaveAnnotation: (annotation: EpisodeAnnotation) => void;
}) {
  const [mujocoMode, setMujocoMode] = useState<MujocoCubeMode>(snapshot.replay.mujocoCubeMode ?? "left");
  const activePath = snapshot.replay.datasetRoot ?? snapshot.replay.dataset;
  const matchingProcessing =
    snapshot.processing.find((item) => item.path === activePath) ?? null;
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
          mujocoMode={mujocoMode}
          onAbort={onAbort}
        />
      </div>
      <EpisodeSelector status={snapshot.replay} annotation={snapshot.annotation} busy={busy} onSelectEpisode={onSelectEpisode} onDeleteEpisode={onDeleteEpisode} />
      <ReplayInspector
        api={api}
        datasetPath={activePath}
        episode={snapshot.replay.episode}
        fallbackFps={snapshot.replay.fps}
        revision={snapshot.replay.revision ?? 0}
        mujocoMode={mujocoMode}
        onMujocoModeChange={setMujocoMode}
        onRunMujoco={onMujocoReplay}
        onApproveMujoco={onApproveMujoco}
        replayStatus={snapshot.replay}
        busy={busy}
        mujocoRefreshKey={`${snapshot.replay.mujocoValidation?.updatedAt ?? ""}:${snapshot.replay.state}`}
      />
      <RealRobotReplayPanel status={snapshot.replay} busy={busy} onStart={onRealReplay} />
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
