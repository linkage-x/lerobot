import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GuiSnapshot } from "../api";
import type { BoxPreviewPayload, BoxCaliLog, BoxCaliLogLine, CollectionTask, ConfigSummary, DeviceStatus, EpisodeAnnotation, EventLogItem, ProcessingItem, ProcessingStatus, RecordedDataset, RecordingStatus, ReplayStatus, SubtaskSegment, TaskStatus, DatasetExportStatus, AnnotationOutcome, AnnotationQuality, ReviewStatus, MujocoCubeMode, RealCubeMode, RealEndEffectorMode, RealSensePreviewStatus, RealSensePreviewCameraStatus, ReplayTimeline } from "../types";
import { StatusDot, Metric, PageHeader, stateLabel, QualityOverview, processingStatusLabel, datasetNamePrefixes, taskDatasetBaseName, processingItemsForTask, taskNeedsQcExportConfirmation, mujocoValidationMatchesSelection } from "../shared/ui";
import { ReplayInspector } from "../ReplayInspector";
import { api } from "../apiClient";

export function ReplayPanel({
  status,
  busy,
  onPreflight,
  mujocoMode,
  cubeSelection = true,
  onAbort
}: {
  status: ReplayStatus;
  busy: boolean;
  onPreflight: () => void;
  mujocoMode: MujocoCubeMode;
  cubeSelection?: boolean;
  onAbort: () => void;
}) {
  const isActive = status.state === "sim_replay" || status.state === "replaying";
  const canReplayData = status.dataStatus === "loaded" && (status.recordedFrames ?? status.totalFrames) > 0;
  const validation = status.mujocoValidation;
  const validationMatchesMode = mujocoValidationMatchesSelection(validation?.cubeMode, mujocoMode, cubeSelection);
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
  // One line each: what the verdict is, what to do next, what it measured. These used to run
  // together in a single paragraph where the label and the gateway's own message said the same
  // thing twice and the numbers trailed off the end.
  const validationGuidance = mujocoPassed
    ? "Nothing to do; this episode's validation is current."
    : validation?.status === "passed" && !validationMatchesMode
      ? `The saved result is for ${validation.cubeMode ?? "left"}; run ${mujocoMode} before Real Robot.`
      : validation?.status === "failed"
        ? "Inspect the trajectory below, then re-run MuJoCo."
        : validation?.status === "running"
          ? "Waiting for the run to finish."
          : "Run MuJoCo before real-robot replay.";
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
        <button disabled={busy || !isActive} onClick={onAbort}>Abort</button>
      </div>
      <p className="panel-note">Safety {status.safety} · {status.message}</p>
      <p className={`validation-note validation-${validationLabel}`}>
        <strong>MuJoCo replay {validationLabel}</strong> · {validationGuidance}
        <br />
        {validationMetric}
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
  onStart: (mode: RealCubeMode, robotIp: string, endEffectorMode: RealEndEffectorMode, overrideMujocoFailure: boolean) => void;
}) {
  const validation = status.mujocoValidation;
  const validationMode = validation?.cubeMode ?? status.mujocoCubeMode;
  const mode: RealCubeMode = validationMode === "left" ? "left" : "right";
  const endEffectorMode: RealEndEffectorMode = "pika_gripper_ee";
  const [robotIp, setRobotIp] = useState(status.realRobotIp || "192.168.1.206");
  const [monitorRequested, setMonitorRequested] = useState(status.state === "replaying");
  const [overridePromptOpen, setOverridePromptOpen] = useState(false);
  const [cameraStatus, setCameraStatus] = useState<RealSensePreviewStatus | null>(null);
  const [frameKey, setFrameKey] = useState(0);
  const [timeline, setTimeline] = useState<ReplayTimeline | null>(null);
  const recordedVideoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  const validationPassed =
    validation?.status === "passed" &&
    validation.isCurrentForSelection === true &&
    (validation.cubeMode ?? status.mujocoCubeMode) === mode;
  const validationFailedButReviewable =
    validation?.status === "failed" &&
    validation.exitCode === 0 &&
    validation.hasStructuredResult === true &&
    validation.completedFrames > 0 &&
    validation.completedFrames >= validation.totalFrames &&
    validation.datasetRoot === (status.datasetRoot ?? status.dataset) &&
    validation.episode === status.episode &&
    validation.fps === status.fps &&
    (validation.cubeMode ?? status.mujocoCubeMode) === mode;
  const validationDecisionAvailable = validationPassed || validationFailedButReviewable;
  const ipValid = validIpv4(robotIp);
  const active = status.state === "replaying" || status.state === "sim_replay";
  const disabled =
    busy || active || status.datasetKind === "exported" || !validationDecisionAvailable || !ipValid;
  const datasetPath = status.datasetRoot || status.dataset;
  const timelineCameraKeys = timeline?.cameraKeys ?? [];
  const cameraMatches = (cameraStatus?.cameras ?? []).filter((camera): camera is RealSensePreviewCameraStatus =>
    Boolean(camera.cameraKey && timelineCameraKeys.includes(camera.cameraKey))
  );
  const recordedVideoUrl = (cameraKey: string) => datasetPath ? api.videoUrl(datasetPath, cameraKey, status.episode) : "";
  const shortCameraName = (cameraKey: string) => cameraKey.split(".").pop() ?? cameraKey;

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

  useEffect(() => {
    if (!datasetPath || status.datasetKind === "exported") {
      setTimeline(null);
      return;
    }
    let mounted = true;
    api.fetchReplayTimeline(datasetPath, status.episode).then((next) => {
      if (mounted) setTimeline(next);
    });
    return () => {
      mounted = false;
    };
  }, [datasetPath, status.datasetKind, status.episode, status.revision]);

  useEffect(() => {
    Object.values(recordedVideoRefs.current).forEach((video) => {
      if (!video) return;
      if (status.state !== "replaying") {
        video.pause();
        return;
      }
      video.currentTime = 0;
      void video.play().catch(() => {
        // Browser autoplay policy can still block muted video in some environments; controls remain visible.
      });
    });
  }, [cameraMatches.map((camera) => camera.cameraKey).join("|"), status.state]);

  // Never a literal: the gateway builds the replay command from `robot.target_frame_name`, and the
  // two FR3 tool frames are 411 mm apart on the same URDF. A label that disagrees with the config
  // is worse than no label, because replaying a dataset from the other frame still "works" -- so a
  // gateway too old to report the frame says so rather than being guessed at.
  const targetFrame = status.targetFrameName || "an unreported tool frame";

  const startConfirmed = (overrideMujocoFailure: boolean) => {
    setOverridePromptOpen(false);
    setMonitorRequested(true);
    setCameraStatus({ available: null, running: true, error: "Connecting to RealSense…" });
    onStart(mode, robotIp.trim(), endEffectorMode, overrideMujocoFailure);
  };

  return (
    <section className="panel real-robot-panel">
      <div className="panel-heading">
        <h2>Real Robot Replay</h2>
        <span>FR3 · Pika gripper · recorded end-effector trajectory</span>
      </div>
      <div className="real-robot-layout">
        <div className="real-robot-settings">
          <div className="teleop-config-grid">
            <div><span>Robot</span><strong>Franka Research 3</strong></div>
            <div><span>End effector</span><strong>Pika gripper · {targetFrame}</strong></div>
          </div>
          <label className="real-robot-ip-field">
            <span>Robot IP</span>
            <input value={robotIp} onChange={(event) => setRobotIp(event.target.value)} placeholder="192.168.1.206" />
          </label>
          <button className="danger real-robot-run" disabled={disabled} onClick={() => setOverridePromptOpen(true)} type="button">
            {status.state === "replaying" ? "Real-robot replay running…" : "Run real-robot replay"}
          </button>
          <p className="panel-note">
            {status.datasetKind === "exported"
              ? "Real robot replay is disabled for exported datasets."
              : !validationDecisionAvailable
                ? "Run MuJoCo validation to completion for this dataset and episode first. A run " +
                  "that finishes and fails can still be overridden below, with its errors in front " +
                  "of you; a validation that never ran leaves nothing to judge, so there is no " +
                  "override for it."
                : !ipValid
                  ? "Enter a valid robot IPv4 address."
                  : validationFailedButReviewable
                    ? "MuJoCo failed. You may click Run and make the final Yes/No decision in the warning window."
                    : `The gateway preflights this FR3, moves ${targetFrame} to frame 0, then streams the trajectory.`}
          </p>
        </div>
        <div className="realsense-monitor replay-camera-compare">
          <div className="realsense-monitor-heading">
            <strong>Replay camera monitor</strong>
            <span>{status.state === "replaying" ? "live vs recorded" : "starts with real replay"}</span>
          </div>
          {cameraMatches.length ? (
            <div className="replay-camera-compare-grid">
              {cameraMatches.map((camera) => (
                <div className="replay-camera-pair" key={camera.cameraKey}>
                  <div className="replay-camera-pair-heading">
                    <strong>{camera.cameraKey}</strong>
                    <span>config {camera.configKey ?? shortCameraName(camera.cameraKey)} · S/N {camera.serial ?? "?"}</span>
                  </div>
                  <div className="replay-camera-pair-grid">
                    <div className="replay-camera-card">
                      <div className="replay-camera-card-heading">
                        <strong>Live RealSense</strong>
                        <span>{camera.running ? "live" : "waiting"}</span>
                      </div>
                      <div className="replay-camera-media">
                        {camera.available && camera.running ? (
                          <img src={api.realSenseSnapshotUrl(frameKey, camera.cameraKey)} alt={`Live RealSense ${camera.cameraKey} during real robot replay`} />
                        ) : (
                          <div className="realsense-monitor-empty">
                            {camera.error || "This matched RealSense connects automatically when real-robot replay starts."}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="replay-camera-card">
                      <div className="replay-camera-card-heading">
                        <strong>Recorded reference</strong>
                        <span>episode {status.episode}</span>
                      </div>
                      <div className="replay-camera-media">
                        <video
                          ref={(element) => {
                            recordedVideoRefs.current[camera.cameraKey] = element;
                          }}
                          src={recordedVideoUrl(camera.cameraKey)}
                          muted
                          playsInline
                          controls
                          preload="metadata"
                          aria-label={`Recorded ${shortCameraName(camera.cameraKey)} camera for episode ${status.episode}`}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="realsense-monitor-empty replay-camera-no-match">
              {cameraStatus?.error ||
                (status.datasetKind === "exported"
                  ? "Exported datasets do not expose source camera video here."
                  : "No connected RealSense camera matches this episode's recorded camera observations.")}
            </div>
          )}
          <p className="replay-camera-note">
            During real replay, every dataset camera observation that matches a configured RealSense key is shown as a live/recorded pair. Matching uses the dataset camera key and the configured RealSense serial, not camera discovery order.
          </p>
        </div>
      </div>
      <div className="real-replay-log-block">
        <div className="real-replay-log-heading">
          <strong>Real replay diagnostics</strong>
          <span>{status.state}</span>
        </div>
        <pre className="real-replay-log">
          {status.realReplayLog?.length
            ? status.realReplayLog.join("\n")
            : "Waiting for a replay request. Preflight, camera, initial-pose, and trajectory output will appear here."}
        </pre>
      </div>
      {overridePromptOpen ? (
        <div className="danger-modal-backdrop" role="presentation">
          <div aria-labelledby="mujoco-override-title" aria-modal="true" className="danger-modal" role="dialog">
            <h3 id="mujoco-override-title">
              {validationFailedButReviewable ? "MuJoCo validation failed" : "Confirm real-robot replay"}
            </h3>
            <p>
              Dataset <strong>{status.datasetRoot ?? status.dataset}</strong>, episode <strong>{status.episode}</strong>, target <strong>{targetFrame}</strong> on <strong>{robotIp.trim()}</strong>.
            </p>
            <p>{validationFailedButReviewable ? "The recorded trajectory did not meet the simulation limits:" : "MuJoCo passed within these limits:"}</p>
            <ul>
              <li>
                Position: <strong>{validation?.maxPositionErrorMm?.toFixed(2)} mm</strong> · limit {validation?.maxPositionThresholdMm.toFixed(2)} mm
              </li>
              <li>
                Rotation: <strong>{validation?.maxRotationErrorDeg?.toFixed(2)} deg</strong> · limit {validation?.maxRotationThresholdDeg.toFixed(2)} deg
              </li>
            </ul>
            <p>
              {validationFailedButReviewable
                ? "Proceeding may cause unexpected or unsafe robot motion. Do you want to run the real robot anyway?"
                : "A simulation pass does not guarantee safe hardware motion. Do you want to run the real robot?"}
            </p>
            <div className="danger-modal-actions">
              <button autoFocus onClick={() => setOverridePromptOpen(false)} type="button">No, cancel</button>
              <button className="danger" onClick={() => startConfirmed(validationFailedButReviewable)} type="button">Yes, proceed</button>
            </div>
          </div>
        </div>
      ) : null}
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
                  {/* A view replays its own derived action contract -- which is the point of
                      replaying one -- so name the contract rather than just the dataset. */}
                  {dataset.datasetKind === "training_view" ? (
                    <em>training view{dataset.actionContract ? ` · ${dataset.actionContract}` : ""}</em>
                  ) : null}
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
  onOpenProcessing,
  supportsTrajectoryGeneration = true
}: {
  status: ReplayStatus;
  processing: ProcessingItem | null;
  busy: boolean;
  onGenerate: () => void;
  onOpenProcessing: () => void;
  supportsTrajectoryGeneration?: boolean;
}) {
  const trajectoryReady =
    status.dataStatus === "loaded" &&
    (status.trajectoryKind === "pose" || status.trajectoryKind === "gripper_width") &&
    (!processing ||
      processing.status === "qc_pass" ||
      processing.status === "qc_warn" ||
      processing.status === "pose_ready");
  const qcLabel = !processing
    ? "—"
    : processing.status === "qc_pass"
      ? "Pass"
      : processing.status === "qc_warn"
        ? "Warnings"
        : processing.status === "qc_failed"
          ? "Fail"
          : "Pending";
  const validFrames = processing?.validFramesPct != null ? `${processing.validFramesPct}%` : "—";

  if (!trajectoryReady) {
    return (
      <section className="panel readiness-card readiness-missing">
        <div className="panel-heading">
          <h2>Trajectory: Missing</h2>
          <span>{supportsTrajectoryGeneration ? "EE trajectory not generated" : "recorded EE trajectory unavailable"}</span>
        </div>
        <p className="panel-note">
          {supportsTrajectoryGeneration
            ? "Generate EE trajectory before replaying this dataset. Replay is disabled until the trajectory is ready."
            : "Run QC on the recorded FR3 trajectory before replaying this dataset."}
        </p>
        <div className="control-row">
          {supportsTrajectoryGeneration ? (
            <button disabled={busy || !processing} onClick={onGenerate}>Generate EE Trajectory</button>
          ) : null}
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
            <span>
              Use for training
              <small>unchecked: Build View leaves this episode out; the recording is untouched</small>
            </span>
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
  onMujocoReplay: (mode: MujocoCubeMode) => void;
  onApproveMujoco: (mode: MujocoCubeMode) => void;
  onRealReplay: (mode: RealCubeMode, robotIp: string, endEffectorMode: RealEndEffectorMode, overrideMujocoFailure: boolean) => void;
  onAbort: () => void;
  onSelectDataset: (path: string) => void;
  onSelectEpisode: (episode: number) => void;
  onDeleteEpisode: (episode: number) => void;
  onGenerateForActive: () => void;
  onOpenProcessing: () => void;
  onSaveAnnotation: (annotation: EpisodeAnnotation) => void;
}) {
  const [mujocoMode, setMujocoMode] = useState<MujocoCubeMode>(snapshot.replay.mujocoCubeMode ?? "left");
  // The workstation replays the arm's own recorded EE stream; there are no AprilTag cubes to
  // pick between, and the gateway ignores the cube mode on that profile entirely.
  const workstationProfile = (snapshot.deployment?.profile ?? "thor") === "workstation";
  const cubeSelection = !workstationProfile;
  const activePath = snapshot.replay.datasetRoot ?? snapshot.replay.dataset;
  const matchingProcessing =
    snapshot.processing.find((item) => item.path === activePath) ?? null;
  return (
    <div className="page-stack">
      <PageHeader title="Episode Replay" subtitle="consume processed trajectories: timeline review, safety preflight, MuJoCo validation, and real-robot replay" />
      <ReplayReadinessCard
        status={snapshot.replay}
        processing={matchingProcessing}
        busy={busy}
        onGenerate={onGenerateForActive}
        onOpenProcessing={onOpenProcessing}
        supportsTrajectoryGeneration={!workstationProfile}
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
          mujocoMode={mujocoMode}
          cubeSelection={cubeSelection}
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
        cubeSelection={cubeSelection}
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
