import { useState } from "react";
import type { CameraCropSpecs, GuiSnapshot } from "../api";
import type { RecordedDataset } from "../types";
import {
  StatusDot,
  Metric,
  PageHeader,
  processingStatusLabel,
  qcWarnings,
  stateLabel,
  taskStatusDot
} from "../shared/ui";

type ActionMode = "absolute_ee" | "delta_ee_from_prev_cmd" | "delta_ee_from_current";

// Rates a training view may be built at. Mirrors TRAINING_VIEW_FPS_CHOICES in gateway.py.
// 0 means "keep whatever the recording is", which the exporter only allows when every source
// in one build already agrees.
const VIEW_FPS_CHOICES = [30, 15, 60, 0] as const;
const DEFAULT_VIEW_FPS = 30;

/** Why this recording cannot be built at `viewFps`, or "" when it can. */
function viewFpsProblem(sourceFps: number | undefined, viewFps: number): string {
  if (!sourceFps || viewFps === 0) return "";
  if (sourceFps < viewFps) {
    return `recorded at ${sourceFps} fps — a ${viewFps} fps view would have to invent frames`;
  }
  if (sourceFps % viewFps !== 0) {
    // Only integer decimation exists. Nearest-frame resampling would make the gap between
    // kept frames alternate between 1 and 2 source frames, and the action is a per-frame
    // delta, so that swing lands directly in the values the policy learns.
    const divisors = [1, 2, 3, 4].filter((n) => sourceFps % n === 0).map((n) => sourceFps / n);
    return `${sourceFps} fps is not an integer multiple of ${viewFps} — pick ${divisors.join(", ")}`;
  }
  return "";
}

const actionModeCopy: Record<ActionMode, { label: string; blurb: string }> = {
  absolute_ee: {
    label: "Absolute EE",
    blurb:
      "action = absolute target pose (quaternion rotation). Rate-independent; the contract the recorder stores natively."
  },
  delta_ee_from_prev_cmd: {
    label: "Delta EE — vs previous command",
    blurb:
      "action = per-frame increment against the pose commanded on the previous frame (rotvec rotation). Arm tracking lag stays out of the action; a held frame is an exact zero."
  },
  delta_ee_from_current: {
    label: "Delta EE — vs measured pose",
    blurb:
      "action = per-frame increment against the measured pose. Purely reactive at deployment, but it bakes this rig's tracking lag into every action."
  }
};

function contractLabel(contract: string): string {
  return actionModeCopy[contract as ActionMode]?.label ?? contract;
}

type CameraCropInput = { x: string; y: string; w: string; h: string };

function evenFloor(value: number): number {
  return Math.max(0, Math.floor(value / 2) * 2);
}

function fullFrameCropInput(width: number, height: number): CameraCropInput {
  return { x: "0", y: "0", w: String(evenFloor(width)), h: String(evenFloor(height)) };
}

function recommendedCropInput(key: string, width: number, height: number): CameraCropInput {
  if (!key.endsWith(".side")) {
    return fullFrameCropInput(width, height);
  }
  if (width >= 640 && height >= 480) {
    return {
      x: "224",
      y: "0",
      w: String(Math.min(416, evenFloor(width - 224))),
      h: String(Math.min(346, evenFloor(height)))
    };
  }
  const x = evenFloor(width * 0.35);
  return { x: String(x), y: "0", w: String(evenFloor(width - x)), h: String(evenFloor(height * 0.72)) };
}

function cropInputForFeature(dataset: RecordedDataset, inputs: Record<string, CameraCropInput>, key: string): CameraCropInput {
  const feature = (dataset.cameraFeatures ?? []).find((candidate) => candidate.key === key);
  return inputs[key] ?? fullFrameCropInput(feature?.width ?? 0, feature?.height ?? 0);
}

function cropSpecsForDataset(
  dataset: RecordedDataset,
  enabled: boolean,
  inputs: Record<string, CameraCropInput>
): { crops?: CameraCropSpecs; error?: string; label?: string } {
  if (!enabled) return {};
  const features = (dataset.cameraFeatures ?? []).filter((feature) => feature.width > 0 && feature.height > 0);
  if (features.length === 0) return { error: "Camera crop needs dataset camera metadata" };
  const crops: CameraCropSpecs = {};
  for (const feature of features) {
    const input = cropInputForFeature(dataset, inputs, feature.key);
    const x = Number(input.x);
    const y = Number(input.y);
    const w = Number(input.w);
    const h = Number(input.h);
    if (![x, y, w, h].every(Number.isInteger)) {
      return { error: `${feature.key} crop must be integer pixels` };
    }
    if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > feature.width || y + h > feature.height) {
      return { error: `${feature.key} crop is outside ${feature.width}x${feature.height}` };
    }
    if ([x, y, w, h].some((value) => value % 2 !== 0)) {
      return { error: `${feature.key} crop must use even x/y/w/h for H.264` };
    }
    if (x !== 0 || y !== 0 || w !== evenFloor(feature.width) || h !== evenFloor(feature.height)) {
      crops[feature.key] = [x, y, w, h];
    }
  }
  const count = Object.keys(crops).length;
  return count > 0 ? { crops, label: `${count} camera crop${count === 1 ? "" : "s"}` } : { label: "full frame" };
}

/** Workstation counterpart of Dataset Export: build the policy-ready view of a v3 recording. */
function TrainingViewPage({
  snapshot,
  busy,
  onBuildView,
  onOpenProcessing,
  onOpenReplay
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onBuildView: (path: string, actionMode?: string, cameraCrops?: CameraCropSpecs, viewFps?: number) => void;
  onOpenProcessing: () => void;
  onOpenReplay: (path: string) => void;
}) {
  const exportStatus = snapshot.datasetExport;
  const building = exportStatus.state === "exporting";
  const [actionMode, setActionMode] = useState<ActionMode>("delta_ee_from_prev_cmd");
  const [viewFps, setViewFps] = useState<number>(DEFAULT_VIEW_FPS);
  const [cropEnabled, setCropEnabled] = useState(false);
  const [cameraCropInputs, setCameraCropInputs] = useState<Record<string, CameraCropInput>>({});
  const allDatasets = snapshot.recordedDatasets ?? [];
  // Views are replay candidates, so they arrive in the same list as the recordings. They belong
  // under the recording they were derived from, not next to it as another build source.
  const datasets = allDatasets.filter((dataset) => dataset.datasetKind !== "training_view");
  const cropCameraFeatures = Array.from(
    new Map(
      datasets
        .flatMap((dataset) => dataset.cameraFeatures ?? [])
        .filter((feature) => feature.width > 0 && feature.height > 0)
        .map((feature) => [feature.key, feature] as const)
    ).values()
  );
  const setCameraCropValue = (key: string, field: keyof CameraCropInput, value: string) => {
    setCameraCropInputs((current) => {
      const feature = cropCameraFeatures.find((candidate) => candidate.key === key);
      const fallback = fullFrameCropInput(feature?.width ?? 0, feature?.height ?? 0);
      return { ...current, [key]: { ...(current[key] ?? fallback), [field]: value } };
    });
  };
  const useRecommendedCameraCrop = () => {
    setCropEnabled(true);
    setCameraCropInputs(
      Object.fromEntries(
        cropCameraFeatures.map((feature) => [
          feature.key,
          recommendedCropInput(feature.key, feature.width, feature.height)
        ])
      )
    );
  };
  const resetCameraCrops = () => {
    setCameraCropInputs(
      Object.fromEntries(
        cropCameraFeatures.map((feature) => [feature.key, fullFrameCropInput(feature.width, feature.height)])
      )
    );
  };
  // The gateway refuses to build a view of a dataset that has not passed QC, so the row has to
  // say where a dataset stands before the button is pressed. Shown rather than filtered: on this
  // profile every recording is a build candidate, and hiding the ones that need QC is what made
  // the Thor page look like it had silently lost datasets.
  const qcItemFor = (dataset: RecordedDataset) =>
    snapshot.processing.find((item) => item.path === dataset.path);
  const viewsFor = (dataset: RecordedDataset) =>
    allDatasets.filter(
      (candidate) =>
        candidate.datasetKind === "training_view" &&
        (candidate.viewOf
          ? candidate.viewOf === dataset.path
          : candidate.viewOfName === dataset.name || candidate.name.startsWith(`${dataset.name}__`))
    );

  return (
    <div className="page-stack">
      <PageHeader
        title="Training View"
        subtitle="re-express a recorded v3 dataset in the action contract the policy will be trained on"
      />
      <section className="panel">
        <div className="panel-heading">
          <h2>Action Contract</h2>
          <span>{actionModeCopy[actionMode].label}</span>
        </div>
        <p className="panel-note">
          Recording always stores absolute EE. The delta contracts are derived here by differencing
          consecutive dataset frames — a delta computed during capture would span one control tick
          (200 Hz) instead of one frame (30 Hz) and drive the arm ~6.7&times; too slow. Videos are
          symlinked unless crop is enabled, so a full-frame view costs almost no disk.
        </p>
        <div className="mujoco-mode-picker" role="group" aria-label="Action contract">
          {(Object.keys(actionModeCopy) as ActionMode[]).map((mode) => (
            <button
              key={mode}
              className={actionMode === mode ? "active" : ""}
              disabled={busy || building}
              onClick={() => setActionMode(mode)}
              type="button"
            >
              {actionModeCopy[mode].label}
            </button>
          ))}
        </div>
        <p className="panel-note">{actionModeCopy[actionMode].blurb}</p>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>Frame Rate</h2>
          <span>{viewFps === 0 ? "keep source rate" : `${viewFps} fps`}</span>
        </div>
        <p className="panel-note">
          The action is a <em>per-frame</em> delta, so the same real motion is twice as large per
          frame at 30 fps as at 60. Views built at different rates therefore cannot be merged —
          the difference lands in the action values themselves and nothing downstream can see it.
          Building every view at one rate is what lets a 60 fps session join the 30 fps baseline.
          Frames are dropped, never interpolated, and the videos are not re-encoded.
        </p>
        <div className="mujoco-mode-picker" role="group" aria-label="Training view frame rate">
          {VIEW_FPS_CHOICES.map((choice) => (
            <button
              key={choice}
              className={viewFps === choice ? "active" : ""}
              disabled={busy || building}
              onClick={() => setViewFps(choice)}
              type="button"
            >
              {choice === 0 ? "Source rate" : `${choice} fps`}
            </button>
          ))}
        </div>
      </section>

      <section className="panel camera-crop-panel">
        <div className="panel-heading">
          <h2>Camera Crop</h2>
          <span>{cropEnabled ? "enabled" : "full frame"}</span>
        </div>
        <p className="panel-note">
          Crop is applied only to the generated training view. The raw recording stays unchanged.
        </p>
        <div className="control-row">
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={cropEnabled}
              disabled={busy || building || cropCameraFeatures.length === 0}
              onChange={(event) => setCropEnabled(event.target.checked)}
            />
            <span>Use crop for training view</span>
          </label>
          <button disabled={busy || building || cropCameraFeatures.length === 0} onClick={useRecommendedCameraCrop}>
            Use Side ROI
          </button>
          <button disabled={busy || building || cropCameraFeatures.length === 0} onClick={resetCameraCrops}>
            Full Frame
          </button>
        </div>
        {cropEnabled && cropCameraFeatures.length > 0 ? (
          <div className="camera-crop-grid">
            {cropCameraFeatures.map((feature) => {
              const input = cameraCropInputs[feature.key] ?? fullFrameCropInput(feature.width, feature.height);
              return (
                <div className="camera-crop-row" key={feature.key}>
                  <div>
                    <strong>{feature.key}</strong>
                    <small>{feature.width}x{feature.height}</small>
                  </div>
                  {(["x", "y", "w", "h"] as const).map((field) => (
                    <label key={field}>
                      <span>{field}</span>
                      <input
                        type="number"
                        min={field === "w" || field === "h" ? 2 : 0}
                        step={2}
                        value={input[field]}
                        disabled={busy || building}
                        onChange={(event) => setCameraCropValue(feature.key, field, event.target.value)}
                      />
                    </label>
                  ))}
                </div>
              );
            })}
          </div>
        ) : null}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>Recorded Datasets</h2>
          <span>{datasets.length} available</span>
        </div>
        {datasets.length === 0 ? (
          <div className="empty-dataset-list">No recorded datasets yet. Record an episode first.</div>
        ) : (
          <div className="processing-list">
            {datasets.map((dataset) => {
              const views = viewsFor(dataset);
              const buildingThis = building && exportStatus.datasetRoot === dataset.path;
              const excluded = dataset.excludedEpisodes ?? [];
              const kept = Math.max(0, dataset.totalEpisodes - excluded.length);
              const qc = qcItemFor(dataset);
              const qcStatus = qc?.status;
              const warned = qcStatus === "qc_warn";
              const qcReady = qcStatus === "qc_pass" || warned;
              const warnings = qc ? qcWarnings(qc) : [];
              const cropResult = cropSpecsForDataset(dataset, cropEnabled, cameraCropInputs);
              const fpsProblem = viewFpsProblem(dataset.fps, viewFps);
              const blockedReason = !qcReady
                ? qcStatus === "qc_failed"
                  ? "QC failed — fix or re-record before training on this"
                  : "Run QC in Dataset Processing before building a view"
                : kept === 0
                  ? "Every episode is marked not for training"
                  : (cropResult.error ?? fpsProblem) || undefined;
              return (
                <div className="processing-row" key={dataset.path}>
                  <div className="processing-row-main static">
                    <div>
                      <div className="row-title">
                        <StatusDot
                          state={
                            qcStatus === "qc_pass"
                              ? "running"
                              : warned
                                ? "warning"
                                : qcStatus === "qc_failed"
                                  ? "error"
                                  : "idle"
                          }
                        />
                        <strong>{dataset.name}</strong>
                        <em>{qcStatus ? processingStatusLabel[qcStatus] : "QC not run"}</em>
                      </div>
                      <p>
                        {dataset.totalEpisodes} episode(s) · {dataset.totalFrames} frames
                        {dataset.fps ? ` · ${dataset.fps} fps` : ""} · {dataset.path}
                      </p>
                      {qcReady && fpsProblem ? (
                        <p className="panel-note">Cannot build at {viewFps} fps: {fpsProblem}.</p>
                      ) : null}
                      {qcReady && !fpsProblem && dataset.fps && viewFps > 0 && dataset.fps !== viewFps ? (
                        <p className="panel-note">
                          Keeping 1 frame of {dataset.fps / viewFps} — the view will hold about{" "}
                          {Math.ceil(dataset.totalFrames / (dataset.fps / viewFps)).toLocaleString()} frames
                          at {viewFps} fps.
                        </p>
                      ) : null}
                      {qcReady ? null : (
                        // The gate is stated on the row rather than left to the error the build
                        // would have returned: QC is a page away, and "why is this disabled" has
                        // to be answerable without pressing the button first.
                        <p className="panel-note">
                          {blockedReason}. {qc?.qcSummary ?? "QC has not run on this recording."}
                        </p>
                      )}
                      {warned ? (
                        <p className="panel-note">
                          {warnings.length ? warnings.join(" · ") : qc?.message} — building asks for
                          confirmation first.
                        </p>
                      ) : null}
                      {cropEnabled && cropResult.label ? (
                        <p className="panel-note">Training view video input: {cropResult.label}.</p>
                      ) : null}
                      {excluded.length > 0 ? (
                        // Shown before the build, not after: this is the operator's own review
                        // deciding what reaches training, and it changes what the button does.
                        <p className="panel-note">
                          {kept} of {dataset.totalEpisodes} will be built — episode
                          {excluded.length > 1 ? "s" : ""} {excluded.join(", ")} marked not for
                          training in Episode Replay.
                        </p>
                      ) : null}
                    </div>
                    <div className="processing-stats">
                      <button
                        disabled={
                          busy ||
                          building ||
                          !qcReady ||
                          kept === 0 ||
                          Boolean(cropResult.error) ||
                          Boolean(fpsProblem)
                        }
                        title={blockedReason}
                        onClick={() => onBuildView(dataset.path, actionMode, cropResult.crops, viewFps)}
                      >
                        {buildingThis ? "Building…" : "Build View"}
                      </button>
                      {qcReady ? null : (
                        // Navigates; it does not run QC. Naming it for what it does keeps the
                        // page honest about where the gate is cleared.
                        <button disabled={busy} onClick={onOpenProcessing}>
                          Open Processing
                        </button>
                      )}
                    </div>
                  </div>
                  {views.length > 0 || buildingThis ? (
                    <div className="view-list">
                      {views.map((view) => (
                        <div className="view-row" key={view.path}>
                          <div>
                            <strong>{contractLabel(view.actionContract ?? "")}</strong>
                            <p>
                              {view.totalEpisodes} episode(s) · {view.totalFrames} frames · built {view.updatedAt}
                            </p>
                            <p className="view-path">{view.path}</p>
                          </div>
                          <button disabled={busy || building} onClick={() => onOpenReplay(view.path)}>
                            Open in Replay
                          </button>
                        </div>
                      ))}
                      {buildingThis ? (
                        <div className="view-row pending">
                          <div>
                            <strong>{contractLabel(exportStatus.target)}</strong>
                            <p>{exportStatus.message || "Building…"}</p>
                          </div>
                          <StatusDot state="running" />
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              );
            })}
          </div>
        )}
      </section>

      {exportStatus.state !== "idle" && (
        <section className="panel">
          <div className="panel-heading">
            <h2>Build Status</h2>
            <span className="state-pill">
              <StatusDot state={exportStatus.state} />
              {stateLabel(exportStatus.state)}
            </span>
          </div>
          <div className="summary-grid">
            <Metric label="Source" value={exportStatus.datasetRoot || "—"} />
            <Metric label="Contract" value={contractLabel(exportStatus.target)} />
            <Metric label="View" value={exportStatus.outputPath || "—"} />
            <Metric label="Episodes" value={exportStatus.selectedEpisodes} />
            <Metric label="Frames" value={exportStatus.totalFrames} />
            <Metric label="Latest log" value={exportStatus.message || "—"} />
          </div>
        </section>
      )}
    </div>
  );
}

export function DatasetExportPage({
  snapshot,
  busy,
  onExportTask,
  onExportApprovedDataset,
  onOpenProcessing,
  onOpenReplay
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onExportTask: (id: string) => void;
  onExportApprovedDataset: (
    path: string,
    actionMode?: string,
    cameraCrops?: CameraCropSpecs,
    viewFps?: number
  ) => void;
  onOpenProcessing: () => void;
  onOpenReplay: (path: string) => void;
}) {
  const exportStatus = snapshot.datasetExport;
  // A warned dataset is exportable, with the warnings acknowledged. Leaving it out of this list
  // is what made a single warn look like "QC pending" and silently withdraw the dataset.
  const eligible = snapshot.processing.filter(
    (item) => item.status === "qc_pass" || item.status === "qc_warn"
  );
  const hasEligible = eligible.length > 0;
  const exportableTasks = (snapshot.tasks ?? []).filter((t) => t.datasetRepoId);
  const exporting = exportStatus.state === "exporting";

  // The FR3 workstation recorder already writes LeRobot v3, so there is no raw->v3 export to
  // run here (that is the Thor GMSL2 path). What it needs instead is a training view: the same
  // episodes with the action column in whichever contract the policy will be trained on.
  if (snapshot.deployment?.profile === "workstation") {
    return (
      <TrainingViewPage
        snapshot={snapshot}
        busy={busy}
        onBuildView={onExportApprovedDataset}
        onOpenProcessing={onOpenProcessing}
        onOpenReplay={onOpenReplay}
      />
    );
  }

  return (
    <div className="page-stack">
      <PageHeader title="Dataset Export" subtitle="consolidate task sessions or QC-approved datasets into LeRobot v3" />
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
        {exportStatus.outputPath ? (
          <div className="control-row">
            <button disabled={busy || exporting} onClick={() => onOpenReplay(exportStatus.outputPath)}>Open Replay</button>
          </div>
        ) : null}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>Approved Datasets</h2>
          <span>{eligible.length} ready</span>
        </div>
        {hasEligible ? (
          <div className="processing-list">
            {eligible.map((item) => {
              const warnings = qcWarnings(item);
              const warned = item.status === "qc_warn";
              return (
                <div className="processing-row" key={item.path}>
                  <div className="processing-row-main static">
                    <div>
                      <div className="row-title">
                        <StatusDot state={warned ? "warning" : "running"} />
                        <strong>{item.name}</strong>
                        {item.trajectoryVersion ? <em>{item.trajectoryVersion}</em> : null}
                      </div>
                      <p>{item.qcSummary}</p>
                      {warned ? (
                        <p className="panel-note">
                          {warnings.length ? warnings.join(" · ") : item.message} — exporting asks for
                          confirmation first.
                        </p>
                      ) : null}
                    </div>
                    <div className="processing-stats">
                      <span>{item.totalEpisodes} ep · {item.totalFrames} fr</span>
                      <small>{item.updatedAt}</small>
                      <button
                        disabled={busy || exporting}
                        onClick={() => onExportApprovedDataset(item.path)}
                      >
                        {exporting && exportStatus.datasetRoot === item.path ? "Exporting…" : "Export v3"}
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
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

