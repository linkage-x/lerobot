import { useState } from "react";
import type { GuiSnapshot } from "../api";
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
  onBuildView: (path: string, actionMode?: string) => void;
  onOpenProcessing: () => void;
  onOpenReplay: (path: string) => void;
}) {
  const exportStatus = snapshot.datasetExport;
  const building = exportStatus.state === "exporting";
  const [actionMode, setActionMode] = useState<ActionMode>("delta_ee_from_prev_cmd");
  const allDatasets = snapshot.recordedDatasets ?? [];
  // Views are replay candidates, so they arrive in the same list as the recordings. They belong
  // under the recording they were derived from, not next to it as another build source.
  const datasets = allDatasets.filter((dataset) => dataset.datasetKind !== "training_view");
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
          symlinked, so a view costs almost no disk.
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
              const blockedReason = !qcReady
                ? qcStatus === "qc_failed"
                  ? "QC failed — fix or re-record before training on this"
                  : "Run QC in Dataset Processing before building a view"
                : kept === 0
                  ? "Every episode is marked not for training"
                  : undefined;
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
                        {dataset.totalEpisodes} episode(s) · {dataset.totalFrames} frames · {dataset.path}
                      </p>
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
                        disabled={busy || building || !qcReady || kept === 0}
                        title={blockedReason}
                        onClick={() => onBuildView(dataset.path, actionMode)}
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
  onExportApprovedDataset: (path: string, actionMode?: string) => void;
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

