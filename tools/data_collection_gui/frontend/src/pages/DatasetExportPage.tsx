import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GuiSnapshot } from "../api";
import type { BoxPreviewPayload, BoxCaliLog, BoxCaliLogLine, CollectionTask, ConfigSummary, DeviceStatus, EpisodeAnnotation, EventLogItem, ProcessingItem, ProcessingStatus, RecordedDataset, RecordingStatus, ReplayStatus, SubtaskSegment, TaskStatus, DatasetExportStatus, AnnotationOutcome, AnnotationQuality, ReviewStatus } from "../types";
import { StatusDot, Metric, PageHeader, stateLabel, QualityOverview, processingStatusLabel, datasetNamePrefixes, taskDatasetBaseName, processingItemsForTask, taskNeedsQcExportConfirmation, taskStatusLabel, taskStatusDot } from "../shared/ui";

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
  onExportApprovedDataset: (path: string) => void;
  onOpenProcessing: () => void;
  onOpenReplay: (path: string) => void;
}) {
  const exportStatus = snapshot.datasetExport;
  const eligible = snapshot.processing.filter((item) => item.status === "qc_pass");
  const hasEligible = eligible.length > 0;
  const exportableTasks = (snapshot.tasks ?? []).filter((t) => t.datasetRepoId);
  const exporting = exportStatus.state === "exporting";
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
                    <button
                      disabled={busy || exporting}
                      onClick={() => onExportApprovedDataset(item.path)}
                    >
                      {exporting && exportStatus.datasetRoot === item.path ? "Exporting…" : "Export v3"}
                    </button>
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

