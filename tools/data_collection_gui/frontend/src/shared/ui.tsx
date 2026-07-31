// Shared UI primitives + cross-page helpers, extracted from App.tsx so the
// page modules can share them without a circular dependency on App.
import type { GuiSnapshot } from "../api";
import type { CollectionTask, ProcessingItem, ProcessingStatus, TaskStatus } from "../types";

export const taskStatusLabel: Record<TaskStatus, string> = {
  pending: "Pending",
  in_progress: "In Progress",
  completed: "Completed",
  paused: "Paused"
};

export const taskStatusDot: Record<TaskStatus, string> = {
  pending: "idle",
  in_progress: "running",
  completed: "completed",
  paused: "warning"
};

export function mujocoValidationMatchesSelection(
  validationCubeMode: string | undefined,
  selectedCubeMode: string,
  cubeSelection: boolean
): boolean {
  // A rig without AprilTag cubes has no selection to disagree with: the gateway ignores the cube
  // mode on that profile, so letting a mismatch through here would withdraw a validation the run
  // actually produced -- Real Robot locks and a passed result reads as "recommended", because of
  // a picker that changed nothing about what ran.
  if (!cubeSelection) return true;
  return (validationCubeMode ?? "left") === selectedCubeMode;
}

export function stateLabel(state: string) {
  return state.replace("_", " ");
}

export function datasetNamePrefixes(name: string): Set<string> {
  const prefixes = new Set([name]);
  const match = name.match(/^(?<base>.+)_\d{8}_\d{6}(?:_\d{2})?$/);
  if (match?.groups?.base) {
    prefixes.add(match.groups.base);
  }
  return prefixes;
}

export function taskDatasetBaseName(task: CollectionTask): string {
  return task.datasetRepoId.split("/").pop()?.trim() ?? "";
}

export function processingItemsForTask(task: CollectionTask, processing: ProcessingItem[]): ProcessingItem[] {
  const baseName = taskDatasetBaseName(task);
  if (!baseName) {
    return [];
  }
  return processing.filter((item) => datasetNamePrefixes(item.name).has(baseName));
}

export function taskNeedsQcExportConfirmation(task: CollectionTask, processing: ProcessingItem[]) {
  const taskItems = processingItemsForTask(task, processing);
  const notQcPassed = taskItems.filter((item) => item.status !== "qc_pass");
  const processingEpisodes = taskItems.reduce((total, item) => total + item.totalEpisodes, 0);
  const missingEpisodeCount = Math.max(task.completedEpisodes - processingEpisodes, 0);
  return { taskItems, notQcPassed, missingEpisodeCount };
}

export function StatusDot({ state }: { state: string }) {
  return <span className={`status-dot status-${state}`} aria-hidden="true" />;
}

export function Metric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}


export function PageHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="page-header">
      <div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
      </div>
    </div>
  );
}

export function QualityOverview({ snapshot }: { snapshot: GuiSnapshot }) {
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


export const processingStatusLabel: Record<ProcessingStatus, string> = {
  pose_missing: "Pose missing",
  queued: "Queued",
  running: "Running",
  pose_ready: "Pose ready",
  qc_pass: "QC pass",
  qc_failed: "QC failed",
  error: "Error"
};

