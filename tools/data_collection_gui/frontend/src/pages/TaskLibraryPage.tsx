import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GuiSnapshot } from "../api";
import type { BoxPreviewPayload, BoxCaliLog, BoxCaliLogLine, CollectionTask, ConfigSummary, DeviceStatus, EpisodeAnnotation, EventLogItem, ProcessingItem, ProcessingStatus, RecordedDataset, RecordingStatus, ReplayStatus, SubtaskSegment, TaskStatus, DatasetExportStatus, AnnotationOutcome, AnnotationQuality, ReviewStatus } from "../types";
import { StatusDot, Metric, PageHeader, stateLabel, QualityOverview, processingStatusLabel, datasetNamePrefixes, taskDatasetBaseName, processingItemsForTask, taskNeedsQcExportConfirmation, taskStatusLabel, taskStatusDot } from "../shared/ui";
import type { PageId } from "../App";

export function TaskLibraryPage({
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

