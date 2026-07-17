import { useEffect, useRef, useState } from "react";
import { type GuiSnapshot } from "./api";
import { api } from "./apiClient";
import "./styles.css";
import type { CollectionTask, EpisodeAnnotation, ProcessingItem } from "./types";
import { StatusDot, Metric, processingStatusLabel, taskNeedsQcExportConfirmation } from "./shared/ui";
import { CalibrationPage } from "./calibration/CalibrationPage";
import { historyRepo, deviceBoxId } from "./calibration/adapters";
import { summarizeKinds } from "./calibration/status";
import type { CalibrationKind } from "./calibration/types";
import { LiveRecordPage } from "./pages/LiveRecordPage";
import { EpisodeReplayPage } from "./pages/EpisodeReplayPage";
import { DatasetProcessingPage } from "./pages/DatasetProcessingPage";
import { QcReportPage } from "./pages/QcReportPage";
import { DatasetExportPage } from "./pages/DatasetExportPage";
import { TaskLibraryPage } from "./pages/TaskLibraryPage";
import { DeviceManagerPage } from "./pages/DeviceManagerPage";
import { PlaceholderPage } from "./pages/PlaceholderPage";

export type PageId =
  | "live-record"
  | "dataset-processing"
  | "episode-replay"
  | "dataset-export"
  | "dashboard"
  | "qc-report"
  | "model-evaluation"
  | "device-manager"
  | "task-library"
  | "calibration"
  | "annotation-audit";

type PageMeta = { id: PageId; label: string; kind: "mvp" | "deferred" };

const mvpPages: PageMeta[] = [
  { id: "live-record", label: "Live Record", kind: "mvp" },
  { id: "dataset-processing", label: "Dataset Processing", kind: "mvp" },
  { id: "episode-replay", label: "Episode Replay", kind: "mvp" },
  { id: "dataset-export", label: "Dataset Export", kind: "mvp" },
  { id: "task-library", label: "Task Library", kind: "mvp" },
  { id: "calibration", label: "标定中心", kind: "mvp" },
  { id: "device-manager", label: "Device Manager", kind: "mvp" }
];

const deferredPages: PageMeta[] = [
  { id: "dashboard", label: "Dashboard", kind: "deferred" },
  { id: "qc-report", label: "QC Report", kind: "deferred" },
  { id: "model-evaluation", label: "Model Evaluation", kind: "deferred" },
  { id: "annotation-audit", label: "Annotation & Audit", kind: "deferred" }
];

const pages: PageMeta[] = [...mvpPages, ...deferredPages];

// Grouped sidebar layout (spec §2). Ids reference `pages` above so hash routing
// and validity are unchanged; only the presentation is grouped.
type NavGroup = { label: string; ids: PageId[] };
const navGroups: NavGroup[] = [
  { label: "Overview", ids: ["dashboard"] },
  { label: "采集", ids: ["live-record", "task-library"] },
  { label: "标定中心", ids: ["calibration"] },
  { label: "设备", ids: ["device-manager"] },
  { label: "数据", ids: ["dataset-processing", "episode-replay", "dataset-export"] }
];
const navPageLabels: Partial<Record<PageId, string>> = {
  dashboard: "Overview"
};
const deferredNavIds: PageId[] = ["qc-report", "model-evaluation", "annotation-audit"];

function pageFromHash(): PageId {
  const hash = window.location.hash.replace(/^#\/?/, "") as PageId;
  return pages.some((page) => page.id === hash) ? hash : "live-record";
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
        onDeleteEpisode={(episode) => run(() => api.deleteReplayEpisode(episode))}
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
        onExportTask={exportTaskWithQcGuard}
        onExportApprovedDataset={(path) => run(() => api.exportApprovedDataset(path))}
        onOpenProcessing={() => navigate("dataset-processing")}
        onOpenReplay={(path) => selectAndOpenReplay(path)}
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
    ) : activePage === "calibration" ? (
      <CalibrationPage snapshot={snapshot} api={api} />
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
        <SidebarNav activePage={activePage} onNavigate={navigate} snapshot={snapshot} />
        <section className="page-content">{pageNode}</section>
      </div>
    </main>
  );
}

// Small freshness badge shown on the 标定中心 nav entry: worst-case of the three
// calibration kinds, read from the shared local history repo.
function calibrationNavBadge(): { text: string; tone: "running" | "warning" | "error" } {
  const records = historyRepo.list();
  const latestByKind: Partial<Record<CalibrationKind, number | null>> = {
    force_origin: records.find((r) => r.kind === "force_origin")?.timestamp ?? null,
    force_dynamic: records.find((r) => r.kind === "force_dynamic")?.timestamp ?? null,
    touch: records.find((r) => r.kind === "touch")?.timestamp ?? null
  };
  const summary = summarizeKinds(latestByKind);
  const needs = summary.overdue + summary.unknown;
  if (needs > 0) return { text: `${needs} overdue`, tone: "error" };
  if (summary.dueSoon > 0) return { text: `${summary.dueSoon} due soon`, tone: "warning" };
  return { text: "Ready", tone: "running" };
}

function SidebarNav({
  activePage,
  onNavigate,
  snapshot
}: {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
  snapshot: GuiSnapshot;
}) {
  const [showDeferred, setShowDeferred] = useState(false);
  const labelFor = (id: PageId) => navPageLabels[id] ?? pages.find((p) => p.id === id)?.label ?? id;
  const caliBadge = calibrationNavBadge();

  const onlineCount = snapshot.devices.filter((d) => d.state === "running").length;
  const totalCount = snapshot.devices.length;
  const boxDevices = snapshot.devices.filter((d) => d.kind === "box_collection");
  const boxIds = [...new Set(boxDevices.map((d) => deviceBoxId(d)))];
  const boxIdLabel = boxIds.length ? boxIds.map((b) => b || "默认").join(", ") : "—";

  return (
    <nav className="sidebar" aria-label="Robot data factory pages">
      {navGroups.map((group) => (
        <div className="nav-group" key={group.label}>
          <small className="nav-group-label">{group.label}</small>
          {group.ids.map((id) => (
            <button
              className={id === activePage ? "nav-item active" : "nav-item"}
              key={id}
              onClick={() => onNavigate(id)}
            >
              <span>{labelFor(id)}</span>
              {id === "calibration" && (
                <small className={`nav-badge nav-badge-${caliBadge.tone}`}>
                  <span className={`status-dot status-${caliBadge.tone}`} />
                  {caliBadge.text}
                </small>
              )}
            </button>
          ))}
        </div>
      ))}
      <div className="nav-group">
        <button className="nav-toggle" onClick={() => setShowDeferred((value) => !value)}>
          <span>Deferred</span>
          <small>{showDeferred ? "hide" : `${deferredNavIds.length} hidden`}</small>
        </button>
        {showDeferred
          ? deferredNavIds.map((id) => (
              <button
                className={id === activePage ? "nav-item active deferred" : "nav-item deferred"}
                key={id}
                onClick={() => onNavigate(id)}
              >
                <span>{labelFor(id)}</span>
                <small>deferred</small>
              </button>
            ))
          : null}
      </div>

      {/* Global device status footer (spec §2). Environment temp/humidity has no
          backend feed yet, so it renders as "--" rather than a fake value. */}
      <div className="nav-footer">
        <div className="nav-footer-row">
          <span className={`status-dot status-${onlineCount === totalCount && totalCount > 0 ? "running" : "warning"}`} />
          <span>{onlineCount} / {totalCount} 设备在线</span>
        </div>
        <div className="nav-footer-row">
          <span className={`status-dot status-${caliBadge.tone}`} />
          <span>标定：{caliBadge.text}</span>
        </div>
        <div className="nav-footer-row nav-footer-muted">环境：-- ℃ / -- %RH</div>
        <div className="nav-footer-row nav-footer-muted">BOX ID：{boxIdLabel}</div>
      </div>
    </nav>
  );
}

export default App;
