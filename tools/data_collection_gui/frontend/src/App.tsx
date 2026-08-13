import { useEffect, useRef, useState } from "react";
import { type GuiSnapshot } from "./api";
import { api } from "./apiClient";
import "./styles.css";
import type { CollectionTask, EpisodeAnnotation, ProcessingItem } from "./types";
import { StatusDot, Metric, stateLabel, processingStatusLabel, qcWarnings, taskNeedsQcExportConfirmation } from "./shared/ui";
import { CalibrationPage } from "./calibration/CalibrationPage";
import { DashboardPage } from "./pages/DashboardPage";
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
import { TeleoperationPage } from "./pages/TeleoperationPage";
import { PlaceholderPage } from "./pages/PlaceholderPage";

export type PageId =
  | "live-record"
  | "teleoperation"
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
  { id: "dashboard", label: "Dashboard", kind: "mvp" },
  { id: "live-record", label: "Live Record", kind: "mvp" },
  { id: "teleoperation", label: "Teleoperation", kind: "mvp" },
  { id: "dataset-processing", label: "Dataset Processing", kind: "mvp" },
  { id: "episode-replay", label: "Episode Replay", kind: "mvp" },
  { id: "dataset-export", label: "Dataset Export", kind: "mvp" },
  { id: "task-library", label: "Task Library", kind: "mvp" },
  { id: "calibration", label: "Calibration", kind: "mvp" },
  { id: "device-manager", label: "Device Manager", kind: "mvp" }
];

const deferredPages: PageMeta[] = [
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
  { label: "Capture", ids: ["live-record", "teleoperation", "task-library"] },
  { label: "Calibration", ids: ["calibration"] },
  { label: "Devices", ids: ["device-manager"] },
  { label: "Data", ids: ["dataset-processing", "episode-replay", "dataset-export"] }
];

const workstationPageIds = new Set<PageId>([
  "teleoperation",
  "live-record",
  // The task overlay is rig-independent: the gateway patches dataset.repo_id/root/single_task
  // onto whichever config the profile records with, so an FR3 session lands in the task's
  // dataset exactly like a Thor one. Without this page a workstation operator can only name a
  // task by editing fr3_record_config.yaml and restarting the gateway.
  "task-library",
  "device-manager",
  "dataset-processing",
  "episode-replay",
  "dataset-export"
]);

function pageAllowedForProfile(page: PageId, profile: "thor" | "workstation"): boolean {
  return profile === "workstation" ? workstationPageIds.has(page) : page !== "teleoperation";
}
const navPageLabels: Partial<Record<PageId, string>> = {
  dashboard: "Overview"
};
const deferredNavIds: PageId[] = ["qc-report", "model-evaluation", "annotation-audit"];

function pageFromHash(): PageId {
  const hash = window.location.hash.replace(/^#\/?/, "") as PageId;
  return pages.some((page) => page.id === hash) ? hash : "dashboard";
}


function App() {
  const [snapshot, setSnapshot] = useState<GuiSnapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const [activePage, setActivePage] = useState<PageId>(() => pageFromHash());
  // Kept outside the snapshot on purpose: the snapshot is replaced wholesale by the poll below.
  const [commandError, setCommandError] = useState<string | null>(null);
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
    if (!snapshot) return;
    const profile = snapshot.deployment?.profile ?? "thor";
    if (!pageAllowedForProfile(activePage, profile)) {
      const fallback = (snapshot.deployment?.defaultRoute as PageId) ?? "live-record";
      window.location.hash = fallback;
      setActivePage(fallback);
    }
  }, [activePage, snapshot?.deployment?.profile, snapshot?.deployment?.defaultRoute]);

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

  /** Run a gateway command; false means the gateway rejected it and `commandError` now says why. */
  async function run(command: () => Promise<GuiSnapshot>): Promise<boolean> {
    setBusy(true);
    api.consumeCommandFailure();
    try {
      setSnapshot(await command());
      const failure = api.consumeCommandFailure();
      setCommandError(failure ? `${failure.command}: ${failure.message}` : null);
      return failure === null;
    } catch (error) {
      setCommandError(error instanceof Error ? error.message : String(error));
      return false;
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
  const deployment = snapshot.deployment ?? { profile: "thor", label: "Thor Acquisition", defaultRoute: "live-record", capabilities: [] };
  const workstationProfile = deployment.profile === "workstation";

  // The recorder session is a global prerequisite (it streams the box/camera
  // live data Calibration and Device Manager also consume), so Connect/Disconnect
  // lives in the topbar next to the Recorder status -- not only in Live Record.
  const recorderConnected = ["connecting", "armed", "recording", "review", "saving", "discarding"].includes(
    snapshot.recording.state,
  );
  const recorderWriting = ["recording", "saving", "discarding"].includes(snapshot.recording.state);

  async function selectAndOpenReplay(path: string) {
    // Navigating after a rejected selection is what made a failed "Open in Replay" look like a
    // no-op: the page changed but kept showing whatever dataset was selected before.
    if (await run(() => api.selectRecordedDataset(path))) {
      navigate("episode-replay");
    }
  }

  async function queueTrajGenAndOpenProcessing(path: string) {
    if (await run(() => api.queueTrajGen(path))) {
      navigate("dataset-processing");
    }
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

  const exportApprovedWithWarningGuard = (path: string, actionMode?: string) => {
    const item = snapshot.processing.find((candidate) => candidate.path === path);
    // Only the QC-warned case asks. A pass exports straight through, and anything else is
    // refused by the gateway with its own reason.
    if (item?.status !== "qc_warn") {
      run(() => api.exportApprovedDataset(path, actionMode));
      return;
    }
    const warnings = qcWarnings(item);
    // Name the action being overridden, not the endpoint it shares: on the workstation this
    // button builds the training view a policy will be trained on, not a v3 export.
    const overriding =
      snapshot.deployment?.profile === "workstation"
        ? "Build the training view anyway?"
        : "Export it anyway?";
    const ok = window.confirm(
      [
        `QC passed with warnings for "${item.name}".`,
        "",
        ...(warnings.length ? warnings : [item.qcSummary]),
        "",
        overriding
      ].join("\n")
    );
    if (!ok) {
      return;
    }
    run(() => api.exportApprovedDataset(path, actionMode, true));
  };

  const pageNode =
    activePage === "teleoperation" ? (
      <TeleoperationPage
        snapshot={snapshot}
        busy={busy}
        onStartSimTeleop={() => run(() => api.startSimTeleop())}
        onStartRealTeleop={() => run(() => api.startRealTeleop())}
        onStopTeleop={() => run(() => api.stopTeleop())}
        cameraUrl={(view, backend) =>
          backend === "real"
            ? api.cameraSnapshotUrl(view.deviceId ?? (view.id === "wrist" ? "ee" : "side"))
            : api.teleopCameraUrl(view.id)
        }
      />
    ) : activePage === "live-record" ? (
      <LiveRecordPage
        snapshot={snapshot}
        busy={busy}
        onConnect={(backend, episodeTimeS) => run(() => api.connectRecording(backend, episodeTimeS))}
        onStart={() => run(() => api.startRecording())}
        onStop={(action) => run(() => api.stopRecording(action))}
        onSetStartPose={() => run(() => api.setRecordingStartPose())}
        onOpenInReplay={() => selectAndOpenReplay(latestRecordedPath)}
        onQueueTrajGen={() => queueTrajGenAndOpenProcessing(firstMissingPath)}
        onGoToProcessing={() => navigate("dataset-processing")}
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
        onMujocoReplay={(mode) => run(() => api.startMujocoReplay(mode))}
        onApproveMujoco={(mode) => run(() => api.approveMujocoReplay(mode))}
        onRealReplay={(mode, robotIp, endEffectorMode, overrideMujocoFailure) =>
          run(() => api.startRealCubeReplay(mode, robotIp, endEffectorMode, overrideMujocoFailure))
        }
        onAbort={() => run(() => api.abortReplay())}
        onSelectDataset={(path) => run(() => api.selectRecordedDataset(path))}
        onSelectEpisode={(episode) => run(() => api.selectReplayEpisode(episode))}
        onDeleteEpisode={(episode) => run(() => api.deleteReplayEpisode(episode))}
        onGenerateForActive={() => {
          if (!workstationProfile && replayMatch) queueTrajGenAndOpenProcessing(replayMatch.path);
        }}
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
        onExportApprovedDataset={(path, actionMode) => exportApprovedWithWarningGuard(path, actionMode)}
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
      <CalibrationPage
        snapshot={snapshot}
        api={api}
        busy={busy}
        onRunMultiCameraCalibration={() => run(() => api.runCalibration())}
      />
    ) : activePage === "device-manager" ? (
      <DeviceManagerPage snapshot={snapshot} />
    ) : activePage === "dashboard" ? (
      <DashboardPage
        snapshot={snapshot}
        busy={busy}
        onConnect={() => run(() => api.connectRecording(undefined, snapshot.configSummary.episodeTimeS))}
        onNavigate={navigate}
      />
    ) : (
      <PlaceholderPage title={activeMeta.label} />
    );

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>{deployment.label}</h1>
          <p>
            {workstationProfile
              ? `FR3 + Pika · SpaceMouse · ${snapshot.devices.filter((d) => d.kind === "camera").length} RealSense cameras`
              : `Thor GMSL2 · ${snapshot.devices.filter((d) => d.kind === "camera").length} cameras · ${snapshot.configSummary.fps} fps`}
          </p>
        </div>
        <div className="topbar-status">
          <span><StatusDot state={snapshot.gateway.state === "online" ? "running" : "warning"} /> Gateway {snapshot.gateway.state}</span>
          {workstationProfile ? (
            <span><StatusDot state={snapshot.teleop.state} /> Teleop {stateLabel(snapshot.teleop.state)}</span>
          ) : (
            <span className="topbar-recorder">
              <StatusDot state={snapshot.recording.state} /> Recorder {stateLabel(snapshot.recording.state)}
              {recorderConnected ? (
                <button
                  className="topbar-btn"
                  disabled={busy || recorderWriting}
                  title={recorderWriting ? "录制进行中：请先在 Live Record 保存/停止" : "断开录制器会话"}
                  onClick={() => run(() => api.stopRecording("exit"))}
                >
                  Disconnect
                </button>
              ) : (
                <button
                  className="topbar-btn topbar-btn-primary"
                  disabled={busy}
                  title="连接录制器（按当前绑定 Task 启动，供录制 / 标定 / 设备预览共用）"
                  onClick={() => run(() => api.connectRecording(undefined, snapshot.configSummary.episodeTimeS))}
                >
                  Connect
                </button>
              )}
            </span>
          )}
          {!workstationProfile && snapshot.configSummary.hardwareSync && (
            <span>
              <StatusDot state={snapshot.configSummary.hardwareSync.enabled ? "running" : "warning"} />
              {" "}HW Sync {snapshot.configSummary.hardwareSync.enabled ? "ON" : "OFF"}
            </span>
          )}
          <span><StatusDot state={snapshot.replay.safety === "fault" ? "error" : snapshot.replay.safety === "active" ? "running" : "idle"} /> Replay safety {snapshot.replay.safety}</span>
        </div>
      </header>
      {commandError ? (
        <div className="command-error" role="alert">
          <span>{commandError}</span>
          <button type="button" onClick={() => setCommandError(null)} aria-label="Dismiss error">
            ×
          </button>
        </div>
      ) : null}
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
  const profile = snapshot.deployment?.profile ?? "thor";
  const visibleNavGroups = navGroups
    .map((group) => ({ ...group, ids: group.ids.filter((id) => pageAllowedForProfile(id, profile)) }))
    .filter((group) => group.ids.length > 0);

  const onlineCount = snapshot.devices.filter((d) => d.state === "running").length;
  const totalCount = snapshot.devices.length;
  const boxDevices = snapshot.devices.filter((d) => d.kind === "box_collection");
  const boxIds = [...new Set(boxDevices.map((d) => deviceBoxId(d)))];
  const boxIdLabel = boxIds.length ? boxIds.map((b) => b || "默认").join(", ") : "—";

  return (
    <nav className="sidebar" aria-label="Robot data factory pages">
      {visibleNavGroups.map((group) => (
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
      {profile === "thor" && <div className="nav-group">
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
      </div>}

      {/* Global device status footer (spec §2). Thor has no ambient temp/humidity
          sensor, so environment shows a target range rather than a fake value. */}
      <div className="nav-footer">
        <div className="nav-footer-row">
          <span className={`status-dot status-${onlineCount === totalCount && totalCount > 0 ? "running" : "warning"}`} />
          <span>{onlineCount} / {totalCount} devices online</span>
        </div>
        {profile === "thor" ? (
          <>
            <div className="nav-footer-row">
              <span className={`status-dot status-${caliBadge.tone}`} />
              <span>Calibration: {caliBadge.text}</span>
            </div>
            <div className="nav-footer-row nav-footer-muted">Env: 15–30℃ / 30–70%RH target</div>
            <div className="nav-footer-row nav-footer-muted">BOX ID: {boxIdLabel}</div>
          </>
        ) : (
          <div className="nav-footer-row nav-footer-muted">FR3 workstation · standalone</div>
        )}
      </div>
    </nav>
  );
}

export default App;
