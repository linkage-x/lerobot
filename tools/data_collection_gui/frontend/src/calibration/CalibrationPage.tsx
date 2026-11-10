// Calibration center (标定中心). A status bar on top, then one tab per subject
// (cameras, hand-held end effector, FR3, BOX sensors, laser tracker, log), so
// the page shows the dozen panels a few at a time instead of one long scroll.
// The overview tab answers the usual question first -- does anything need
// redoing? -- and says which tab to go to for what.
//
// Panels mount only on their own tab: their polling stops when hidden, and
// their jobs live on the gateway, so switching tabs loses nothing.
//
// Backwards compatible with a single BOX: single-box rigs carry an empty
// box_id and bare sensor ids, so `boxGroups` yields one group and every
// per-box loop degrades to a single row.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { DataCollectionGuiApi, GuiSnapshot } from "../api";
import type { BoxCaliLogLine, DeviceStatus } from "../types";
import { envTargetLabel, fmtTimestamp } from "./config";
import { userGuideUrl } from "../shared/ui";
import {
  boxDisplayName,
  boxSensorSuffix,
  boxSerial,
  deviceBoxId,
  firmwareVersion,
  historyRepo,
  softwareVersion,
} from "./adapters";
import { computeValidity } from "./status";
import type { CalibrationKind, CalibrationRecord, ReadinessItem } from "./types";
import { ReadinessChecklist } from "./ReadinessChecklist";
import { useTactileActivation, useWarmup } from "./useReadiness";
import { ForceSensorCard, ForceStaticValidationCard, TactileSensorCard } from "./SensorMonitors";
import { ForceCalibrationCard, TactileCalibrationCard } from "./CalibrationCards";
import { CalibrationLogPanel, toLogEntries } from "./CalibrationLogPanel";
import { CalibrationHistory } from "./CalibrationHistory";
import { RigCheckPanel } from "./RigCheckPanel";
import { IntrinsicsCoveragePanel } from "./IntrinsicsCoveragePanel";
import { WorldFramePanel } from "./WorldFramePanel";
import { Fr3BaseEdgePanel } from "./Fr3BaseEdgePanel";
import { Fr3PivotPanel } from "./Fr3PivotPanel";
import { CrossCameraPanel } from "./CrossCameraPanel";
import { CalibrationWizard } from "./CalibrationWizard";
import { MarkerTcpPanel } from "./MarkerTcpPanel";
import { HandEyePanel } from "./HandEyePanel";
import { TrackerMountPanel } from "./TrackerMountPanel";
import { Fr3PayloadPanel } from "./Fr3PayloadPanel";
import { Fr3SixDZeroPanel } from "./Fr3SixDZeroPanel";
import { SixDFitPanel } from "./SixDFitPanel";
import { CALIBRATION_TABS, readStoredTab, storeTab, type CalibrationTabId } from "./calibrationTabs";
import type { PayloadCalibrationResponse, ZeroReport } from "./payloadCalibration";
import { armZeroTimes, boxDeviceNumber, boxRoleRank, resolveBoxRole, type BoxRole } from "./boxRoles";

const OPERATOR_KEY = "lerobot.calibration.operator";

// Only block calibration while the recorder is ACTIVELY writing an episode.
// "armed" (Connected, waiting to start) and "review" (episode finished, awaiting
// save/discard) are connected-but-not-recording, so they must NOT block — a
// successful Connect does not mean data is being recorded.
const RECORDING_BLOCK_STATES = new Set(["recording", "saving", "discarding"]);
// The BOX roles and the arm-held zero reports change rarely; poll them only while the BOX tab is open.
const BOX_ROLE_POLL_MS = 15000;

function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(t);
  }, [intervalMs]);
  return now;
}

export function CalibrationPage({
  snapshot,
  api,
  busy,
  onRunMultiCameraCalibration,
}: {
  snapshot: GuiSnapshot;
  api: DataCollectionGuiApi;
  busy: boolean;
  onRunMultiCameraCalibration: (
    options?: { forceRedetect?: boolean; refitIntrinsics?: boolean; experiment?: boolean }
  ) => void;
}) {
  const now = useNow();
  const [operator, setOperator] = useState<string>(
    () => window.localStorage.getItem(OPERATOR_KEY) ?? "",
  );
  const [historyTick, setHistoryTick] = useState(0);
  const [forceLog, setForceLog] = useState<BoxCaliLogLine[]>([]);
  const [touchLog, setTouchLog] = useState<BoxCaliLogLine[]>([]);
  const refreshRef = useRef<() => void>(() => {});
  const [tab, setTabState] = useState<CalibrationTabId>(() => readStoredTab());
  const setTab = (next: CalibrationTabId) => {
    setTabState(next);
    storeTab(next);
    window.scrollTo({ top: 0 });
  };
  // the BOX calibration logs only matter on the tabs that show them
  const wantsCaliLog = tab === "box" || tab === "log";

  // Which BOX is which (box_devices.json, fr3_arms.json) and when each arm BOX was last zeroed by the
  // arm route: fetched once for the tab badge, polled while the BOX tab is open.
  const [payloadCal, setPayloadCal] = useState<PayloadCalibrationResponse | null>(null);
  const refreshPayloadCal = useCallback(async () => {
    const next = await api.fetchPayloadCalibration();
    if (next) setPayloadCal(next);
  }, [api]);
  useEffect(() => {
    void refreshPayloadCal();
    if (tab !== "box") return;
    const timer = window.setInterval(() => void refreshPayloadCal(), BOX_ROLE_POLL_MS);
    return () => window.clearInterval(timer);
  }, [refreshPayloadCal, tab]);

  // Live calibration log buffers for the unified log panel.
  useEffect(() => {
    if (!wantsCaliLog) return;
    let cancelled = false;
    const load = async () => {
      const [f, t] = await Promise.all([api.fetchBoxCaliLog(), api.fetchBoxTouchCaliLog()]);
      if (cancelled) return;
      if (f) setForceLog(f.lines);
      if (t) setTouchLog(t.lines);
    };
    refreshRef.current = load;
    load();
    const timer = window.setInterval(load, 1500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [api, wantsCaliLog]);

  const setOperatorPersist = (value: string) => {
    setOperator(value);
    try {
      window.localStorage.setItem(OPERATOR_KEY, value);
    } catch {
      /* ignore storage denial */
    }
  };

  const records = useMemo(() => historyRepo.list(), [historyTick]);
  const onRecord = (recs: CalibrationRecord[]) => {
    recs.forEach((r) => historyRepo.add(r));
    setHistoryTick((n) => n + 1);
  };

  // --- device grouping: by BOX device, sensors nested (single-box safe) ------
  const boxDevices = snapshot.devices.filter((d) => d.kind === "box_collection");
  const boxIds = [...new Set(boxDevices.map((d) => deviceBoxId(d)))];
  const boxIdLabel = boxIds.length ? boxIds.map((b) => b || "默认").join(", ") : "—";

  const boxGroups: BoxGroup[] = boxIds
    .map((bid) => {
      const devs = boxDevices.filter((d) => deviceBoxId(d) === bid);
      return {
        boxId: bid,
        devices: devs,
        force: devs.filter((d) => boxSensorSuffix(d.id) === "box_six_d_force"),
        touch: devs.filter((d) => boxSensorSuffix(d.id).startsWith("box_touch")),
        role: resolveBoxRole(boxDeviceNumber(bid, devs), payloadCal),
      };
    })
    .sort((a, b) => boxRoleRank(a.role) - boxRoleRank(b.role));
  // An arm's gripper BOX that is not online (recorder not connected): still listed, with its zero
  // panel and reports, so the arm route is found here whether or not the BOX is up.
  const offlineArmBoxes = (payloadCal?.arms ?? [])
    .filter((a) => a.gripperBox != null && !boxGroups.some((g) => g.role.deviceId === String(a.gripperBox)))
    .map((a) => resolveBoxRole(String(a.gripperBox), payloadCal));

  // Per-box, per-kind calibration history (validity + result recall).
  const historyFor = (kind: CalibrationKind, bid: string) =>
    records.filter((r) => r.kind === kind && (r.boxId || "") === bid);

  // When each kind was last done for a BOX. An arm BOX's 6D zeros are dated by the arm route's reports
  // on Thor; a manual zero from this browser does not count for it (it may have been at any attitude).
  const zeroReports: ZeroReport[] = [...(payloadCal?.zeroLatest ?? []), ...(payloadCal?.zeroReports ?? [])];
  const lastRunFor = (kind: CalibrationKind, group: BoxGroup): number | null => {
    if (group.role.kind === "fr3_gripper" && kind !== "touch") {
      return armZeroTimes(zeroReports, group.role.arm ?? "", group.role.deviceId)[kind] ?? null;
    }
    return historyFor(kind, group.boxId)[0]?.timestamp ?? null;
  };

  const recorderConnected = ["connecting", "armed", "recording", "review", "saving", "discarding"].includes(
    snapshot.recording.state,
  );
  const boxOnline = boxDevices.some((d) => d.state === "running") || recorderConnected;
  const recording = RECORDING_BLOCK_STATES.has(snapshot.recording.state);

  const guard = (): string | null => {
    if (recording) return "正在录制数据，已暂停标定（避免干扰采集）";
    return null;
  };

  const logEntries = useMemo(() => toLogEntries(forceLog, touchLog), [forceLog, touchLog]);

  // BOX tab badge: calibrations past their validity, over every BOX and kind it has
  const boxOverdue = boxGroups.reduce(
    (n, g) =>
      n +
      groupKinds(g).filter((k) => {
        const st = computeValidity(lastRunFor(k, g), k, now).state;
        return st === "overdue" || st === "unknown";
      }).length,
    0,
  );
  const badges: Partial<Record<CalibrationTabId, { text: string; tone: "error" | "warning" | "running" }>> = {
    ...(boxOverdue ? { box: { text: String(boxOverdue), tone: "error" as const } } : {}),
    ...(recording ? { overview: { text: "录制中", tone: "warning" as const } } : {}),
  };

  return (
    <div className="page-stack">
      {/* --- global status bar (spec §1) --- */}
      <section className="panel cali-statusbar">
        <div className="cali-statusbar-main">
          <span className="cali-statusbar-item">
            <span className={`status-dot status-${boxOnline ? "running" : "idle"}`} />
            BOX {boxOnline ? "已连接" : "未连接"}
          </span>
          <span className="cali-statusbar-item">BOX ID：{boxIdLabel}</span>
          <span className="cali-statusbar-item">GUI {softwareVersion()}</span>
          <span className="cali-statusbar-item">环境 {envTargetLabel()}</span>
          <span className="cali-statusbar-item">{fmtTimestamp(now)}</span>
          <span className="cali-statusbar-item">
            采集：{snapshot.recording.state}
          </span>
          <span className={`cali-badge cali-badge-${recording ? "error" : "running"}`}>
            <span className={`status-dot status-${recording ? "error" : "running"}`} />
            {recording ? "标定已暂停" : "允许标定"}
          </span>
        </div>
        <div className="cali-statusbar-side">
          <label className="cali-operator">
            操作者
            <input
              value={operator}
              placeholder="姓名/工号"
              onChange={(e) => setOperatorPersist(e.target.value)}
            />
          </label>
          <button onClick={() => refreshRef.current()}>刷新</button>
        </div>
      </section>

      <nav className="cali-tabs" role="tablist" aria-label="标定分类">
        {CALIBRATION_TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            className={tab === t.id ? "cali-tab active" : "cali-tab"}
            title={t.hint}
            onClick={() => setTab(t.id)}
          >
            {t.label}
            {badges[t.id] ? <small className={`nav-badge nav-badge-${badges[t.id]!.tone}`}>{badges[t.id]!.text}</small> : null}
          </button>
        ))}
      </nav>

      {tab === "overview" && (
        <>
          {/* Self-check sits above the calibration itself: the usual question is
              "does anything need redoing?", and the answer is usually no. */}
          {/* Wrapped: this button hands its click event to whatever it is given, and
              the solve now takes an options object. */}
          <RigCheckPanel api={api} busy={busy} onRecalibrate={() => onRunMultiCameraCalibration()} />
          <section className="panel">
            <div className="panel-heading">
              <h2>什么时候做哪项标定</h2>
            </div>
            <div className="cali-map">
              {CALIBRATION_TABS.filter((t) => t.id !== "overview").map((t) => (
                <button key={t.id} className="cali-map-card" onClick={() => setTab(t.id)}>
                  <strong>
                    {t.label}
                    {badges[t.id] ? <small className={`nav-badge nav-badge-${badges[t.id]!.tone}`}>{badges[t.id]!.text}</small> : null}
                  </strong>
                  <span>{t.when}</span>
                  <em>{t.contents}</em>
                </button>
              ))}
            </div>
          </section>
        </>
      )}

      {tab === "cameras" && (
        <>
          <CalibrationWizard snapshot={snapshot} api={api} busy={busy} onSolve={onRunMultiCameraCalibration} />

          {/* The self-check only sees pose, and says so -- a bump invalidates
              the extrinsics and leaves the lens alone. This is the other half: the
              intrinsics do not go stale from bumps, they go stale from having been
              fitted on a sweep that never reached the frame edge, which no
              reprojection number on this page can reveal. */}
          <IntrinsicsCoveragePanel api={api} onRecapture={() => onRunMultiCameraCalibration()} />

          {/* Whether a bump happened is the self-check; whether the world
              frame survived it is a different question, and the one that decides
              if today's absolute poses can be compared with last week's. */}
          <WorldFramePanel api={api} busy={busy} />

          {/* The checks above compare with the past, so an extrinsic that was
              wrong from the day it was solved passes them. This one compares the
              cameras with each other, in the workspace, now. */}
          <CrossCameraPanel api={api} busy={busy} />
        </>
      )}

      {tab === "handheld" && (
        <>
          <MarkerTcpPanel snapshot={snapshot} api={api} busy={busy} />

          {/* The pivot above measures translation and is structurally blind to
              rotation -- a spherical joint leaves it in the null space. This is the
              other half of the same constant, and the larger of the two errors: the
              rotation in production is a declared 2.0 deg that was never measured. */}
          <HandEyePanel api={api} busy={busy} />
        </>
      )}

      {tab === "fr3" && (
        <>
          <p className="panel-note">
            把一台臂接入回放（负载 → 球窝核对 → 同球检验 → 换边）的完整步骤和验收标准：{" "}
            <a href={userGuideUrl("left-arm")} target="_blank" rel="noopener noreferrer">使用指南 §10 启用左臂</a>
          </p>
          {/* What hangs below the flange, for Desk: every other FR3 panel moves or
              holds the arm with that load configured, so it comes first. */}
          <Fr3PayloadPanel api={api} busy={busy} />

          {/* The gripper BOX's 6D zero with the arm as the vendor's fixture (fingertips up) lives
              with that BOX on the BOX tab, next to its readiness and validity. */}
          <section className="panel">
            <div className="cali-result-head">
              <h3>夹爪 6D 零点（机械臂当工装，指尖朝上）</h3>
              <button className="cali-mini-btn" onClick={() => setTab("box")}>
                去「BOX 传感器」页
              </button>
            </div>
            <p className="cali-muted">
              已移到「BOX 传感器」页对应的臂上 BOX 那一组（如「右臂夹爪 BOX」）：那里同时显示这只 BOX 的预热、有效期和零点记录。
              每 30 分钟一次动态校零，每天第一次做 origin；机械臂会动，要有人守急停。
            </p>
          </section>

          {/* The one edge that leaves the world for the FR3 base. Every replay and
              fr3_base export goes through it. */}
          <Fr3BaseEdgePanel api={api} busy={busy} />

          {/* A referee for that edge that needs no laser and no TCP definition:
              the FR3 and the handheld pivot about one ball on the table. */}
          <Fr3PivotPanel api={api} busy={busy} />
        </>
      )}

      {tab === "box" && (
        <>
          {/* --- per-BOX groups: readiness + monitors + calibration by device --- */}
          {boxGroups.map((group) => (
            <BoxCalibrationGroup
              key={group.boxId || "single"}
              api={api}
              busy={busy}
              group={group}
              operator={operator}
              guard={guard}
              onRecord={onRecord}
              historyFor={historyFor}
              lastRunFor={lastRunFor}
              now={now}
              recorderConnected={recorderConnected}
              onZeroEnded={() => void refreshPayloadCal()}
            />
          ))}

          {offlineArmBoxes.map((role) => (
            <section key={role.deviceId} className="panel cali-box-panel cali-box-arm">
              <div className="panel-heading">
                <h2>{role.title}</h2>
                <span>不在线</span>
              </div>
              <p className="cali-muted cali-box-role-note">
                录制器没有 Connect，或这只 BOX 没上电。零点要先在顶栏 Connect（校零经录制器做）；下面的记录照常可看。
              </p>
              {role.note ? <p className="cali-warn cali-box-role-note">{role.note}</p> : null}
              <Fr3SixDZeroPanel
                api={api}
                busy={busy}
                fixedArm={role.arm || undefined}
                embedded
                onJobEnded={() => void refreshPayloadCal()}
              />
            </section>
          ))}

          {boxDevices.length === 0 && (
            <section className="panel">
              <p className="cali-muted">
                未检测到 BOX 传感器。零点/origin/触觉校准要先点顶栏的「Connect」连接 GMSL2/BOX 录制器（录制器供录制 / 标定 / 设备预览共用）。
                下面的 6D 拟合反过来要求录制器断开。
              </p>
            </section>
          )}

          {/* The zero (above) shifts the reading at one attitude; K takes out the
              payload's weight at every attitude. Both are needed for a usable 6D. */}
          <SixDFitPanel api={api} busy={busy} />
        </>
      )}

      {tab === "laser" && (
        /* The laser tracker's own two constants. Neither is readable from CAD:
           T_WG depends on where the tracker was parked, c on where the plate was
           bolted this time. This fit absorbs the marker->TCP constant, so it can
           measure everything about the cube pose *except* what those panels are for. */
        <TrackerMountPanel api={api} busy={busy} snapshot={snapshot} />
      )}

      {tab === "log" && (
        <>
          <CalibrationLogPanel entries={logEntries} />
          <CalibrationHistory
            records={records}
            onClear={() => {
              historyRepo.clear();
              setHistoryTick((n) => n + 1);
            }}
          />
        </>
      )}
    </div>
  );
}

type BoxGroup = {
  boxId: string;
  devices: DeviceStatus[];
  force: DeviceStatus[];
  touch: DeviceStatus[];
  role: BoxRole;
};

/** The calibration kinds a BOX has, by the sensors it actually reports. */
function groupKinds(group: BoxGroup): CalibrationKind[] {
  return [
    ...(group.force.length ? (["force_origin", "force_dynamic"] as CalibrationKind[]) : []),
    ...(group.touch.length ? (["touch"] as CalibrationKind[]) : []),
  ];
}

// One BOX device group: its own readiness (connected / warm-up timer / tactile
// activation / validity), live monitors, and calibration cards. Extracted so the
// per-box readiness hooks (useWarmup / useTactileActivation) obey the rules of
// hooks — they can't be called inside the boxGroups.map() callback.
function BoxCalibrationGroup({
  api,
  busy,
  group,
  operator,
  guard,
  onRecord,
  historyFor,
  lastRunFor,
  now,
  recorderConnected,
  onZeroEnded,
}: {
  api: DataCollectionGuiApi;
  busy: boolean;
  group: BoxGroup;
  operator: string;
  guard: () => string | null;
  onRecord: (records: CalibrationRecord[]) => void;
  historyFor: (kind: CalibrationKind, bid: string) => CalibrationRecord[];
  lastRunFor: (kind: CalibrationKind, group: BoxGroup) => number | null;
  now: number;
  recorderConnected: boolean;
  onZeroEnded: () => void;
}) {
  const bid = group.boxId;
  const role = group.role;
  const onArm = role.kind === "fr3_gripper";
  // an arm BOX's generic zero cards stay hidden until the operator confirms the arm is fingertips up
  const [manualZero, setManualZero] = useState(false);
  const hasTouch = group.touch.length > 0;
  const online = group.devices.some((d) => d.state === "running") || recorderConnected;
  const fw = firmwareVersion(group.devices);
  const sn = boxSerial(group.devices);

  const warmup = useWarmup(bid, now);
  const tactile = useTactileActivation(api, bid, group.touch, hasTouch);

  // This box's calibration validity, across only the kinds it actually has.
  const kinds = groupKinds(group);
  const validityStates = kinds.map((k) => computeValidity(lastRunFor(k, group), k, now).state);
  const needs = validityStates.filter((s) => s === "overdue" || s === "unknown").length;
  const due = validityStates.filter((s) => s === "due_soon").length;

  const warmupAction =
    warmup.state === "complete" ? (
      <button className="cali-mini-btn" onClick={warmup.reset}>
        重置
      </button>
    ) : (
      <>
        {!warmup.running && (
          <button className="cali-mini-btn" onClick={warmup.start}>
            开始计时
          </button>
        )}
        <button className="cali-mini-btn" onClick={warmup.confirm}>
          确认已预热
        </button>
        {warmup.running && (
          <button className="cali-mini-btn" onClick={warmup.reset}>
            重置
          </button>
        )}
      </>
    );

  const items: ReadinessItem[] = [
    {
      id: `${bid}-connected`,
      label: "BOX 已连接",
      state: online ? "complete" : "pending",
      detail: online ? "在线" : "未连接",
    },
    {
      id: `${bid}-warmup`,
      label: "设备已预热 30 分钟",
      state: warmup.state,
      detail: warmup.detail,
      action: warmupAction,
    },
    ...(hasTouch
      ? [
          {
            id: `${bid}-tactile`,
            label: "每个 pad 各完成 3 次满量程激活",
            state: tactile.state,
            detail: tactile.detail,
            action: (
              <button
                className="cali-mini-btn"
                disabled={tactile.state === "pending"}
                onClick={tactile.reset}
              >
                重置计数
              </button>
            ),
          } satisfies ReadinessItem,
        ]
      : []),
    {
      id: `${bid}-valid`,
      label: onArm ? "必须标定仍在有效期（6D 零点按机械臂流程的记录算）" : "必须标定仍在有效期",
      state: needs > 0 ? "failed" : due > 0 ? "warning" : kinds.length ? "complete" : "unavailable",
      detail:
        needs > 0 ? `${needs} 项需标定` : due > 0 ? `${due} 项即将过期` : kinds.length ? "全部有效" : "无可标定传感器",
    },
  ];

  return (
    <section className={onArm ? "panel cali-box-panel cali-box-arm" : "panel cali-box-panel"}>
      <div className="panel-heading">
        <h2>{role.deviceId ? role.title : bid ? `BOX ${boxDisplayName(bid)}` : boxDisplayName(bid)}</h2>
        <span>
          固件 {fw}
          {sn ? ` · SN ${sn}` : ""} · {group.devices.filter((d) => d.state === "running").length}/
          {group.devices.length} 在线
        </span>
      </div>

      {role.note ? <p className="cali-warn cali-box-role-note">{role.note}</p> : null}

      {/* per-box collection readiness */}
      <ReadinessChecklist items={items} />

      {/* live monitors for this box */}
      <div className="cali-monitor-grid">
        {group.force.map((d) => (
          <ForceSensorCard key={d.id} api={api} device={d} />
        ))}
        {group.force.map((d) => (
          <ForceStaticValidationCard key={`${d.id}-static`} api={api} device={d} />
        ))}
        {group.touch.map((d) => (
          <TactileSensorCard key={d.id} api={api} device={d} />
        ))}
      </div>

      {/* An arm BOX: its fixture is the arm, fingertips up, so its 6D zero is the arm route. */}
      {onArm && group.force.length > 0 && (
        <>
          <Fr3SixDZeroPanel api={api} busy={busy} fixedArm={role.arm || undefined} embedded onJobEnded={onZeroEnded} />
          <details className="cali-advanced" onToggle={(e) => !e.currentTarget.open && setManualZero(false)}>
            <summary className="cali-muted">高级：不动机械臂，直接手动校零（一般不要用）</summary>
            <p className="cali-warn">
              手动校零会在机械臂<b>当前</b>的姿态下归零。不在指尖朝上的零点姿态时，之后每个姿态的 6D 读数都会带着偏差，而且不会报错。
              这里的记录只存在这台电脑的浏览器里，<b>不算进上面的有效期</b>；正常请用上面的机械臂流程。
            </p>
            <label className="cali-check">
              <input type="checkbox" checked={manualZero} onChange={(e) => setManualZero(e.target.checked)} />
              我确认机械臂已经静止在指尖朝上的零点姿态（例如上面的流程中途停在那里）
            </label>
          </details>
        </>
      )}

      {/* calibration operations for this box */}
      {group.force.length > 0 && (!onArm || manualZero) && (
        <div className="cali-op-grid">
          <ForceCalibrationCard
            variant="origin"
            api={api}
            boxId={bid}
            devices={group.force}
            operator={operator}
            guard={guard}
            onRecord={onRecord}
            history={historyFor("force_origin", bid)}
          />
          <ForceCalibrationCard
            variant="dynamic"
            api={api}
            boxId={bid}
            devices={group.force}
            operator={operator}
            guard={guard}
            onRecord={onRecord}
            history={historyFor("force_dynamic", bid)}
          />
        </div>
      )}

      {hasTouch && (
        <TactileCalibrationCard
          api={api}
          boxId={bid}
          devices={group.touch}
          operator={operator}
          guard={guard}
          onRecord={onRecord}
          history={historyFor("touch", bid)}
        />
      )}
    </section>
  );
}
