// Calibration center (标定中心). Composes the global status bar, readiness
// checklist, and — grouped per BOX device — the live monitors and the three
// calibration operation cards (origin + dynamic 6D force, tactile), plus a
// shared operation log and history list.
//
// Backwards compatible with a single BOX: single-box rigs carry an empty
// box_id and bare sensor ids, so `boxGroups` yields one group and every
// per-box loop degrades to a single row.

import { useEffect, useMemo, useRef, useState } from "react";
import type { DataCollectionGuiApi, GuiSnapshot } from "../api";
import type { BoxCaliLogLine, DeviceStatus } from "../types";
import { envTargetLabel, fmtTimestamp } from "./config";
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
import { CalibrationWizard } from "./CalibrationWizard";
import { MarkerTcpPanel } from "./MarkerTcpPanel";
import { HandEyePanel } from "./HandEyePanel";

const OPERATOR_KEY = "lerobot.calibration.operator";

// Only block calibration while the recorder is ACTIVELY writing an episode.
// "armed" (Connected, waiting to start) and "review" (episode finished, awaiting
// save/discard) are connected-but-not-recording, so they must NOT block — a
// successful Connect does not mean data is being recorded.
const RECORDING_BLOCK_STATES = new Set(["recording", "saving", "discarding"]);

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

  // Live calibration log buffers for the unified log panel.
  useEffect(() => {
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
  }, [api]);

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

  const boxGroups = boxIds.map((bid) => {
    const devs = boxDevices.filter((d) => deviceBoxId(d) === bid);
    return {
      boxId: bid,
      devices: devs,
      force: devs.filter((d) => boxSensorSuffix(d.id) === "box_six_d_force"),
      touch: devs.filter((d) => boxSensorSuffix(d.id).startsWith("box_touch")),
    };
  });

  // Per-box, per-kind calibration history (validity + result recall).
  const historyFor = (kind: CalibrationKind, bid: string) =>
    records.filter((r) => r.kind === kind && (r.boxId || "") === bid);

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

      {/* Self-check sits above the calibration itself: the usual question is
          "does anything need redoing?", and the answer is usually no. */}
      {/* Wrapped: this button hands its click event to whatever it is given, and
          the solve now takes an options object. */}
      <RigCheckPanel api={api} busy={busy} onRecalibrate={() => onRunMultiCameraCalibration()} />

      {/* The self-check above only sees pose, and says so -- a bump invalidates
          the extrinsics and leaves the lens alone. This is the other half: the
          intrinsics do not go stale from bumps, they go stale from having been
          fitted on a sweep that never reached the frame edge, which no
          reprojection number on this page can reveal. */}
      <IntrinsicsCoveragePanel api={api} onRecapture={() => onRunMultiCameraCalibration()} />

      {/* Whether a bump happened is the self-check above; whether the world
          frame survived it is a different question, and the one that decides
          if today's absolute poses can be compared with last week's. */}
      <WorldFramePanel api={api} busy={busy} />

      <CalibrationWizard
        snapshot={snapshot}
        api={api}
        busy={busy}
        onSolve={onRunMultiCameraCalibration}
      />

      <MarkerTcpPanel snapshot={snapshot} api={api} busy={busy} />

      {/* The pivot above measures translation and is structurally blind to
          rotation -- a spherical joint leaves it in the null space. This is the
          other half of the same constant, and the larger of the two errors: the
          rotation in production is a declared 2.0 deg that was never measured. */}
      <HandEyePanel api={api} busy={busy} />

      {/* --- per-BOX groups: readiness + monitors + calibration by device --- */}
      {boxGroups.map((group) => (
        <BoxCalibrationGroup
          key={group.boxId || "single"}
          api={api}
          group={group}
          operator={operator}
          guard={guard}
          onRecord={onRecord}
          historyFor={historyFor}
          now={now}
          recorderConnected={recorderConnected}
        />
      ))}

      {boxDevices.length === 0 && (
        <section className="panel">
          <p className="cali-muted">
            未检测到 BOX 传感器。请先点顶栏的「Connect」连接 GMSL2/BOX 录制器（录制器供录制 / 标定 / 设备预览共用）。
          </p>
        </section>
      )}

      {/* --- log + history (spec §10, §11) --- */}
      <CalibrationLogPanel entries={logEntries} />
      <CalibrationHistory
        records={records}
        onClear={() => {
          historyRepo.clear();
          setHistoryTick((n) => n + 1);
        }}
      />
    </div>
  );
}

type BoxGroup = {
  boxId: string;
  devices: DeviceStatus[];
  force: DeviceStatus[];
  touch: DeviceStatus[];
};

// One BOX device group: its own readiness (connected / warm-up timer / tactile
// activation / validity), live monitors, and calibration cards. Extracted so the
// per-box readiness hooks (useWarmup / useTactileActivation) obey the rules of
// hooks — they can't be called inside the boxGroups.map() callback.
function BoxCalibrationGroup({
  api,
  group,
  operator,
  guard,
  onRecord,
  historyFor,
  now,
  recorderConnected,
}: {
  api: DataCollectionGuiApi;
  group: BoxGroup;
  operator: string;
  guard: () => string | null;
  onRecord: (records: CalibrationRecord[]) => void;
  historyFor: (kind: CalibrationKind, bid: string) => CalibrationRecord[];
  now: number;
  recorderConnected: boolean;
}) {
  const bid = group.boxId;
  const hasTouch = group.touch.length > 0;
  const online = group.devices.some((d) => d.state === "running") || recorderConnected;
  const fw = firmwareVersion(group.devices);
  const sn = boxSerial(group.devices);

  const warmup = useWarmup(bid, now);
  const tactile = useTactileActivation(api, bid, group.touch, hasTouch);

  // This box's calibration validity, across only the kinds it actually has.
  const kinds: CalibrationKind[] = [
    ...(group.force.length ? (["force_origin", "force_dynamic"] as CalibrationKind[]) : []),
    ...(hasTouch ? (["touch"] as CalibrationKind[]) : []),
  ];
  const validityStates = kinds.map((k) => computeValidity(historyFor(k, bid)[0]?.timestamp ?? null, k).state);
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
      label: "必须标定仍在有效期",
      state: needs > 0 ? "failed" : due > 0 ? "warning" : kinds.length ? "complete" : "unavailable",
      detail:
        needs > 0 ? `${needs} 项需标定` : due > 0 ? `${due} 项即将过期` : kinds.length ? "全部有效" : "无可标定传感器",
    },
  ];

  return (
    <section className="panel cali-box-panel">
      <div className="panel-heading">
        <h2>{bid ? `BOX ${boxDisplayName(bid)}` : boxDisplayName(bid)}</h2>
        <span>
          固件 {fw}
          {sn ? ` · SN ${sn}` : ""} · {group.devices.filter((d) => d.state === "running").length}/
          {group.devices.length} 在线
        </span>
      </div>

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

      {/* calibration operations for this box */}
      {group.force.length > 0 && (
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
