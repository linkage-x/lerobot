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
  readinessTactileActivation,
  readinessWarmup,
  softwareVersion,
} from "./adapters";
import { computeValidity } from "./status";
import type { CalibrationKind, CalibrationRecord, ReadinessItem } from "./types";
import { ReadinessChecklist } from "./ReadinessChecklist";
import { ForceSensorCard, TactileSensorCard } from "./SensorMonitors";
import { ForceCalibrationCard, TactileCalibrationCard } from "./CalibrationCards";
import { CalibrationLogPanel, toLogEntries } from "./CalibrationLogPanel";
import { CalibrationHistory } from "./CalibrationHistory";
import { MultiCameraCalibrationPanel } from "./MultiCameraCalibrationPanel";

const OPERATOR_KEY = "lerobot.calibration.operator";

const RECORDING_BLOCK_STATES = new Set(["recording", "saving", "discarding", "armed"]);

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
  onRunMultiCameraCalibration: () => void;
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

  const historyByKind = (kind: CalibrationKind) => records.filter((r) => r.kind === kind);
  const latestByKind: Partial<Record<CalibrationKind, number | null>> = {
    force_origin: historyByKind("force_origin")[0]?.timestamp ?? null,
    force_dynamic: historyByKind("force_dynamic")[0]?.timestamp ?? null,
    touch: historyByKind("touch")[0]?.timestamp ?? null,
  };
  const validitySummary = summarizeKinds(latestByKind, now);

  // --- readiness items -------------------------------------------------------
  const readiness: ReadinessItem[] = [
    {
      id: "box-connected",
      label: "BOX 已连接",
      state: boxOnline ? "complete" : "pending",
      detail: boxOnline ? "在线" : "未连接",
    },
    readinessWarmup(),
    readinessTactileActivation(),
    readinessEnvironment(),
    {
      id: "cali-valid",
      label: "所有必须标定的传感器仍在有效期",
      state:
        validitySummary.overdue > 0 || validitySummary.unknown > 0
          ? "failed"
          : validitySummary.dueSoon > 0
            ? "warning"
            : "complete",
      detail:
        validitySummary.overdue + validitySummary.unknown > 0
          ? `${validitySummary.overdue + validitySummary.unknown} 项需标定`
          : validitySummary.dueSoon > 0
            ? `${validitySummary.dueSoon} 项即将过期`
            : "全部有效",
    },
  ];

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

      <ReadinessChecklist items={readiness} />

      <MultiCameraCalibrationPanel
        status={snapshot.calibration}
        busy={busy}
        onRun={onRunMultiCameraCalibration}
      />

      {/* --- per-BOX groups: monitors + calibration nested by device (spec §3-6) --- */}
      {boxGroups.map((group) => {
        const online = group.devices.filter((d) => d.state === "running").length;
        const fw = firmwareVersion(group.devices);
        const sn = boxSerial(group.devices);
        return (
          <section className="panel cali-box-panel" key={group.boxId || "single"}>
            <div className="panel-heading">
              <h2>BOX {boxDisplayName(group.boxId)}</h2>
              <span>
                固件 {fw}
                {sn ? ` · SN ${sn}` : ""} · {online}/{group.devices.length} 在线
              </span>
            </div>

            {/* live monitors for this box */}
            <div className="cali-monitor-grid">
              {group.force.map((d) => (
                <ForceSensorCard key={d.id} api={api} device={d} />
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
                  boxId={group.boxId}
                  devices={group.force}
                  operator={operator}
                  guard={guard}
                  onRecord={onRecord}
                  history={historyFor("force_origin", group.boxId)}
                />
                <ForceCalibrationCard
                  variant="dynamic"
                  api={api}
                  boxId={group.boxId}
                  devices={group.force}
                  operator={operator}
                  guard={guard}
                  onRecord={onRecord}
                  history={historyFor("force_dynamic", group.boxId)}
                />
              </div>
            )}

            {group.touch.length > 0 && (
              <TactileCalibrationCard
                api={api}
                boxId={group.boxId}
                devices={group.touch}
                operator={operator}
                guard={guard}
                onRecord={onRecord}
                history={historyFor("touch", group.boxId)}
              />
            )}
          </section>
        );
      })}

      {boxDevices.length === 0 && (
        <section className="panel">
          <p className="cali-muted">
            未检测到 BOX 传感器。请先在「设备 · Device Manager」连接 GMSL2/BOX 录制器。
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
