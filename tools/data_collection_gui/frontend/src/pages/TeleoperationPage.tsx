import { useEffect, useRef, useState } from "react";

import type { GuiSnapshot } from "../api";
import { Metric, PageHeader, StatusDot } from "../shared/ui";
import type { TeleopCameraView, TeleopGainField, TeleopGains, TeleopGainValues } from "../types";

const defaultCameraViews: TeleopCameraView[] = [
  { id: "external", label: "External", source: "D435I", fps: 30, deviceId: "side" },
  { id: "wrist", label: "Wrist", source: "D405", fps: 30, deviceId: "ee" }
];

type TeleopBackend = "mujoco" | "real";

function TeleopCameraTile({
  view,
  active,
  backend,
  cameraUrl
}: {
  view: TeleopCameraView;
  active: boolean;
  backend: TeleopBackend;
  cameraUrl: (view: TeleopCameraView, backend: TeleopBackend) => string;
}) {
  const [src, setSrc] = useState("");
  const [frameReady, setFrameReady] = useState(false);
  const timerRef = useRef<number | null>(null);
  const cameraUrlRef = useRef(cameraUrl);
  cameraUrlRef.current = cameraUrl;

  const scheduleNext = (delayMs: number) => {
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => setSrc(cameraUrlRef.current(view, backend)), delayMs);
  };

  useEffect(() => {
    if (active) {
      setFrameReady(false);
      setSrc(cameraUrlRef.current(view, backend));
    } else {
      setSrc("");
      setFrameReady(false);
    }
    return () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    };
  }, [active, backend, view.id]);

  return (
    <div className="teleop-camera-tile">
      <div className="teleop-camera-media">
        {src ? (
          <img
            src={src}
            alt=""
            onLoad={() => {
              setFrameReady(true);
              scheduleNext(Math.max(Math.round(1000 / view.fps), 33));
            }}
            onError={() => {
              setFrameReady(false);
              scheduleNext(300);
            }}
          />
        ) : (
          <div className="teleop-camera-offline">OFFLINE</div>
        )}
        <span className="teleop-camera-state">
          <StatusDot state={active && frameReady ? "running" : "idle"} />
          {active && frameReady ? "LIVE" : "OFFLINE"}
        </span>
      </div>
      <div className="teleop-camera-label">
        <strong>{view.label}</strong>
        <span>{view.source} · {view.fps} fps</span>
      </div>
    </div>
  );
}

// The order the gains are shown in, and what each one drives. The per-axis rows sit under the
// global gain they override so it reads as "this, unless the axis says otherwise".
export const GAIN_ROWS: Array<{ field: TeleopGainField; label: string; hint: string; group: "translation" | "rotation" }> = [
  { field: "translation_scale", label: "Translation", hint: "all three axes, m per device tick", group: "translation" },
  { field: "scale_x", label: "X", hint: "base +x, overrides translation", group: "translation" },
  { field: "scale_y", label: "Y", hint: "base +y, overrides translation", group: "translation" },
  { field: "scale_z", label: "Z", hint: "base +z, overrides translation", group: "translation" },
  { field: "rotation_scale", label: "Rotation", hint: "all three axes, rad per device tick", group: "rotation" },
  { field: "scale_wx", label: "Roll (wx)", hint: "overrides rotation; 0 disables the axis", group: "rotation" },
  { field: "scale_wy", label: "Pitch (wy)", hint: "overrides rotation; 0 disables the axis", group: "rotation" },
  { field: "scale_wz", label: "Yaw (wz)", hint: "overrides rotation; 0 disables the axis", group: "rotation" }
];

// The SpaceMouse is polled at this rate by the recorder config, so a gain times this rate is the
// commanded speed at full stick deflection -- the number an operator can actually feel.
const SPACEMOUSE_POLL_HZ = 200;

const formatGain = (value: number | null | undefined): string =>
  value === null || value === undefined ? "" : String(value);

/**
 * What an axis will actually do, resolving the "unset means follow the global" fallback.
 *
 * The fallback is not the bare global: the teleoperator multiplies it by a per-axis calibration
 * first, so an unset z runs at 59% of `translation_scale`. An axis that *is* set replaces the
 * calibrated value rather than scaling it, which is why typing the global's own number into z is
 * not a no-op. Both halves come from teleop_spacemouse.py through the snapshot.
 */
export function effectiveAxisGain(
  values: TeleopGainValues,
  row: (typeof GAIN_ROWS)[number],
  axisCalibration: TeleopGains["axisCalibration"]
): number | null {
  const own = values[row.field];
  if (own !== null && own !== undefined) return own;
  const fallback = values[row.group === "translation" ? "translation_scale" : "rotation_scale"];
  if (fallback === null || fallback === undefined) return null;
  return fallback * (axisCalibration[row.field] ?? 1);
}

function TeleopGainsPanel({
  snapshot,
  busy,
  sessionActive,
  onSetTeleopGains
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  sessionActive: boolean;
  onSetTeleopGains: (gains: TeleopGainValues) => void;
}) {
  const gains = snapshot.teleopGains;
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [dirty, setDirty] = useState(false);

  // While the operator is mid-edit their text is the truth; once applied (or reset) the snapshot is.
  useEffect(() => {
    if (dirty) return;
    setDrafts(Object.fromEntries(GAIN_ROWS.map((row) => [row.field, formatGain(gains.values[row.field])])));
  }, [dirty, gains.values]);

  const parseDraft = (raw: string): { value: number | null; valid: boolean } => {
    const trimmed = raw.trim();
    if (!trimmed) return { value: null, valid: true };
    const parsed = Number(trimmed);
    if (!Number.isFinite(parsed) || Math.abs(parsed) > gains.absMax) return { value: null, valid: false };
    return { value: parsed, valid: true };
  };

  const rowState = GAIN_ROWS.map((row) => {
    const raw = drafts[row.field] ?? "";
    const { value, valid: withinRange } = parseDraft(raw);
    // A global gain is the fallback for three axes, so it cannot be blank and cannot be zero.
    const isGlobal = row.field === "translation_scale" || row.field === "rotation_scale";
    const valid = withinRange && !(isGlobal && (value === null || value <= 0));
    const changed = formatGain(gains.values[row.field]) !== raw.trim();
    return { row, raw, value, valid, changed, isGlobal };
  });

  const allValid = rowState.every((entry) => entry.valid);
  const anyChanged = rowState.some((entry) => entry.changed);
  const overridden = new Set(gains.overridden);

  const apply = () => {
    if (!allValid) return;
    const payload: TeleopGainValues = {};
    for (const entry of rowState) {
      if (entry.value !== null) payload[entry.row.field] = entry.value;
    }
    setDirty(false);
    onSetTeleopGains(payload);
  };

  const reset = () => {
    setDirty(false);
    onSetTeleopGains({});
  };

  return (
    <section className="panel teleop-gains-panel">
      <div className="panel-heading">
        <h2>SpaceMouse 6D Gains</h2>
        <span>{overridden.size ? `${overridden.size} overridden` : "recorder config"}</span>
      </div>
      <p className="teleop-gains-note">
        Applied on the next start — the teleoperator reads its gains once, when it connects. The same
        values drive recording, so an episode is captured at whatever is set here. Leave an axis blank
        to follow its global gain; enter <code>0</code> to disable that axis outright. A blank axis
        is scaled by the device's own per-axis calibration (shown per row) and a filled one is not,
        so typing a global's number into an axis is not the same as leaving it blank.
      </p>
      <div className="teleop-gains-grid">
        {rowState.map(({ row, raw, value, valid, changed, isGlobal }) => {
          const effective = valid ? (isGlobal ? value : effectiveAxisGain(
            { ...gains.values, [row.field]: value },
            row,
            gains.axisCalibration
          )) : null;
          const perSecond = effective === null ? null : Math.abs(effective) * SPACEMOUSE_POLL_HZ;
          const unit = row.group === "translation" ? "m/s" : "rad/s";
          // Only worth saying where it changes the number; scale_x and scale_wx are exactly 1.
          const calibration = gains.axisCalibration[row.field];
          const hint =
            !isGlobal && value === null && calibration !== undefined && calibration !== 1
              ? `${row.hint}, x${calibration.toFixed(2)}`
              : row.hint;
          return (
            <label
              className={`teleop-gain-row${valid ? "" : " teleop-gain-invalid"}${isGlobal ? " teleop-gain-global" : ""}`}
              key={row.field}
            >
              <span className="teleop-gain-label">
                {row.label}
                {overridden.has(row.field) ? <em title="differs from the recorder config">·</em> : null}
              </span>
              <input
                type="number"
                step="0.000001"
                disabled={busy}
                value={raw}
                placeholder={isGlobal ? "required" : "follows global"}
                onChange={(event) => {
                  setDirty(true);
                  setDrafts((current) => ({ ...current, [row.field]: event.target.value }));
                }}
              />
              <small>
                {valid
                  ? perSecond === null
                    ? hint
                    : `${hint} · ${perSecond.toFixed(3)} ${unit} full deflection`
                  : isGlobal
                    ? `positive, at most ${gains.absMax}`
                    : `at most +/-${gains.absMax}`}
              </small>
              {changed && valid ? <b className="teleop-gain-pending">pending</b> : null}
            </label>
          );
        })}
      </div>
      <div className="control-row">
        <button type="button" disabled={busy || !allValid || !anyChanged} onClick={apply}>
          Apply Gains
        </button>
        <button type="button" className="ghost" disabled={busy || (!overridden.size && !anyChanged)} onClick={reset}>
          Reset to Config
        </button>
      </div>
      {sessionActive ? (
        <div className="teleop-gate-note">
          A session is live; these values reach the arm when it is restarted, not now.
        </div>
      ) : null}
      <div className="teleop-gains-sim-note">
        MuJoCo teleop does not read the recorder YAML. Left untouched it runs its own script defaults
        (translation {formatGain(gains.simDefaults.translation_scale)}, rotation{" "}
        {formatGain(gains.simDefaults.rotation_scale)}, all three rotation axes zeroed), so sim and
        hardware do not feel alike until a gain is applied here — an override is sent to both.
      </div>
    </section>
  );
}

export function TeleoperationPage({
  snapshot,
  busy,
  onStartSimTeleop,
  onStartRealTeleop,
  onStopTeleop,
  onSetTeleopGains,
  cameraUrl
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onStartSimTeleop: () => void;
  onStartRealTeleop: () => void;
  onStopTeleop: () => void;
  onSetTeleopGains: (gains: TeleopGainValues) => void;
  cameraUrl: (view: TeleopCameraView, backend: TeleopBackend) => string;
}) {
  const teleop = snapshot.teleop;
  const sessionActive = teleop.state === "running" || teleop.state === "starting";
  const [selectedBackend, setSelectedBackend] = useState<TeleopBackend>(teleop.backend);
  const statusState = sessionActive ? "running" : teleop.state === "error" ? "error" : "idle";
  const cameraViews = teleop.cameraViews?.length ? teleop.cameraViews : defaultCameraViews;
  const workstationDevices = snapshot.devices.filter((device) =>
    ["robot", "gripper", "teleoperator", "camera"].includes(device.kind)
  );
  const detectedDeviceCount = workstationDevices.filter((device) => device.state === "running").length;
  const realCameraActive = selectedBackend === "real";
  const simCameraActive =
    selectedBackend === "mujoco" && sessionActive && teleop.backend === "mujoco";

  useEffect(() => {
    if (sessionActive) setSelectedBackend(teleop.backend);
  }, [sessionActive, teleop.backend]);

  return (
    <div className="page-stack">
      <PageHeader title="FR3 Pika Teleoperation" subtitle="workstation control and observation" />
      <section className="panel teleop-panel">
        <div className="panel-heading">
          <h2>Control Session</h2>
          <span><StatusDot state={statusState} /> {teleop.state}</span>
        </div>
        <div className="mujoco-mode-picker teleop-backend-picker" role="group" aria-label="Teleoperation backend">
          <button
            className={selectedBackend === "mujoco" ? "active" : ""}
            disabled={busy || sessionActive}
            onClick={() => setSelectedBackend("mujoco")}
            type="button"
          >
            MuJoCo
          </button>
          <button
            className={selectedBackend === "real" ? "active" : ""}
            disabled={busy || sessionActive}
            onClick={() => setSelectedBackend("real")}
            type="button"
          >
            Real Robot
          </button>
        </div>
        <div className="summary-grid">
          <Metric label="Backend" value={selectedBackend === "real" ? "FR3 hardware" : "MuJoCo"} />
          <Metric label="Input" value={teleop.inputDevice} />
          <Metric label="Robot" value={teleop.robotModel} />
          <Metric label="Target" value={teleop.targetFrameName} />
        </div>
        <div className="teleop-config-grid">
          <div><span>FR3</span><strong>192.168.1.206</strong></div>
          <div><span>Pika gripper</span><strong>/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0</strong></div>
          <div><span>URDF</span><strong>{teleop.urdfPath}</strong></div>
          <div>
            <span>{selectedBackend === "mujoco" ? "MJCF" : "FCI connection"}</span>
            <strong>
              {selectedBackend === "mujoco"
                ? teleop.simXmlPath
                : teleop.backend === "real" && teleop.realRobotReady
                  ? "connected"
                  : teleop.backend === "real" && teleop.state === "starting"
                    ? "connecting"
                    : "checked on start"}
            </strong>
          </div>
          <div><span>PID</span><strong>{sessionActive ? teleop.pid ?? "-" : "-"}</strong></div>
          <div><span>Cameras</span><strong>D435I external · D405 wrist</strong></div>
        </div>
        <div className="control-row">
          <button
            disabled={busy || sessionActive}
            onClick={selectedBackend === "real" ? onStartRealTeleop : onStartSimTeleop}
          >
            {selectedBackend === "real" ? "Start Real Robot Teleop" : "Start MuJoCo Teleop"}
          </button>
          <button className="danger" disabled={busy || !sessionActive} onClick={onStopTeleop}>Stop Teleop</button>
        </div>
        {selectedBackend === "real" && !sessionActive ? (
          <div className="teleop-gate-note">FCI availability is reported by the control process after launch; it does not gate this action or the camera streams.</div>
        ) : null}
        <div className="teleop-message">{teleop.message}</div>
      </section>

      <TeleopGainsPanel
        snapshot={snapshot}
        busy={busy}
        sessionActive={sessionActive}
        onSetTeleopGains={onSetTeleopGains}
      />

      <section className="panel teleop-observation-panel">
        <div className="panel-heading">
          <h2>{selectedBackend === "real" ? "Real Camera Views" : "Simulation Views"}</h2>
          <span>{cameraViews.length} cameras</span>
        </div>
        <div className="teleop-camera-grid">
          {cameraViews.map((view) => (
            <TeleopCameraTile
              key={view.id}
              view={view}
              active={selectedBackend === "real" ? realCameraActive : simCameraActive}
              backend={selectedBackend}
              cameraUrl={cameraUrl}
            />
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <h2>Workstation I/O</h2>
          <span>{detectedDeviceCount}/{workstationDevices.length} detected</span>
        </div>
        <div className="teleop-device-grid">
          {workstationDevices.map((device) => (
            <div className="teleop-device" key={device.kind + ":" + device.id}>
              <div>
                <strong>{device.label}</strong>
                <span>{device.kind}</span>
              </div>
              <StatusDot state={device.state} />
              <small>{device.detail}</small>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
