import { useEffect, useRef, useState } from "react";

import type { GuiSnapshot } from "../api";
import { Metric, PageHeader, StatusDot } from "../shared/ui";
import type { TeleopCameraView } from "../types";
import { api } from "../apiClient";
import { ForceSensorCard, TactileSensorCard } from "../calibration/SensorMonitors";

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

export function TeleoperationPage({
  snapshot,
  busy,
  onStartSimTeleop,
  onStartRealTeleop,
  onStopTeleop,
  cameraUrl
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onStartSimTeleop: () => void;
  onStartRealTeleop: () => void;
  onStopTeleop: () => void;
  cameraUrl: (view: TeleopCameraView, backend: TeleopBackend) => string;
}) {
  const teleop = snapshot.teleop;
  const thorFr3 = snapshot.deployment?.capabilities.includes("fr3_bridge") ?? false;
  const telemetry = teleop.realRobotReady ? teleop.telemetry : undefined;
  const sessionActive = teleop.state === "running" || teleop.state === "starting";
  const [selectedBackend, setSelectedBackend] = useState<TeleopBackend>(teleop.backend);
  const statusState = sessionActive ? "running" : teleop.state === "error" ? "error" : "idle";
  const cameraViews = thorFr3
    ? snapshot.devices.filter((d) => d.kind === "camera").map((d) => ({
        id: d.id, deviceId: d.id, label: d.label, source: "Sengyun GMSL2", fps: d.fps || 60
      }))
    : teleop.cameraViews?.length ? teleop.cameraViews : defaultCameraViews;
  const workstationDevices = snapshot.devices.filter((device) =>
    ["robot", "gripper", "teleoperator", "camera"].includes(device.kind)
  );
  const detectedDeviceCount = workstationDevices.filter((device) => device.state === "running").length;
  const realCameraActive = selectedBackend === "real" && (!thorFr3 || snapshot.recording.state !== "idle");
  const simCameraActive =
    selectedBackend === "mujoco" && sessionActive && teleop.backend === "mujoco";

  useEffect(() => {
    if (sessionActive || thorFr3) setSelectedBackend(thorFr3 ? "real" : teleop.backend);
  }, [sessionActive, teleop.backend, thorFr3]);

  return (
    <div className="page-stack">
      <PageHeader title={thorFr3 ? "FR3 + Thor Teleoperation" : "FR3 Pika Teleoperation"}
        subtitle={thorFr3 ? "SpaceMouse · Sengyun cameras · BOX tactile and force" : "workstation control and observation"} />
      <section className="panel teleop-panel">
        <div className="panel-heading">
          <h2>Control Session</h2>
          <span><StatusDot state={statusState} /> {teleop.state}</span>
        </div>
        <div className="mujoco-mode-picker teleop-backend-picker" role="group" aria-label="Teleoperation backend">
          <button
            className={selectedBackend === "mujoco" ? "active" : ""}
            disabled={busy || sessionActive || thorFr3}
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
          <div><span>Cameras</span><strong>{thorFr3 ? "Sengyun GMSL2" : "D435I external · D405 wrist"}</strong></div>
        </div>
        <div className="control-row">
          <button
            disabled={busy || sessionActive || (thorFr3 && !teleop.realRobotReady)}
            onClick={selectedBackend === "real" ? onStartRealTeleop : onStartSimTeleop}
          >
            {selectedBackend === "real" ? "Start Real Robot Teleop" : "Start MuJoCo Teleop"}
          </button>
          <button className="danger" disabled={busy || !sessionActive} onClick={onStopTeleop}>Stop Teleop</button>
        </div>
        {thorFr3 ? (
          <div className="teleop-gate-note">Connect devices in Live Record first. Start enables SpaceMouse and gripper control;
            Stop keeps telemetry connected. Use Live Record to save or discard episodes. Input: {teleop.inputSource || "thor"}.</div>
        ) : selectedBackend === "real" && !sessionActive ? (
          <div className="teleop-gate-note">FCI availability is reported by the control process after launch; it does not gate this action or the camera streams.</div>
        ) : null}
        <div className="teleop-message">{teleop.message}</div>
      </section>

      {thorFr3 ? <section className="panel">
        <div className="panel-heading"><h2>Measured FR3 State</h2><span>{teleop.realRobotReady ? "Live" : "Disconnected"}</span></div>
        <div className="summary-grid">
          <Metric label="FCI command success" value={telemetry?.control_command_success_rate != null
            ? `${(telemetry.control_command_success_rate * 100).toFixed(2)}%` : "—"} />
          <Metric label="Bridge round trip" value={telemetry?.round_trip_ms != null ? `${telemetry.round_trip_ms.toFixed(2)} ms` : "—"} />
          <Metric label="Clock uncertainty" value={telemetry?.clock_uncertainty_s != null
            ? `±${(telemetry.clock_uncertainty_s * 1000).toFixed(2)} ms` : "—"} />
          <Metric label="Measured gripper opening" value={telemetry?.gripper_measured_m != null
            ? `${(telemetry.gripper_measured_m * 1000).toFixed(1)} mm` : "—"} />
        </div>
        <table><thead><tr><th>Joint</th><th>Position (rad)</th><th>Velocity (rad/s)</th><th>Torque (N·m)</th><th>External torque (N·m)</th></tr></thead>
          <tbody>{Array.from({ length: 7 }, (_, i) => <tr key={i}><td>{i + 1}</td>
            {[telemetry?.q, telemetry?.dq, telemetry?.tau_J, telemetry?.tau_ext_hat_filtered].map((v, j) =>
              <td key={j}>{v?.[i]?.toFixed(4) ?? "—"}</td>)}</tr>)}</tbody></table>
        <p>Measured task TCP [x, y, z (m), rotation vector (rad)]: {telemetry?.measured_tcp?.map((v) => v.toFixed(4)).join(", ") || "—"}</p>
        <p>FR3 estimated external wrench [Fx, Fy, Fz (N), Mx, My, Mz (N·m)]: {telemetry?.O_F_ext_hat_K?.map((v) => v.toFixed(3)).join(", ") || "—"}</p>
      </section> : null}

      {thorFr3 ? <section className="panel">
        <div className="panel-heading"><h2>Gripper Tactile and Force</h2></div>
        <div className="teleop-device-grid">{snapshot.devices.filter((d) => d.id.endsWith("box_six_d_force") || d.id.includes("box_touch_")).map((device) =>
          device.id.endsWith("box_six_d_force") ? <ForceSensorCard key={device.id} api={api} device={device} />
            : <TactileSensorCard key={device.id} api={api} device={device} />)}</div>
      </section> : null}

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
          <h2>{thorFr3 ? "Thor / FR3 I/O" : "Workstation I/O"}</h2>
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
