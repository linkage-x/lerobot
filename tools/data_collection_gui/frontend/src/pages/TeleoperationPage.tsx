import { useEffect, useRef, useState } from "react";

import type { GuiSnapshot } from "../api";
import { Metric, PageHeader, StatusDot } from "../shared/ui";
import type { TeleopCameraView } from "../types";

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
