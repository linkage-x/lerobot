import { useEffect, useRef, useState } from "react";

import type { GuiSnapshot } from "../api";
import { Metric, PageHeader, StatusDot } from "../shared/ui";
import type { TeleopCameraView } from "../types";

const defaultCameraViews: TeleopCameraView[] = [
  { id: "external", label: "External", source: "D435I", fps: 30 },
  { id: "wrist", label: "Wrist", source: "D405", fps: 30 }
];

function TeleopCameraTile({
  view,
  running,
  cameraUrl
}: {
  view: TeleopCameraView;
  running: boolean;
  cameraUrl: (viewId: string) => string;
}) {
  const [src, setSrc] = useState("");
  const [frameReady, setFrameReady] = useState(false);
  const timerRef = useRef<number | null>(null);
  const cameraUrlRef = useRef(cameraUrl);
  cameraUrlRef.current = cameraUrl;

  const scheduleNext = (delayMs: number) => {
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => setSrc(cameraUrlRef.current(view.id)), delayMs);
  };

  useEffect(() => {
    if (running) {
      setSrc(cameraUrlRef.current(view.id));
    } else {
      setSrc("");
      setFrameReady(false);
    }
    return () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    };
  }, [running, view.id]);

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
          <StatusDot state={running && frameReady ? "running" : "idle"} />
          {running && frameReady ? "LIVE" : "OFFLINE"}
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
  onStopTeleop,
  cameraUrl
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onStartSimTeleop: () => void;
  onStopTeleop: () => void;
  cameraUrl: (viewId: string) => string;
}) {
  const teleop = snapshot.teleop;
  const teleopRunning = teleop.state === "running";
  const statusState = teleopRunning ? "running" : teleop.state === "error" ? "error" : "idle";
  const cameraViews = teleop.cameraViews?.length ? teleop.cameraViews : defaultCameraViews;
  const workstationDevices = snapshot.devices.filter((device) =>
    ["robot", "gripper", "teleoperator", "camera"].includes(device.kind)
  );
  const detectedDeviceCount = workstationDevices.filter((device) => device.state === "running").length;

  return (
    <div className="page-stack">
      <PageHeader title="FR3 Pika Teleoperation" subtitle="workstation control and observation" />
      <section className="panel teleop-panel">
        <div className="panel-heading">
          <h2>Control Session</h2>
          <span><StatusDot state={statusState} /> {teleop.state}</span>
        </div>
        <div className="summary-grid">
          <Metric label="Backend" value={teleop.backend} />
          <Metric label="Input" value={teleop.inputDevice} />
          <Metric label="Robot" value={teleop.robotModel} />
          <Metric label="Target" value={teleop.targetFrameName} />
        </div>
        <div className="teleop-config-grid">
          <div><span>URDF</span><strong>{teleop.urdfPath}</strong></div>
          <div><span>MJCF</span><strong>{teleop.simXmlPath}</strong></div>
          <div><span>PID</span><strong>{teleop.pid ?? "-"}</strong></div>
          <div><span>Real Robot</span><strong>{teleop.realRobotReady ? "ready" : "reserved"}</strong></div>
        </div>
        <div className="control-row">
          <button disabled={busy || teleopRunning} onClick={onStartSimTeleop}>Start MuJoCo Teleop</button>
          <button className="danger" disabled={busy || !teleopRunning} onClick={onStopTeleop}>Stop Teleop</button>
        </div>
        <div className="teleop-message">{teleop.message}</div>
      </section>

      <section className="panel teleop-observation-panel">
        <div className="panel-heading">
          <h2>Simulation Views</h2>
          <span>{cameraViews.length} cameras</span>
        </div>
        <div className="teleop-camera-grid">
          {cameraViews.map((view) => (
            <TeleopCameraTile
              key={view.id}
              view={view}
              running={teleopRunning}
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
            <div className="teleop-device" key={`${device.kind}:${device.id}`}>
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
