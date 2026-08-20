// The live view that sits with the record buttons during a calibration sweep.
//
// Polling model is the Device Manager's: one short snapshot request per tile,
// self-throttled off onLoad/onError so a slow camera never piles up overlapping
// requests, and no MJPEG streams (eleven of those exhaust the browser's
// per-origin connection limit). See stepPreview.ts for why it goes quiet the
// moment an episode opens instead of trying to stay live through it.
import { useEffect, useRef, useState } from "react";
import type { DataCollectionGuiApi } from "../api";

// The recorder publishes ~5 preview frames a second; polling faster only costs
// requests for frames that do not exist yet.
const POLL_MS = 200;
const RETRY_MS = 600;
// Argus cold-spawn can 503 for a few seconds after Connect; only call the
// preview unavailable once it has stayed missing well past that.
const MAX_FAILURES = 25;

function PreviewTile({
  api,
  camera,
  live,
  badge,
}: {
  api: DataCollectionGuiApi;
  camera: string;
  live: boolean;
  badge: string;
}) {
  const [src, setSrc] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  const failuresRef = useRef(0);
  const activeRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (!live) {
      // Stop requesting, but keep the last decoded frame on screen: what the
      // camera is pointed at is still worth seeing while the episode runs.
      activeRef.current = false;
      if (timerRef.current != null) window.clearTimeout(timerRef.current);
      return;
    }
    activeRef.current = true;
    failuresRef.current = 0;
    setUnavailable(false);
    setSrc(api.cameraSnapshotUrl(camera));
    return () => {
      activeRef.current = false;
      if (timerRef.current != null) window.clearTimeout(timerRef.current);
    };
  }, [api, camera, live]);

  const scheduleNext = (delayMs: number) => {
    if (timerRef.current != null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => {
      if (activeRef.current) setSrc(api.cameraSnapshotUrl(camera));
    }, delayMs);
  };

  const handleLoad = () => {
    failuresRef.current = 0;
    setLoaded(true);
    setUnavailable(false);
    scheduleNext(POLL_MS);
  };

  const handleError = () => {
    failuresRef.current += 1;
    if (failuresRef.current >= MAX_FAILURES) setUnavailable(true);
    scheduleNext(RETRY_MS);
  };

  return (
    <div className="camera-tile cali-preview-tile">
      <div className="camera-tile-media">
        {src ? (
          <img
            className="camera-tile-img"
            src={src}
            alt={`${camera} preview`}
            style={{ display: loaded ? "block" : "none", opacity: live ? 1 : 0.55 }}
            onLoad={handleLoad}
            onError={handleError}
          />
        ) : null}
        {!loaded ? (
          <div className="camera-tile-empty">{unavailable ? "暂无画面" : "等待画面…"}</div>
        ) : null}
        {badge ? <span className="cali-preview-badge">{badge}</span> : null}
      </div>
      <div className="camera-tile-label">
        <strong>{camera}</strong>
      </div>
    </div>
  );
}

export function StepCameraPreview({
  api,
  cameras,
  live,
  note,
}: {
  api: DataCollectionGuiApi;
  cameras: string[];
  live: boolean;
  note: string;
}) {
  if (cameras.length === 0) return null;
  return (
    <div className="cali-preview">
      <div className={cameras.length > 1 ? "cali-preview-grid" : "cali-preview-single"}>
        {cameras.map((camera) => (
          <PreviewTile
            key={camera}
            api={api}
            camera={camera}
            live={live}
            badge={live ? "" : "已暂停"}
          />
        ))}
      </div>
      {note ? <p className="panel-note small">{note}</p> : null}
    </div>
  );
}
