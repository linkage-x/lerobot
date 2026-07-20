import { useEffect, useMemo, useRef, useState } from "react";
import type { GuiSnapshot } from "../api";
import type { BoxPreviewPayload, CollectionTask, ConfigSummary, DeviceStatus, EpisodeAnnotation, EventLogItem, ProcessingItem, ProcessingStatus, RecordedDataset, RecordingStatus, ReplayStatus, SubtaskSegment, TaskStatus, DatasetExportStatus, AnnotationOutcome, AnnotationQuality, ReviewStatus } from "../types";
import { StatusDot, Metric, PageHeader, stateLabel, QualityOverview, processingStatusLabel, datasetNamePrefixes, taskDatasetBaseName, processingItemsForTask, taskNeedsQcExportConfirmation } from "../shared/ui";
import { api } from "../apiClient";

export const DEVICE_TOUCH_ROW_LENGTHS = [13, 13, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 13, 13];
export const DEVICE_TOUCH_COLUMNS = 17;

export function interpolatePreviewChannel(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

export function previewTouchColor(value: number, scaleMax: number): string {
  const stops = [
    [17, 24, 39],
    [37, 99, 235],
    [20, 184, 166],
    [250, 204, 21],
    [239, 68, 68]
  ];
  const normalized = Math.max(0, Math.min(1, value / Math.max(scaleMax, 1)));
  const scaled = normalized * (stops.length - 1);
  const index = Math.min(Math.floor(scaled), stops.length - 2);
  const t = scaled - index;
  const a = stops[index];
  const b = stops[index + 1];
  return `rgb(${interpolatePreviewChannel(a[0], b[0], t)}, ${interpolatePreviewChannel(a[1], b[1], t)}, ${interpolatePreviewChannel(a[2], b[2], t)})`;
}

export function numberArray(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => Number(item)).filter((item) => Number.isFinite(item));
}

export function numberValue(value: unknown): number | null {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export function DeviceTouchPreview({ sensor }: { sensor?: Record<string, unknown> | null }) {
  const values = numberArray(sensor?.fz_0p1N);
  const hasData = values.length >= 239;
  const scaleMax = hasData ? Math.max(1, ...values.map((value) => Math.abs(value))) : 1;
  const localMax = hasData ? Math.max(...values.map((value) => Math.abs(value))) : 0;
  const activePoints = values.filter((value) => Math.abs(value) > 0).length;
  let cursor = 0;

  return (
    <div className="touch-map device-touch-map">
      <div className="touch-map-heading">
        <strong>tactile</strong>
        <span>max {localMax.toFixed(1)} · active {activePoints}</span>
      </div>
      {hasData ? (
        <div className="touch-grid" aria-label="live tactile preview">
          {DEVICE_TOUCH_ROW_LENGTHS.map((length, rowIndex) => {
            const offset = Math.floor((DEVICE_TOUCH_COLUMNS - length) / 2);
            const row = values.slice(cursor, cursor + length);
            const startIndex = cursor;
            cursor += length;
            return (
              <div className="touch-row" key={rowIndex}>
                {Array.from({ length: offset }).map((_, index) => (
                  <span className="touch-cell touch-cell-empty" key={`pre-${index}`} />
                ))}
                {row.map((value, index) => {
                  const pointIndex = startIndex + index + 1;
                  return (
                    <span
                      className="touch-cell"
                      key={pointIndex}
                      title={`#${pointIndex} fz=${value.toFixed(1)} (0.1N)`}
                      style={{ backgroundColor: previewTouchColor(Math.abs(value), scaleMax) }}
                    />
                  );
                })}
                {Array.from({ length: DEVICE_TOUCH_COLUMNS - length - offset }).map((_, index) => (
                  <span className="touch-cell touch-cell-empty" key={`post-${index}`} />
                ))}
              </div>
            );
          })}
        </div>
      ) : (
        <div className="touch-empty">no touch sample</div>
      )}
      <div className="touch-map-footer">
        <span>ts {String(sensor?.timestamp ?? "-")}</span>
        <span>fz 0.1N</span>
      </div>
    </div>
  );
}

export function DeviceForcePreview({ sensor }: { sensor?: Record<string, unknown> | null }) {
  const values = numberArray(sensor?.fxyz_mxyz);
  const labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"];
  const maxAbs = Math.max(1, ...values.slice(0, 6).map((value) => Math.abs(value)));
  if (values.length < 6) {
    return <div className="preview-empty">no force sample</div>;
  }
  return (
    <div className="force-preview-grid">
      {labels.map((label, index) => {
        const value = values[index] ?? 0;
        return (
          <div className="force-preview-row" key={label}>
            <span>{label}</span>
            <div className="force-preview-track">
              <i style={{ width: `${Math.min(100, Math.abs(value) / maxAbs * 100)}%` }} />
            </div>
            <strong>{value.toFixed(2)}</strong>
          </div>
        );
      })}
      <small>ts {String(sensor?.timestamp ?? "-")}</small>
    </div>
  );
}

export function RawSensorPreview({ sensor }: { sensor?: Record<string, unknown> | null }) {
  const entries = Object.entries(sensor ?? {}).filter(([, value]) => typeof value !== "object").slice(0, 12);
  if (entries.length === 0) {
    return <div className="preview-empty">no live sample</div>;
  }
  return (
    <div className="device-config-detail device-preview-kv">
      {entries.map(([key, value]) => (
        <div className="device-config-row" key={key}>
          <span className="device-config-key">{key}</span>
          <span className="device-config-value">{String(value)}</span>
        </div>
      ))}
    </div>
  );
}

export function BoxLivePreview({ device }: { device: DeviceStatus }) {
  const [preview, setPreview] = useState<BoxPreviewPayload | null>(null);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      const next = await api.fetchBoxPreview(device.id);
      if (mounted) {
        setPreview(next);
      }
    };
    load();
    const timer = window.setInterval(load, 300);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [device.id]);

  const sensor = preview?.sensor ?? null;
  const staleS = preview?.staleS == null ? null : preview.staleS;
  const isTouch = device.id.startsWith("box_touch") || Array.isArray(sensor?.fz_0p1N);
  const isForce = device.id === "box_six_d_force" || Array.isArray(sensor?.fxyz_mxyz);
  const sampleAge = staleS == null ? "-" : `${staleS.toFixed(1)}s`;
  const queueSize = numberValue(preview?.status?.queue_size);

  return (
    <div className="device-preview-live">
      <div className="device-live-stats">
        <Metric label="Live" value={preview?.active ? "yes" : "no"} />
        <Metric label="Age" value={sampleAge} />
        <Metric label="Queue" value={queueSize ?? "-"} />
      </div>
      {isTouch ? <DeviceTouchPreview sensor={sensor} /> : isForce ? <DeviceForcePreview sensor={sensor} /> : <RawSensorPreview sensor={sensor} />}
    </div>
  );
}

export function DeviceInlinePreview({ device }: { device: DeviceStatus }) {
  // Cameras render their live snapshot through CameraTile in the grid; this
  // inline preview only covers the non-camera expandable rows.
  return (
    <div className="device-inline-preview">
      {device.kind === "box_collection" ? (
        <BoxLivePreview device={device} />
      ) : (
        <div className="preview-empty">no preview stream</div>
      )}
    </div>
  );
}

// BOX device ids are namespaced as `<box_id>/<sensor>` when more than one BOX
// is configured; strip the prefix so we match on the bare sensor name.
export function boxSensorSuffix(deviceId: string): string {
  const slash = deviceId.lastIndexOf("/");
  return slash >= 0 ? deviceId.slice(slash + 1) : deviceId;
}

// The Paxini touch pads and the 6D force sensor carry array payloads that read
// far better as a full-frame visualization (like a camera tile) than as the
// scalar key/value rows the gripper/IMU/trigger use.
export function isVisualBoxSensor(deviceId: string): boolean {
  const sid = boxSensorSuffix(deviceId);
  return sid.startsWith("box_touch") || sid === "box_six_d_force";
}

// Full-frame tactile heatmap: same point layout as DeviceTouchPreview but sized
// to fill the tile media (height-bound, centered) instead of a fixed map.
export function BoxTouchTileView({ sensor }: { sensor?: Record<string, unknown> | null }) {
  const values = numberArray(sensor?.fz_0p1N);
  const hasData = values.length >= 239;
  if (!hasData) {
    return <div className="camera-tile-empty">no touch sample</div>;
  }
  const scaleMax = Math.max(1, ...values.map((value) => Math.abs(value)));
  let cursor = 0;
  return (
    <div className="box-touch-fill" aria-label="live tactile preview">
      {DEVICE_TOUCH_ROW_LENGTHS.map((length, rowIndex) => {
        const offset = Math.floor((DEVICE_TOUCH_COLUMNS - length) / 2);
        const row = values.slice(cursor, cursor + length);
        const startIndex = cursor;
        cursor += length;
        return (
          <div className="touch-row" key={rowIndex}>
            {Array.from({ length: offset }).map((_, index) => (
              <span className="touch-cell touch-cell-empty" key={`pre-${index}`} />
            ))}
            {row.map((value, index) => {
              const pointIndex = startIndex + index + 1;
              return (
                <span
                  className="touch-cell"
                  key={pointIndex}
                  title={`#${pointIndex} fz=${value.toFixed(1)} (0.1N)`}
                  style={{ backgroundColor: previewTouchColor(Math.abs(value), scaleMax) }}
                />
              );
            })}
            {Array.from({ length: DEVICE_TOUCH_COLUMNS - length - offset }).map((_, index) => (
              <span className="touch-cell touch-cell-empty" key={`post-${index}`} />
            ))}
          </div>
        );
      })}
    </div>
  );
}

export const FORCE_TILE_CHANNELS: { label: string; unit: string }[] = [
  { label: "Fx", unit: "N" },
  { label: "Fy", unit: "N" },
  { label: "Fz", unit: "N" },
  { label: "Mx", unit: "N·m" },
  { label: "My", unit: "N·m" },
  { label: "Mz", unit: "N·m" },
];

// Full-frame 6D force/torque: bipolar bars centered on zero (force can push or
// pull on every axis), forces and moments scaled independently so neither
// dwarfs the other.
export function BoxForceTileView({ sensor }: { sensor?: Record<string, unknown> | null }) {
  const values = numberArray(sensor?.fxyz_mxyz);
  if (values.length < 6) {
    return <div className="camera-tile-empty">no force sample</div>;
  }
  const forceMax = Math.max(1, ...values.slice(0, 3).map((value) => Math.abs(value)));
  const momentMax = Math.max(0.1, ...values.slice(3, 6).map((value) => Math.abs(value)));
  return (
    <div className="box-force-fill" aria-label="live 6D force preview">
      {FORCE_TILE_CHANNELS.map((channel, index) => {
        const value = values[index] ?? 0;
        const max = index < 3 ? forceMax : momentMax;
        const ratio = Math.max(-1, Math.min(1, value / max));
        const pct = Math.abs(ratio) * 50;
        const positive = ratio >= 0;
        return (
          <div className="force-bipolar-row" key={channel.label}>
            <span className="force-bipolar-label">{channel.label}</span>
            <div className="force-bipolar-track">
              <i
                className={positive ? "pos" : "neg"}
                style={positive ? { left: "50%", width: `${pct}%` } : { right: "50%", width: `${pct}%` }}
              />
            </div>
            <strong className="force-bipolar-value">
              {value.toFixed(2)}
              <small>{channel.unit}</small>
            </strong>
          </div>
        );
      })}
    </div>
  );
}

// Camera-tile lookalike for the array BOX sensors: the live visualization fills
// the media, and the scalar config / liveness stats live in the hover overlay,
// mirroring CameraTile so the Device Manager grid stays visually uniform.

export function BoxSensorTile({ device }: { device: DeviceStatus }) {
  const [preview, setPreview] = useState<BoxPreviewPayload | null>(null);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      const next = await api.fetchBoxPreview(device.id);
      if (mounted) {
        setPreview(next);
      }
    };
    load();
    const timer = window.setInterval(load, 300);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [device.id]);

  const sensor = preview?.sensor ?? null;
  const isForce = boxSensorSuffix(device.id) === "box_six_d_force" || Array.isArray(sensor?.fxyz_mxyz);
  const staleS = preview?.staleS == null ? null : preview.staleS;
  const sampleAge = staleS == null ? "-" : `${staleS.toFixed(1)}s`;
  const queueSize = numberValue(preview?.status?.queue_size);
  const config = device.config ?? {};
  const configEntries = Object.entries(config).filter(
    ([, v]) => v != null && typeof v !== "object"
  );

  return (
    <div className="camera-tile">
      <div className="camera-tile-media box-sensor-media">
        {isForce ? <BoxForceTileView sensor={sensor} /> : <BoxTouchTileView sensor={sensor} />}
        <div className="camera-tile-overlay">
          <div className="device-live-stats">
            <Metric label="Live" value={preview?.active ? "yes" : "no"} />
            <Metric label="Age" value={sampleAge} />
            <Metric label="Queue" value={queueSize ?? "-"} />
          </div>
          <div className="device-config-grid">
            <div className="device-config-row">
              <span className="device-config-key">label</span>
              <span className="device-config-value">{device.label}</span>
            </div>
            <div className="device-config-row">
              <span className="device-config-key">timestamp</span>
              <span className="device-config-value">{String(sensor?.timestamp ?? "-")}</span>
            </div>
            {device.detail && (
              <div className="device-config-row">
                <span className="device-config-key">detail</span>
                <span className="device-config-value">{device.detail}</span>
              </div>
            )}
            {configEntries.map(([key, value]) => (
              <div className="device-config-row" key={key}>
                <span className="device-config-key">{key}</span>
                <span className="device-config-value">{String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="camera-tile-label">
        <StatusDot state={device.state} />
        <strong>{device.id}</strong>
        <span className="camera-tile-stat">{device.fps} Hz</span>
      </div>
    </div>
  );
}

// Cold-spawn (Argus open + AWB settle) can 503 for a few seconds before the
// first frame; keep retrying with backoff this many times (~15s) before
// surfacing "preview unavailable", then keep probing in case it recovers.
export const PREVIEW_MAX_FAILURES = 30;

export function CameraTile({ device, snapshot }: { device: DeviceStatus; snapshot: GuiSnapshot }) {
  // Idle snapshots use the gateway's temporary preview pipeline; while the
  // recorder owns the cameras, snapshots come from recorder-owned tmpfs JPEGs.
  const previewable = device.state !== "error";
  const config = device.config ?? {};
  const configEntries = Object.entries(config).filter(
    ([, v]) => v != null && typeof v !== "object"
  );

  // Snapshot polling: each request is short, so 11 tiles don't exhaust the
  // browser's ~6-connections-per-origin limit the way 11 live MJPEG streams
  // would. The loop is self-throttled off onLoad/onError, so a slow camera
  // never piles up overlapping requests.
  const [src, setSrc] = useState("");
  const [loaded, setLoaded] = useState(false);
  const failuresRef = useRef(0);
  const activeRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  const refresh = () => {
    if (activeRef.current) setSrc(`${api.cameraSnapshotUrl(device.id)}&t=${Date.now()}`);
  };
  const scheduleNext = (delayMs: number) => {
    if (timerRef.current != null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(refresh, delayMs);
  };

  useEffect(() => {
    if (!previewable) {
      activeRef.current = false;
      if (timerRef.current != null) window.clearTimeout(timerRef.current);
      setSrc("");
      setLoaded(false);
      failuresRef.current = 0;
      return;
    }
    activeRef.current = true;
    failuresRef.current = 0;
    setLoaded(false);
    refresh();
    return () => {
      activeRef.current = false;
      if (timerRef.current != null) window.clearTimeout(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewable, device.id]);

  const handleLoad = () => {
    failuresRef.current = 0;
    setLoaded(true);
    scheduleNext(200); // ~5 fps poll
  };
  const handleError = () => {
    failuresRef.current += 1;
    setLoaded(false);
    scheduleNext(failuresRef.current >= PREVIEW_MAX_FAILURES ? 2000 : 500);
  };

  const placeholder = device.state === "error"
    ? "no signal"
    : failuresRef.current >= PREVIEW_MAX_FAILURES
      ? "preview unavailable"
      : device.state;

  return (
    <div className="camera-tile">
      <div className="camera-tile-media">
        {previewable && src ? (
          <img
            className="camera-tile-img"
            src={src}
            alt={`${device.id} preview`}
            style={{ display: loaded ? "block" : "none" }}
            onLoad={handleLoad}
            onError={handleError}
          />
        ) : null}
        {(!previewable || !loaded) && (
          <div className="camera-tile-empty">{placeholder}</div>
        )}
        <div className="camera-tile-overlay">
          <div className="device-config-grid">
            <div className="device-config-row">
              <span className="device-config-key">fps</span>
              <span className="device-config-value">{device.fps}</span>
            </div>
            <div className="device-config-row">
              <span className="device-config-key">latency</span>
              <span className="device-config-value">{device.latencyMs} ms</span>
            </div>
            {device.detail && (
              <div className="device-config-row">
                <span className="device-config-key">detail</span>
                <span className="device-config-value">{device.detail}</span>
              </div>
            )}
            {configEntries.map(([key, value]) => (
              <div className="device-config-row" key={key}>
                <span className="device-config-key">{key}</span>
                <span className="device-config-value">{String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="camera-tile-label">
        <StatusDot state={device.state} />
        <strong>{device.id}</strong>
        <span className="camera-tile-stat">{device.fps} fps</span>
      </div>
    </div>
  );
}

// Expandable scalar-sensor row used for every non-camera device (and the BOX
// gripper/IMU/trigger). Factored out so the box_collection section can mix tile
// and row rendering without duplicating the markup.
export function DeviceRow({
  device,
  isExpanded,
  onToggle,
}: {
  device: DeviceStatus;
  isExpanded: boolean;
  onToggle: (id: string) => void;
}) {
  const config = device.config ?? {};
  const configEntries = Object.entries(config).filter(
    ([, v]) => v != null && typeof v !== "object"
  );
  return (
    <div className={`device-manager-row ${isExpanded ? "device-manager-row-active" : ""}`}>
      <button className="device-manager-header" onClick={() => onToggle(device.id)}>
        <div className="row-title">
          <StatusDot state={device.state} />
          <strong>{device.id}</strong>
        </div>
        <div className="device-stats">
          <span>{device.fps} fps</span>
          <span>{device.latencyMs} ms</span>
          <small>{device.detail}</small>
          <small>{isExpanded ? "close" : "open"}</small>
        </div>
      </button>
      {isExpanded && (
        <div className="device-config-detail device-config-detail-expanded">
          {configEntries.length > 0 && (
            <div className="device-config-grid">
              {configEntries.map(([key, value]) => (
                <div className="device-config-row" key={key}>
                  <span className="device-config-key">{key}</span>
                  <span className="device-config-value">{String(value)}</span>
                </div>
              ))}
            </div>
          )}
          <DeviceInlinePreview device={device} />
        </div>
      )}
    </div>
  );
}

export function DeviceManagerPage({ snapshot }: { snapshot: GuiSnapshot }) {
  const [hideErrors, setHideErrors] = useState(false);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  const devices = hideErrors
    ? snapshot.devices.filter((d) => d.state !== "error")
    : snapshot.devices;

  const toggleExpanded = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const grouped = useMemo(() => {
    return devices.reduce<Record<string, typeof devices>>((acc, d) => {
      acc[d.kind] = [...(acc[d.kind] ?? []), d];
      return acc;
    }, {});
  }, [devices]);

  const onlineCount = snapshot.devices.filter((d) => d.state === "running").length;
  const errorCount = snapshot.devices.filter((d) => d.state === "error").length;

  const kindLabel = (kind: string): string => {
    if (kind === "camera") return snapshot.configSummary.rigType === "gmsl2" ? "GMSL2 Cameras" : "Cameras";
    if (kind === "box_collection") return "BOX Sensors";
    if (kind === "tactile") return "Tactile Sensors";
    if (kind === "handheld_gripper") return "Handheld Grippers";
    return kind.replace("_", " ");
  };

  return (
    <div className="page-stack">
      <PageHeader title="Device Manager" subtitle="view connected hardware devices, configuration details, and connection status" />
      <section className="panel">
        <div className="panel-heading">
          <h2>Summary</h2>
          <span>{snapshot.devices.length} devices</span>
        </div>
        <div className="summary-grid">
          <Metric label="Total" value={snapshot.devices.length} />
          <Metric label="Online" value={onlineCount} />
          <Metric label="Error" value={errorCount} />
        </div>
        <div className="control-row">
          <label className="annotation-field annotation-toggle">
            <input type="checkbox" checked={hideErrors} onChange={(e) => setHideErrors(e.target.checked)} />
            <span>Hide error devices</span>
          </label>
        </div>
      </section>
      {Object.entries(grouped).map(([kind, items]) => (
        <section className="panel" key={kind}>
          <div className="panel-heading">
            <h2>{kindLabel(kind)}</h2>
            <span>{items.filter((d) => d.state === "running").length}/{items.length} online</span>
          </div>
          {kind === "camera" ? (
            <div className="camera-grid">
              {items.map((device) => (
                <CameraTile key={device.id} device={device} snapshot={snapshot} />
              ))}
            </div>
          ) : kind === "box_collection" ? (
            <>
              {items.some((d) => isVisualBoxSensor(d.id)) && (
                <div className="camera-grid box-sensor-grid">
                  {items
                    .filter((d) => isVisualBoxSensor(d.id))
                    .map((device) => (
                      <BoxSensorTile key={device.id} device={device} />
                    ))}
                </div>
              )}
              {items
                .filter((d) => !isVisualBoxSensor(d.id))
                .map((device) => (
                  <DeviceRow
                    key={device.id}
                    device={device}
                    isExpanded={expandedIds.has(device.id)}
                    onToggle={toggleExpanded}
                  />
                ))}
            </>
          ) : (
            items.map((device) => (
              <DeviceRow
                key={device.id}
                device={device}
                isExpanded={expandedIds.has(device.id)}
                onToggle={toggleExpanded}
              />
            ))
          )}
        </section>
      ))}
    </div>
  );
}

