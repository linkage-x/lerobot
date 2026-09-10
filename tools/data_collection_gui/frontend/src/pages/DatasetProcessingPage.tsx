import { useCallback, useEffect, useMemo, useState } from "react";
import type { GuiSnapshot } from "../api";
import type { BoxPreviewPayload, BoxCaliLog, BoxCaliLogLine, CollectionTask, ConfigSummary, DeviceStatus, EpisodeAnnotation, EventLogItem, ProcessingItem, ProcessingStatus, RecordedDataset, RecordingStatus, ReplayStatus, SubtaskSegment, TaskStatus, DatasetExportStatus, AnnotationOutcome, AnnotationQuality, ReviewStatus, TrackingTarget } from "../types";
import { StatusDot, Metric, PageHeader, stateLabel, QualityOverview, processingStatusLabel, datasetNamePrefixes, taskDatasetBaseName, processingItemsForTask, taskNeedsQcExportConfirmation } from "../shared/ui";

export const processingStatusDot: Record<ProcessingStatus, string> = {
  pose_missing: "warning",
  queued: "connecting",
  running: "running",
  pose_ready: "armed",
  qc_pass: "running",
  qc_failed: "error",
  error: "error"
};

export function ProcessingRow({
  item,
  active,
  busy,
  onSelect,
  onGenerate,
  onRunQc,
  onOpenReplay
}: {
  item: ProcessingItem;
  active: boolean;
  busy: boolean;
  onSelect: () => void;
  onGenerate: () => void;
  onRunQc: () => void;
  onOpenReplay: () => void;
}) {
  const primary =
    item.status === "pose_missing" ? (
      <button disabled={busy} onClick={onGenerate}>Generate EE Trajectory</button>
    ) : item.status === "queued" || item.status === "running" ? (
      <button disabled={busy} onClick={onSelect}>View Log</button>
    ) : item.status === "qc_pass" ? (
      <button disabled={busy} onClick={onOpenReplay}>Open Replay</button>
    ) : item.status === "error" && !item.trajectoryVersion ? (
      // trajectory generation failed: no trajectory to QC, offer to regenerate instead
      <button disabled={busy} onClick={onGenerate}>Regenerate Trajectory</button>
    ) : item.status === "qc_failed" || item.status === "error" ? (
      <button disabled={busy} onClick={onRunQc}>Re-run QC</button>
    ) : (
      <button disabled={busy} onClick={onRunQc}>Run QC</button>
    );

  return (
    <div className={active ? "processing-row active" : "processing-row"}>
      <button className="processing-row-main" onClick={onSelect} disabled={busy}>
        <div>
          <div className="row-title">
            <StatusDot state={processingStatusDot[item.status]} />
            <strong>{item.name}</strong>
            {item.trajectoryVersion ? <em>{item.trajectoryVersion}</em> : null}
          </div>
          <p>{item.message}</p>
        </div>
        <div className="processing-stats">
          <span>{processingStatusLabel[item.status]}</span>
          <small>{item.updatedAt}</small>
        </div>
      </button>
      <div className="processing-actions">{primary}</div>
    </div>
  );
}


export function OnlineSyncManifestBlock({ item }: { item: ProcessingItem }) {
  const summary = item.onlineSync;
  if (!summary) {
    return null;
  }
  const maxDelta = summary.maxSofDeltaMs;
  const statusLabel = summary.status === "pass" ? "pass" : summary.status === "missing" ? "missing" : "fail";
  const shownEpisodes = summary.episodes.slice(0, 6);
  return (
    <div className="qc-block online-sync-block">
      <div className="qc-block-heading">
        <h3>Online Sync Manifest</h3>
        <span className={`sync-status sync-status-${statusLabel}`}>{statusLabel}</span>
      </div>
      <div className="summary-grid compact-summary-grid">
        <Metric label="Episodes" value={`${summary.ok}/${summary.totalEpisodes}`} />
        <Metric label="Actual frames" value={summary.actualFrames} />
        <Metric label="Max SOF delta" value={maxDelta != null ? `${maxDelta.toFixed(3)} ms` : "—"} />
        <Metric label="Missing" value={summary.missing} />
      </div>
      <div className="online-sync-episodes">
        {shownEpisodes.map((episode) => {
          const countText = Object.entries(episode.frameCountByCamera)
            .slice(0, 4)
            .map(([camera, count]) => `${camera}:${count}`)
            .join(" ");
          return (
            <div className="online-sync-episode" key={episode.episode}>
              <strong>ep {episode.episode}</strong>
              <span>{episode.actualFrames != null ? `${episode.actualFrames} frames` : "no manifest"}</span>
              <span>{episode.maxSofDeltaMs != null ? `${episode.maxSofDeltaMs.toFixed(3)} ms` : "—"}</span>
              <small>{episode.ok ? countText || "counts ok" : episode.failure || "failed"}</small>
            </div>
          );
        })}
      </div>
      {summary.failureReasons.length ? <p className="panel-note">{summary.failureReasons[0]}</p> : null}
    </div>
  );
}

export function TrackingDetectionBlock({
  item,
  busy,
  onOpenReplay
}: {
  item: ProcessingItem;
  busy: boolean;
  onOpenReplay: () => void;
}) {
  const summary = item.detectionSummary;
  if (!summary) {
    return null;
  }
  const isCarrier = summary.target === "hybrid_carrier_v1";
  return (
    <div className="qc-block tracking-evidence-block">
      <div className="qc-block-heading">
        <div>
          <h3>Marker detection evidence</h3>
          <p>{summary.label} · camera views, before multi-camera fusion</p>
        </div>
        <button disabled={busy || !summary.overlayAvailable} onClick={onOpenReplay}>
          Inspect overlays
        </button>
      </div>
      <div className="summary-grid compact-summary-grid">
        <Metric label="Target" value={summary.label} />
        <Metric label="Detected views" value={`${summary.detectedViews}/${summary.totalViews}`} />
        <Metric label="Detection rate" value={`${summary.detectionRatePct.toFixed(1)}%`} />
        <Metric label="Replay evidence" value={summary.overlayAvailable ? "available" : "none"} />
      </div>
      <div className="tracking-camera-table">
        <div className="tracking-camera-row tracking-camera-header">
          <span>Camera</span>
          <span>Detected</span>
          <span>Anchors</span>
          <span>Residual</span>
          {isCarrier ? <span>Facet edges</span> : null}
        </div>
        {summary.perCamera.map((camera) => (
          <div className="tracking-camera-row" key={`${camera.camera}-${camera.streamKey}`}>
            <strong>{camera.camera}</strong>
            <span>{camera.detectedViews}/{camera.totalViews} · {camera.detectionRatePct.toFixed(1)}%</span>
            <span>{camera.medianAnchors == null ? "—" : camera.medianAnchors.toFixed(1)}</span>
            <span>{camera.medianRmsePx == null ? "—" : `${camera.medianRmsePx.toFixed(2)} px`}</span>
            {isCarrier ? (
              <span>{camera.medianEdgeSamples == null ? "—" : camera.medianEdgeSamples.toFixed(0)}</span>
            ) : null}
          </div>
        ))}
      </div>
      <p className="panel-note">
        {isCarrier
          ? "Cyan outlines are observed ArUco corners; coloured outlines are visible CAD facets projected from the solved pose."
          : "Cube wireframes are projected from each camera's solved pose."}
      </p>
    </div>
  );
}

export function DatasetProcessingPage({
  snapshot,
  busy,
  onGenerate,
  onRunQc,
  onOpenReplay,
  onSetDatasetsRoot
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onGenerate: (path: string, markerTcpCalibrationPath?: string, trackingTarget?: TrackingTarget) => void;
  onRunQc: (path: string) => void;
  onOpenReplay: (path: string) => void;
  onSetDatasetsRoot: (path: string) => void;
}) {
  const items = snapshot.processing;
  const [selectedPath, setSelectedPath] = useState<string>(items[0]?.path ?? "");
  const selected = items.find((item) => item.path === selectedPath) ?? items[0];
  const [trackingTarget, setTrackingTarget] = useState<TrackingTarget>(
    items[0]?.trackingTarget ?? "april_cube"
  );
  const runningCount = items.filter((item) => item.status === "running" || item.status === "queued").length;
  const failedCount = items.filter((item) => item.status === "qc_failed" || item.status === "error").length;
  const readyCount = items.filter((item) => item.status === "qc_pass").length;
  const currentRoot = snapshot.gateway.datasetsRoot ?? "";
  const [rootInput, setRootInput] = useState<string>(currentRoot);
  const latestMarkerTcpPath = snapshot.markerTcp?.solvePath ?? "";
  // Deliberately NOT pre-filled with the latest solve. A solve directory's bundle is
  // merged from whatever bundle it started from, so it is only fresh for the ONE cube
  // that was just solved -- every other cube in it is whatever the previous bundle held.
  // Pre-filling made "just press Generate" silently pick that over the production bundle
  // named in the tracker YAML, which is the one that gets curated for all cubes.
  // "Latest Solve" is still one click away when that is genuinely what you want.
  useEffect(() => {
    if (selected?.trackingTarget) {
      setTrackingTarget(selected.trackingTarget);
    }
  }, [selected?.path, selected?.trackingTarget]);
  const [markerTcpPathInput, setMarkerTcpPathInput] = useState<string>("");
  useEffect(() => {
    setRootInput(currentRoot);
  }, [currentRoot]);
  const rootDirty = rootInput.trim() !== currentRoot;
  const markerTcpPath = markerTcpPathInput.trim();
  const markerTcpForRun = trackingTarget === "april_cube" ? markerTcpPath : "";

  return (
    <div className="page-stack">
      <PageHeader title="Dataset Processing" subtitle="EE trajectory generation, QC, and post-processing job queue for recorded datasets" />
      <section className="panel">
        <div className="panel-heading">
          <h2>Datasets Root</h2>
          <span>scan path</span>
        </div>
        <div className="datasets-root-row">
          <input
            className="datasets-root-input"
            value={rootInput}
            onChange={(event) => setRootInput(event.target.value)}
            placeholder="outputs/datasets"
            spellCheck={false}
          />
          <button
            disabled={busy || !rootDirty || !rootInput.trim()}
            onClick={() => onSetDatasetsRoot(rootInput.trim())}
          >
            Save
          </button>
          <button
            disabled={busy || rootInput.trim() === currentRoot}
            onClick={() => setRootInput(currentRoot)}
          >
            Reset
          </button>
        </div>
        <p className="panel-note">Absolute or relative to the gateway repo root. Default: outputs/datasets.</p>
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>Detection Target</h2>
          <span>explicit physical model</span>
        </div>
        <div className="tracking-target-grid" role="radiogroup" aria-label="Detection target">
          <button
            className={trackingTarget === "april_cube" ? "tracking-target-card active" : "tracking-target-card"}
            aria-pressed={trackingTarget === "april_cube"}
            disabled={busy}
            onClick={() => setTrackingTarget("april_cube")}
          >
            <strong>AprilTag Cube</strong>
            <span>Legacy 6-face cube tracker and calibrated marker→TCP bundle.</span>
          </button>
          <button
            className={trackingTarget === "hybrid_carrier_v1" ? "tracking-target-card active" : "tracking-target-card"}
            aria-pressed={trackingTarget === "hybrid_carrier_v1"}
            disabled={busy}
            onClick={() => setTrackingTarget("hybrid_carrier_v1")}
          >
            <strong>0907 Hybrid Carrier V1</strong>
            <span>ArUco 30/31/32 anchors plus red/green/blue CAD-facet refinement.</span>
          </button>
        </div>
        <p className="panel-note">
          目标必须显式选择：错误的字典和几何模型可能得到“无检测”，不能靠文件名猜测物理夹具。
        </p>
        {trackingTarget === "april_cube" ? (
          <>
            <div className="datasets-root-row trajectory-option-row">
              <input
                className="datasets-root-input"
                value={markerTcpPathInput}
                onChange={(event) => setMarkerTcpPathInput(event.target.value)}
                placeholder="可选：outputs/.../marker_to_tcp_calibration.json"
                spellCheck={false}
              />
              <button
                disabled={busy || !latestMarkerTcpPath}
                onClick={() => setMarkerTcpPathInput(latestMarkerTcpPath)}
              >
                Latest Solve
              </button>
              <button
                disabled={busy || !markerTcpPathInput.trim()}
                onClick={() => setMarkerTcpPathInput("")}
              >
                Clear
              </button>
            </div>
            <p className="panel-note">
              {markerTcpPath
                ? `Generate 将使用 ${markerTcpPath}（覆盖 tracker YAML 的默认 bundle）`
                : "留空 = 使用 tracker YAML 里的 production marker→TCP bundle，正常情况保持留空。"}
            </p>
          </>
        ) : (
          <div className="carrier-contract-note">
            <strong>TCP comes from geometry</strong>
            <span>
              0907 carrier 的球窝中心就是 CAD 原点；方向约定由版本化 carrier bundle 固定，不接受 cube 标定覆盖。
            </span>
          </div>
        )}
      </section>
      <section className="panel">
        <div className="panel-heading">
          <h2>Queue Overview</h2>
          <span>{items.length} datasets</span>
        </div>
        <div className="summary-grid">
          <Metric label="In flight" value={runningCount} />
          <Metric label="QC passed" value={readyCount} />
          <Metric label="Failed" value={failedCount} />
          <Metric label="Total" value={items.length} />
        </div>
      </section>
      <div className="processing-workspace">
        <section className="panel processing-list-panel">
          <div className="panel-heading">
            <h2>Datasets</h2>
            <span>job queue</span>
          </div>
          {items.length === 0 ? (
            snapshot.gateway.state !== "online" ? (
              <div className="empty-dataset-list">
                <p>Gateway not connected (state: {snapshot.gateway.state}).</p>
                <p>Ensure the gateway is running on Thor:</p>
                <code>ssh nvidia@192.168.111.122</code>
                <code>bash ~/lerobot/run/restart_gateway.sh</code>
                <p>Then set the Vite proxy target:</p>
                <code>GUI_API_TARGET=http://192.168.111.122:8765 npm run dev</code>
              </div>
            ) : (
              <div className="empty-dataset-list">
                No datasets found under <code>{currentRoot || "(unset)"}</code>. Update the path above or
                record an episode first.
              </div>
            )
          ) : (
            <div className="processing-list">
              {items.map((item) => (
                <ProcessingRow
                  key={item.path}
                  item={item}
                  active={item.path === selected?.path}
                  busy={busy}
                  onSelect={() => setSelectedPath(item.path)}
                  onGenerate={() => onGenerate(item.path, markerTcpForRun, trackingTarget)}
                  onRunQc={() => onRunQc(item.path)}
                  onOpenReplay={() => onOpenReplay(item.path)}
                />
              ))}
            </div>
          )}
        </section>
        {selected ? (
          <section className="panel processing-detail-panel">
            <div className="panel-heading">
              <h2>{selected.name}</h2>
              <span className="state-pill">
                <StatusDot state={processingStatusDot[selected.status]} />
                {processingStatusLabel[selected.status]}
              </span>
            </div>
            <div className="summary-grid">
              <Metric label="Episodes" value={selected.totalEpisodes} />
              <Metric label="Frames" value={selected.totalFrames} />
              <Metric label="Trajectory" value={selected.trajectoryVersion ?? "—"} />
              <Metric label="Valid frames" value={selected.validFramesPct != null ? `${selected.validFramesPct}%` : "—"} />
              <Metric label="Tracking target" value={selected.detectionSummary?.label ?? selected.trackingTarget ?? "AprilTag Cube"} />
              <Metric
                label="Marker→TCP"
                value={
                  (selected.trackingTarget ?? "april_cube") === "hybrid_carrier_v1"
                    ? "CAD socket origin"
                    : selected.markerTcpCalibrationPath || "default"
                }
              />
            </div>
            <div className="control-row">
              <button
                disabled={busy || selected.status === "running" || selected.status === "queued"}
                onClick={() => onGenerate(selected.path, markerTcpForRun, trackingTarget)}
              >
                {selected.trajectoryVersion ? "Regenerate Trajectory" : "Generate EE Trajectory"}
              </button>
              <button
                disabled={
                  busy ||
                  !["pose_ready", "qc_pass", "qc_failed", "error"].includes(selected.status) ||
                  // trajectory generation failed: nothing to QC until a trajectory exists
                  (selected.status === "error" && !selected.trajectoryVersion)
                }
                onClick={() => onRunQc(selected.path)}
              >
                {selected.status === "qc_failed" || (selected.status === "error" && selected.trajectoryVersion)
                  ? "Re-run QC"
                  : "Run QC"}
              </button>
              <button
                disabled={busy || !["pose_ready", "qc_pass"].includes(selected.status)}
                onClick={() => onOpenReplay(selected.path)}
              >
                Open Replay
              </button>
            </div>
            <TrackingDetectionBlock item={selected} busy={busy} onOpenReplay={() => onOpenReplay(selected.path)} />
            <div className="qc-block">
              <h3>QC summary</h3>
              <p>{selected.qcSummary}</p>
              {selected.qcChecks?.length ? (
                <div className="qc-diagnostic-list">
                  {selected.qcChecks.map((check, index) => (
                    <div className={`qc-diagnostic-row qc-${check.status}`} key={`${check.name}-${index}`}>
                      <strong>{check.name}</strong>
                      <span>{check.status}</span>
                      <p>{check.message}</p>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
            {selected.ikEvaluation ? (
              <div className="qc-block ik-evaluation-block">
                <div className="qc-block-heading">
                  <h3>FR3 offline IK evaluation</h3>
                  <span className={`sync-status sync-status-${selected.ikEvaluation.status === "pass" ? "pass" : selected.ikEvaluation.status === "skipped" ? "missing" : "fail"}`}>
                    {selected.ikEvaluation.status}
                  </span>
                </div>
                <p>{selected.ikEvaluation.message}</p>
                <div className="ik-cube-list">
                  {selected.ikEvaluation.cubes.map((cube, index) => {
                    const ratio = typeof cube.reachableRatio === "number" ? cube.reachableRatio * 100 : null;
                    const reachableEpisodes = cube.reachableEpisodeIndices ?? [];
                    const unreachableEpisodes = cube.unreachableEpisodeIndices ?? [];
                    const plotParams = new URLSearchParams({
                      path: selected.path,
                      cube: cube.cube,
                      t: selected.updatedAt
                    });
                    return (
                      <div className="ik-cube-card" key={`${cube.cube}-${index}`}>
                        <div className="ik-cube-summary">
                          <strong>{cube.cube}</strong>
                          <span className={`ik-result-badge ik-result-${cube.status}`}>{cube.status}</span>
                          <span>{ratio == null ? "—" : `${ratio.toFixed(2)}% poses reachable`}</span>
                          <small>{cube.message || "No IK summary"}</small>
                        </div>
                        {reachableEpisodes.length || unreachableEpisodes.length ? (
                          <p className="ik-episode-answer">
                            <strong>Reachable:</strong> {reachableEpisodes.length ? reachableEpisodes.map((episode) => `episode ${episode}`).join(", ") : "none"}
                            <span> · </span>
                            <strong>Unreachable:</strong> {unreachableEpisodes.length ? unreachableEpisodes.map((episode) => `episode ${episode}`).join(", ") : "none"}
                          </p>
                        ) : null}
                        {cube.episodes?.length ? (
                          <div className="ik-episode-table">
                            <div className="ik-episode-table-header">
                              <span>Episode</span><span>Result</span><span>Reachable poses</span><span>Max error</span>
                            </div>
                            {cube.episodes.map((episode) => (
                              <div className={`ik-episode-row ik-episode-${episode.status}`} key={episode.episodeIndex}>
                                <strong>{episode.episodeIndex}</strong>
                                <span>{episode.status}</span>
                                <span>{episode.numReachable}/{episode.numTargets} ({(episode.reachableRatio * 100).toFixed(2)}%)</span>
                                <span>{episode.maxPositionErrorMm.toFixed(2)} mm · {episode.maxOrientationErrorDeg.toFixed(2)}°</span>
                              </div>
                            ))}
                          </div>
                        ) : null}
                        {cube.plotAvailable ? (
                          <figure className="ik-error-plot">
                            <img
                              src={`/api/processing/ik-plot?${plotParams.toString()}`}
                              alt={`${cube.cube} FR3 IK position and orientation error over time`}
                              loading="lazy"
                            />
                            <figcaption>{cube.cube} IK error over time</figcaption>
                          </figure>
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : null}
            <OnlineSyncManifestBlock item={selected} />
            <div className="log-block">
              <h3>Log</h3>
              {selected.logTail.length === 0 ? (
                <p className="panel-note">No log entries yet.</p>
              ) : (
                <pre className="log-tail">{selected.logTail.join("\n")}</pre>
              )}
            </div>
          </section>
        ) : null}
      </div>
      <QualityOverview snapshot={snapshot} />
    </div>
  );
}
