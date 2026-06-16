import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pose3DViewer } from "./Pose3DViewer";
import { SeriesPlot } from "./SeriesPlot";
import type { DataCollectionGuiApi } from "./api";
import type { CubeVideoOverlay, EePose, ReplayTimeline, ReplayTimelineFrame, TouchPadFrame } from "./types";

const cubeColors: Record<string, number> = {
  left: 0xc2410c,
  right: 0x0f766e,
  head: 0x2563eb
};

const cubePoseDims = ["x", "y", "z", "qx", "qy", "qz", "qw"] as const;
const cubeEdges: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7]
];

function shortCameraName(key: string): string {
  return key.replace(/^observation\.images\./, "");
}

const TOUCH_ROW_LENGTHS = [13, 13, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 13, 13];
const TOUCH_COLUMNS = 17;

function interpolateChannel(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

function touchColor(value: number, scaleMax: number): string {
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
  return `rgb(${interpolateChannel(a[0], b[0], t)}, ${interpolateChannel(a[1], b[1], t)}, ${interpolateChannel(a[2], b[2], t)})`;
}

function touchScaleMax(timeline: ReplayTimeline | null): number {
  let maxValue = 1;
  for (const entry of timeline?.frames ?? []) {
    for (const sample of Object.values(entry.touch ?? {})) {
      for (const value of sample?.fz ?? []) {
        if (Number.isFinite(value)) {
          maxValue = Math.max(maxValue, Math.abs(value));
        }
      }
    }
  }
  return maxValue;
}

function touchEntries(frame: ReplayTimelineFrame | undefined): Array<[string, TouchPadFrame | undefined]> {
  const entries = Object.entries(frame?.touch ?? {});
  if (entries.length === 0) {
    return [["left", undefined], ["right", undefined]];
  }
  const sideRank = (key: string) => key.endsWith(".left") || key === "left" ? 0 : 1;
  return entries.sort(([a], [b]) => sideRank(a) - sideRank(b) || a.localeCompare(b));
}

function TouchHeatmap({
  title,
  sample,
  scaleMax
}: {
  title: string;
  sample?: TouchPadFrame;
  scaleMax: number;
}) {
  const values = sample?.fz ?? [];
  const hasData = values.length >= 239;
  let cursor = 0;
  const localMax = hasData ? Math.max(...values.map((value) => Math.abs(value))) : 0;
  const activePoints = sample?.activePoints ?? values.filter((value) => Math.abs(value) > 0).length;

  return (
    <div className="touch-map">
      <div className="touch-map-heading">
        <strong>{title}</strong>
        <span>max {localMax.toFixed(1)} · active {activePoints}</span>
      </div>
      {hasData ? (
        <div className="touch-grid" aria-label={title}>
          {TOUCH_ROW_LENGTHS.map((length, rowIndex) => {
            const offset = Math.floor((TOUCH_COLUMNS - length) / 2);
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
                      style={{ backgroundColor: touchColor(Math.abs(value), scaleMax) }}
                    />
                  );
                })}
                {Array.from({ length: TOUCH_COLUMNS - length - offset }).map((_, index) => (
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
        <span>ts {sample?.timestamp ?? "—"}</span>
        <span>t {sample?.tRelS == null ? "—" : `${sample.tRelS.toFixed(3)}s`}</span>
      </div>
    </div>
  );
}


function ensureFullPose(pose: ReplayTimelineFrame["eePose"]): EePose | null {
  if (!pose) {
    return null;
  }
  const { x, y, z, qx, qy, qz, qw, gripper } = pose;
  if (x == null || y == null || z == null) {
    return null;
  }
  return {
    x,
    y,
    z,
    qx: qx ?? 0,
    qy: qy ?? 0,
    qz: qz ?? 0,
    qw: qw ?? 1,
    gripper: gripper == null ? null : gripper
  };
}

function CubeOverlayCanvas({
  overlays,
  video
}: {
  overlays: CubeVideoOverlay[];
  video: HTMLVideoElement | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !video) {
      return;
    }
    const width = video.clientWidth || canvas.clientWidth;
    const height = video.clientHeight || canvas.clientHeight;
    if (width <= 0 || height <= 0) {
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const sourceWidth = video.videoWidth || width;
    const sourceHeight = video.videoHeight || height;
    const sx = width / Math.max(sourceWidth, 1);
    const sy = height / Math.max(sourceHeight, 1);
    const scalePoint = (point: [number, number] | null | undefined): [number, number] | null => {
      if (!point) return null;
      return [point[0] * sx, point[1] * sy];
    };
    const drawLine = (a: [number, number] | null, b: [number, number] | null, color: string, lineWidth = 2) => {
      if (!a || !b) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    };

    for (const overlay of overlays) {
      const corners = overlay.corners.map(scalePoint);
      for (const [a, b] of cubeEdges) {
        drawLine(corners[a], corners[b], overlay.color, 2);
      }
      const origin = scalePoint(overlay.axes.origin);
      drawLine(origin, scalePoint(overlay.axes.x), "#ef4444", 2.5);
      drawLine(origin, scalePoint(overlay.axes.y), "#22c55e", 2.5);
      drawLine(origin, scalePoint(overlay.axes.z), "#3b82f6", 2.5);
      const label = scalePoint(overlay.label);
      if (label) {
        const text = `${overlay.cubeName} m=${overlay.numMarkers} rmse=${overlay.rmsePx == null ? "-" : overlay.rmsePx.toFixed(1)} ${overlay.usedForFusion ? "in" : "out"}`;
        ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
        const metrics = ctx.measureText(text);
        const x = Math.max(4, Math.min(width - metrics.width - 10, label[0]));
        const y = Math.max(18, Math.min(height - 6, label[1]));
        ctx.fillStyle = "rgba(0, 0, 0, 0.72)";
        ctx.fillRect(x - 4, y - 14, metrics.width + 8, 18);
        ctx.fillStyle = overlay.color;
        ctx.fillText(text, x, y);
      }
    }
  }, [overlays, video]);

  return <canvas className="camera-overlay" ref={canvasRef} />;
}

export function ReplayInspector({
  api,
  datasetPath,
  episode,
  fallbackFps
}: {
  api: DataCollectionGuiApi;
  datasetPath: string;
  episode: number;
  fallbackFps: number;
}) {
  const [timeline, setTimeline] = useState<ReplayTimeline | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const videoRefs = useRef<Record<string, HTMLVideoElement | null>>({});
  // Frame counts read from loaded video metadata, keyed per camera. A single
  // corrupt remux cache can report a tiny duration (e.g. 56 frames); using
  // the maximum across cameras keeps that one bad stream from truncating the
  // whole episode timeline.
  const [videoFrameCounts, setVideoFrameCounts] = useState<Record<string, number>>({});

  useEffect(() => {
    let cancelled = false;
    if (!datasetPath) {
      setTimeline(null);
      return;
    }
    setLoading(true);
    setError(null);
    setCurrentFrame(0);
    setPlaying(false);
    setTimeline(null);
    setVideoFrameCounts({});
    api
      .fetchReplayTimeline(datasetPath, episode)
      .then((result) => {
        if (cancelled) {
          return;
        }
        if (!result) {
          setError("Gateway did not return a timeline. Is the dataset under outputs/datasets?");
          setTimeline(null);
        } else if (result.error) {
          setError(result.error);
          setTimeline(result);
        } else {
          setTimeline(result);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [api, datasetPath, episode]);

  const fps = timeline?.fps ?? fallbackFps;
  const backendTotalFrames = timeline?.totalFrames ?? 0;
  // Effective playable frame count: keep the backend timeline as the primary
  // source, but allow a consistent shorter video duration to trim it. Taking
  // the max across cameras avoids one truncated MP4 cache shrinking every
  // plot and slider to its bogus duration.
  const videoFrameCountValues = Object.values(videoFrameCounts);
  const plausibleVideoFrameCounts = backendTotalFrames > 0
    ? videoFrameCountValues.filter((frames) => frames >= backendTotalFrames * 0.5)
    : videoFrameCountValues;
  const maxVideoFrameCount = plausibleVideoFrameCounts.length > 0 ? Math.max(...plausibleVideoFrameCounts) : null;
  const totalFrames = maxVideoFrameCount != null
    ? Math.min(backendTotalFrames || maxVideoFrameCount, maxVideoFrameCount)
    : backendTotalFrames;
  const videoWarmupS = Math.max(0, timeline?.videoWarmupS ?? 0);

  const handleVideoMetadataLoaded = useCallback(
    (key: string, event: React.SyntheticEvent<HTMLVideoElement>) => {
      const target = event.currentTarget;
      const dur = target.duration;
      if (!Number.isFinite(dur) || dur <= 0) return;
      const frames = Math.max(1, Math.round(dur * Math.max(fps, 1)));
      setVideoFrameCounts((prev) => (prev[key] === frames ? prev : { ...prev, [key]: frames }));
    },
    [fps],
  );

  useEffect(() => {
    if (!playing || totalFrames === 0) {
      return;
    }
    // Drive currentFrame from the first available <video>'s wall-clock
    // currentTime, not setInterval. setInterval(1000/fps) drifts in
    // browsers (16.67ms ticks get throttled to ~20ms under GPU/tab load),
    // so the timeline used to lag behind the video and the operator saw
    // the video finish ~1s before the slider reached the end.
    // requestAnimationFrame fires at display refresh and reads the same
    // wall-clock the video element uses, so the two stay locked.
    let rafId = 0;
    const tick = () => {
      const videos = Object.values(videoRefs.current).filter((v): v is HTMLVideoElement => v != null);
      const liveVideos = videos.filter((v) => !v.ended && Number.isFinite(v.currentTime));
      const candidates = liveVideos.length > 0 ? liveVideos : videos.filter((v) => Number.isFinite(v.currentTime));
      const master = candidates.reduce<HTMLVideoElement | null>(
        (best, video) => (best == null || video.currentTime > best.currentTime ? video : best),
        null,
      );
      if (master && Number.isFinite(master.currentTime)) {
        const t = Math.max(0, master.currentTime - videoWarmupS);
        const frame = Math.min(totalFrames - 1, Math.max(0, Math.round(t * fps)));
        setCurrentFrame(frame);
        const allEnded = videos.length > 0 && videos.every((video) => video.ended);
        if (frame >= totalFrames - 1 || allEnded) {
          setPlaying(false);
          return;
        }
      }
      rafId = window.requestAnimationFrame(tick);
    };
    rafId = window.requestAnimationFrame(tick);
    return () => {
      if (rafId) {
        window.cancelAnimationFrame(rafId);
      }
    };
  }, [playing, fps, totalFrames, videoWarmupS]);

  useEffect(() => {
    if (!timeline) {
      return;
    }
    const t = (timeline.frames?.[currentFrame]?.timestamp ?? currentFrame / Math.max(fps, 1));
    Object.values(videoRefs.current).forEach((video) => {
      if (!video) {
        return;
      }
      const target = t + videoWarmupS;
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : Number.POSITIVE_INFINITY;
      const clamped = Math.max(0, Math.min(target, duration));
      if (!playing && Math.abs(video.currentTime - clamped) > 0.05) {
        try {
          video.currentTime = clamped;
        } catch {
          // ignore: browsers may throw if metadata is not yet loaded
        }
      }
    });
  }, [timeline, currentFrame, fps, playing, videoWarmupS]);

  useEffect(() => {
    Object.values(videoRefs.current).forEach((video) => {
      if (!video) {
        return;
      }
      if (playing) {
        video.play().catch(() => {
          // suppressed: user gesture might still be required
        });
      } else {
        video.pause();
      }
    });
  }, [playing]);

  // Backend may legitimately omit `frames` on error responses (pyarrow
  // missing, episode out of range, etc). Don't dereference frames[idx]
  // without a `?.` guard or this component crashes before the totalFrames
  // early-return below has a chance to render the placeholder panel.
  const frame: ReplayTimelineFrame | undefined = timeline?.frames?.[currentFrame];
  const pose = ensureFullPose(frame?.eePose);
  const touchMax = useMemo(() => touchScaleMax(timeline), [timeline]);
  const cubePoseNames = useMemo(() => {
    if (!timeline) {
      return [] as string[];
    }
    const discovered = new Set<string>(timeline.cubePoseNames ?? []);
    for (const entry of timeline.frames) {
      for (const name of Object.keys(entry.cubePoses ?? {})) {
        discovered.add(name);
      }
    }
    return Array.from(discovered).filter((name) =>
      timeline.frames.some((entry) => ensureFullPose(entry.cubePoses?.[name]) !== null)
    );
  }, [timeline]);
  const trajectory = useMemo(() => {
    const frames = timeline?.frames;
    if (!frames || frames.length === 0) {
      return [] as Array<[number, number, number]>;
    }
    return frames
      .map((entry) => entry.eePose)
      .filter((entry): entry is EePose => !!entry && entry.x != null && entry.y != null && entry.z != null)
      .map((entry) => [entry.x as number, entry.y as number, entry.z as number] as [number, number, number]);
  }, [timeline]);
  const cubeTrajectories = useMemo(() => {
    if (!timeline) {
      return [];
    }
    return cubePoseNames.map((name, index) => ({
      name,
      color: cubeColors[name] ?? [0x7c3aed, 0x0891b2, 0x65a30d, 0xbe123c][index % 4],
      points: timeline.frames
        .map((entry) => ensureFullPose(entry.cubePoses?.[name]))
        .filter((entry): entry is EePose => !!entry)
        .map((entry) => [entry.x, entry.y, entry.z] as [number, number, number])
    }));
  }, [timeline, cubePoseNames]);
  const currentCubePoses = useMemo(() => {
    return cubePoseNames.map((name, index) => ({
      name,
      color: cubeColors[name] ?? [0x7c3aed, 0x0891b2, 0x65a30d, 0xbe123c][index % 4],
      pose: ensureFullPose(frame?.cubePoses?.[name])
    }));
  }, [cubePoseNames, frame]);

  const pickState = useCallback(
    (frameIndex: number, dim: number) => timeline?.frames?.[frameIndex]?.state?.[dim] ?? Number.NaN,
    [timeline]
  );
  const pickAction = useCallback(
    (frameIndex: number, dim: number) => timeline?.frames?.[frameIndex]?.action?.[dim] ?? Number.NaN,
    [timeline]
  );
  const cubeSeriesNames = useMemo(() => {
    return cubePoseNames.flatMap((name) => cubePoseDims.map((dim) => `${name}.${dim}`));
  }, [cubePoseNames]);
  const pickCubePose = useCallback(
    (frameIndex: number, dim: number) => {
      const cubeName = cubePoseNames[Math.floor(dim / cubePoseDims.length)];
      const poseDim = cubePoseDims[dim % cubePoseDims.length];
      const cubePose = ensureFullPose(timeline?.frames[frameIndex]?.cubePoses?.[cubeName]);
      return cubePose?.[poseDim] ?? Number.NaN;
    },
    [timeline, cubePoseNames]
  );

  const seek = useCallback((nextFrame: number) => {
    setPlaying(false);
    setCurrentFrame(Math.max(0, Math.min(nextFrame, totalFrames - 1)));
  }, [totalFrames]);

  const togglePlay = useCallback(() => {
    if (playing) {
      setPlaying(false);
      return;
    }
    // Starting playback while already parked on the last frame would be
    // killed instantly by the raf tick's end-of-clip check, so rewind to the
    // start first. The currentFrame->video sync effect skips seeking while
    // `playing` is true, so rewind the <video> elements here directly.
    if (totalFrames > 0 && currentFrame >= totalFrames - 1) {
      setCurrentFrame(0);
      Object.values(videoRefs.current).forEach((video) => {
        if (!video) {
          return;
        }
        try {
          video.currentTime = videoWarmupS;
        } catch {
          // ignore: browsers may throw if metadata is not yet loaded
        }
      });
    }
    setPlaying(true);
  }, [playing, currentFrame, totalFrames, videoWarmupS]);

  if (!datasetPath) {
    return null;
  }

  if (loading && !timeline) {
    return (
      <section className="panel inspector-panel">
        <div className="panel-heading">
          <h2>Replay Inspector</h2>
          <span>switching episode</span>
        </div>
        <p className="panel-note">
          Releasing the previous episode video streams and loading episode {episode} timeline. This can take a moment for large videos or cold parquet reads.
        </p>
      </section>
    );
  }

  if (!timeline || timeline.totalFrames === 0 || !timeline.frames || timeline.frames.length === 0) {
    const note =
      error ??
      timeline?.error ??
      "No replay data available for this dataset.";
    return (
      <section className="panel inspector-panel">
        <div className="panel-heading">
          <h2>Replay Inspector</h2>
          <span>no data</span>
        </div>
        <p className="panel-note">{note}</p>
      </section>
    );
  }

  return (
    <section className="panel inspector-panel">
      <div className="panel-heading">
        <h2>{timeline.name}</h2>
        <span>
          episode {timeline.episode} · {totalFrames} frames @ {fps} fps
        </span>
      </div>
      <div className="inspector-toolbar">
        <button onClick={togglePlay}>{playing ? "Pause" : "Play"}</button>
        <input
          type="range"
          min={0}
          max={Math.max(totalFrames - 1, 0)}
          value={currentFrame}
          onChange={(event) => {
            setPlaying(false);
            setCurrentFrame(Number(event.target.value));
          }}
        />
        <div className="inspector-readout">
          <span>frame</span>
          <strong>{currentFrame}</strong>
          <span>ts</span>
          <strong>{frame?.timestamp.toFixed(3) ?? "—"} s</strong>
        </div>
      </div>
      <div className="camera-grid">
        {timeline.cameraKeys.map((key) => (
          <div className="camera-tile" key={key}>
            <video
              ref={(element) => {
                videoRefs.current[key] = element;
              }}
              src={api.videoUrl(timeline.datasetRoot, key)}
              muted
              playsInline
              // metadata: fetch the mkv header + first frame so the
              // tile renders immediately, but skip downloading the full
              // file until the user clicks Play. preload=auto here used
              // to queue 11 full mkvs at once on Connect, hitting the
              // browser's 6-per-origin HTTP/1.1 cap; cams 7-11 would
              // load slowly or only after a manual refresh.
              preload="metadata"
              onLoadedMetadata={(event) => handleVideoMetadataLoaded(key, event)}
            />
            <CubeOverlayCanvas overlays={frame?.videoOverlays?.[key] ?? []} video={videoRefs.current[key]} />
            <span>{shortCameraName(key)}</span>
          </div>
        ))}
      </div>
      <section className="panel touch-panel">
        <div className="panel-heading">
          <h2>Paxini touch</h2>
          <span>fz pseudo color · 239 points</span>
        </div>
        <div className="touch-heatmaps">
          {touchEntries(frame).map(([key, sample]) => (
            <TouchHeatmap key={key} title={key} sample={sample} scaleMax={touchMax} />
          ))}
        </div>
        <div className="touch-legend" aria-hidden="true">
          <span>0</span>
          <div />
          <span>{touchMax.toFixed(1)}</span>
        </div>
      </section>
      <section className="panel pose-panel">
        <div className="panel-heading">
          <h2>End-effector pose</h2>
          <span>
            drag to orbit · wheel to zoom{cubePoseNames.length ? ` · cubes ${cubePoseNames.join(", ")}` : ""}
          </span>
        </div>
        {pose ? (
          <div className="pose-summary">
            <span>
              pos [<strong>{pose.x.toFixed(3)}</strong>, <strong>{pose.y.toFixed(3)}</strong>, <strong>{pose.z.toFixed(3)}</strong>]
            </span>
            <span>
              quat [<strong>{pose.qx.toFixed(3)}</strong>, <strong>{pose.qy.toFixed(3)}</strong>, <strong>{pose.qz.toFixed(3)}</strong>, <strong>{pose.qw.toFixed(3)}</strong>]
            </span>
            <span>
              gripper <strong>{pose.gripper == null ? "—" : pose.gripper.toFixed(3)}</strong>
            </span>
            <span className="pose-debug">
              raw fields: [{frame?.eePose ? Object.keys(frame.eePose).join(", ") : "(none)"}]
            </span>
          </div>
        ) : (
          <p className="panel-note">No EE pose in this frame.</p>
        )}
        {cubePoseNames.length ? (
          <div className="pose-summary">
            {currentCubePoses.map((entry) => (
              <span key={entry.name}>
                {entry.name} [<strong>{entry.pose?.x.toFixed(3) ?? "—"}</strong>,{" "}
                <strong>{entry.pose?.y.toFixed(3) ?? "—"}</strong>,{" "}
                <strong>{entry.pose?.z.toFixed(3) ?? "—"}</strong>]
              </span>
            ))}
          </div>
        ) : null}
        <Pose3DViewer
          trajectory={trajectory}
          currentPose={pose}
          extraTrajectories={cubeTrajectories}
          currentExtraPoses={currentCubePoses}
        />
      </section>
      <div className="series-grid">
        <SeriesPlot
          title="observation.state"
          names={timeline.stateNames}
          pickValue={pickState}
          currentFrame={currentFrame}
          totalFrames={totalFrames}
          onSeek={seek}
        />
        <SeriesPlot
          title="action"
          names={timeline.actionNames}
          pickValue={pickAction}
          currentFrame={currentFrame}
          totalFrames={totalFrames}
          onSeek={seek}
        />
        {cubeSeriesNames.length ? (
          <SeriesPlot
            title="cube pose trajectories"
            names={cubeSeriesNames}
            pickValue={pickCubePose}
            currentFrame={currentFrame}
            totalFrames={totalFrames}
            onSeek={seek}
            rowHeight={22}
          />
        ) : null}
      </div>
      {error ? <p className="panel-note error">{error}</p> : null}
    </section>
  );
}
