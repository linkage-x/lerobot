import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pose3DViewer } from "./Pose3DViewer";
import { SeriesPlot } from "./SeriesPlot";
import type { DataCollectionGuiApi } from "./api";
import type { CubeVideoOverlay, EePose, ReplayTimeline, ReplayTimelineFrame } from "./types";

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
  const totalFrames = timeline?.totalFrames ?? 0;

  useEffect(() => {
    if (!playing || totalFrames === 0) {
      return;
    }
    const intervalMs = 1000 / Math.max(fps, 1);
    const timer = window.setInterval(() => {
      setCurrentFrame((value) => {
        const next = value + 1;
        if (next >= totalFrames) {
          setPlaying(false);
          return totalFrames - 1;
        }
        return next;
      });
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [playing, fps, totalFrames]);

  useEffect(() => {
    if (!timeline) {
      return;
    }
    const t = (timeline.frames[currentFrame]?.timestamp ?? currentFrame / Math.max(fps, 1));
    Object.values(videoRefs.current).forEach((video) => {
      if (!video) {
        return;
      }
      const clamped = Math.max(0, Math.min(t, (video.duration || t)));
      if (!playing && Math.abs(video.currentTime - clamped) > 0.05) {
        try {
          video.currentTime = clamped;
        } catch {
          // ignore: browsers may throw if metadata is not yet loaded
        }
      }
    });
  }, [timeline, currentFrame, fps, playing]);

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

  const frame: ReplayTimelineFrame | undefined = timeline?.frames[currentFrame];
  const pose = ensureFullPose(frame?.eePose);
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
    if (!timeline) {
      return [] as Array<[number, number, number]>;
    }
    return timeline.frames
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
    (frameIndex: number, dim: number) => timeline?.frames[frameIndex]?.state[dim] ?? Number.NaN,
    [timeline]
  );
  const pickAction = useCallback(
    (frameIndex: number, dim: number) => timeline?.frames[frameIndex]?.action[dim] ?? Number.NaN,
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

  if (!timeline || timeline.totalFrames === 0) {
    return (
      <section className="panel inspector-panel">
        <div className="panel-heading">
          <h2>Replay Inspector</h2>
          <span>no data</span>
        </div>
        <p className="panel-note">{error ?? "No replay data available for this dataset."}</p>
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
        <button onClick={() => setPlaying((value) => !value)}>{playing ? "Pause" : "Play"}</button>
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
              preload="auto"
            />
            <CubeOverlayCanvas overlays={frame?.videoOverlays?.[key] ?? []} video={videoRefs.current[key]} />
            <span>{shortCameraName(key)}</span>
          </div>
        ))}
      </div>
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
