import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pose3DViewer } from "./Pose3DViewer";
import { SeriesPlot } from "./SeriesPlot";
import type { DataCollectionGuiApi } from "./api";
import type { EePose, ReplayTimeline, ReplayTimelineFrame } from "./types";

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
    const t = (timeline.frames?.[currentFrame]?.timestamp ?? currentFrame / Math.max(fps, 1));
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

  // Backend may legitimately omit `frames` on error responses (pyarrow
  // missing, episode out of range, etc). Don't dereference frames[idx]
  // without a `?.` guard or this component crashes before the totalFrames
  // early-return below has a chance to render the placeholder panel.
  const frame: ReplayTimelineFrame | undefined = timeline?.frames?.[currentFrame];
  const pose = ensureFullPose(frame?.eePose);
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

  const pickState = useCallback(
    (frameIndex: number, dim: number) => timeline?.frames?.[frameIndex]?.state?.[dim] ?? Number.NaN,
    [timeline]
  );
  const pickAction = useCallback(
    (frameIndex: number, dim: number) => timeline?.frames?.[frameIndex]?.action?.[dim] ?? Number.NaN,
    [timeline]
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
            <span>{shortCameraName(key)}</span>
          </div>
        ))}
      </div>
      <section className="panel pose-panel">
        <div className="panel-heading">
          <h2>End-effector pose</h2>
          <span>drag to orbit · wheel to zoom</span>
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
        <Pose3DViewer trajectory={trajectory} currentPose={pose} />
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
      </div>
      {error ? <p className="panel-note error">{error}</p> : null}
    </section>
  );
}
