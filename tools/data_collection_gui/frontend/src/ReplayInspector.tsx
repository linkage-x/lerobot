import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MujocoReplayViewer } from "./MujocoReplayViewer";
import { Pose3DViewer } from "./Pose3DViewer";
import { SeriesPlot } from "./SeriesPlot";
import type { DataCollectionGuiApi } from "./api";
import type { CameraControlsMetadata, CubeVideoOverlay, EePose, ForceVector, MujocoCubeMode, MujocoPreview, ReplayStatus, ReplayTimeline, ReplayTimelineFrame, TouchPadFrame } from "./types";
import { TouchHeatmapGrid, touchLayoutForCount, touchSampleActivePoints, touchSampleHasShear, touchSampleLocalMax, touchScaleFromSamples, type TouchScale } from "./touchVisualization";
import { timelineTimeToVideoTime as toVideoTime, videoTimeToTimelineTime as toTimelineTime } from "./shared/videoOffsets";

const cubeColors: Record<string, number> = {
  left: 0xc2410c,
  right: 0x0f766e,
  head: 0x2563eb
};

const cubePoseDims = ["x", "y", "z", "qx", "qy", "qz", "qw"] as const;

function cubePoseSeriesName(name: string, dim: (typeof cubePoseDims)[number]): string {
  return dim.length === 1 ? `${name}.position_${dim}` : `${name}.quat_${dim.slice(1)}`;
}

const cubeEdges: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7]
];

function shortCameraName(key: string): string {
  return key.replace(/^observation\.images\./, "");
}

function asRecord(value: unknown): Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function displayCameraValue(value: unknown, unit = ""): string {
  if (value == null || value === "") return "—";
  if (typeof value === "number") return `${Number.isInteger(value) ? value : value.toFixed(3)}${unit}`;
  return `${String(value)}${unit}`;
}

function displayEnabled(value: unknown): string {
  if (typeof value === "boolean") return value ? "enabled" : "disabled";
  if (typeof value === "number") return value !== 0 ? "enabled" : "disabled";
  return displayCameraValue(value);
}

function cameraControlRows(entry: CameraControlsMetadata["cameras"][string]): Array<[string, string]> {
  const requested = asRecord(entry.requested);
  const effective = asRecord(entry.effective);
  const device = asRecord(effective.device);
  const stream = asRecord(effective.stream);
  const controls = asRecord(effective.controls);
  const requestedStream = [requested.width, requested.height, requested.fps].every((value) => value != null)
    ? `${requested.width}×${requested.height} @ ${requested.fps} fps`
    : "—";
  const effectiveStream = [stream.width, stream.height, stream.fps].every((value) => value != null)
    ? `${stream.width}×${stream.height} @ ${stream.fps} fps${stream.format ? ` · ${String(stream.format)}` : ""}`
    : "—";
  const rows: Array<[string, string]> = [
    ["Requested stream", requestedStream],
    ["Effective stream", effectiveStream],
    ["Model", displayCameraValue(device.name)],
    ["Serial", displayCameraValue(device.serial_number ?? requested.serial_number_or_name)],
    ["Firmware", displayCameraValue(device.firmware_version)],
    ["USB", displayCameraValue(device.usb_type_descriptor)],
    ["Auto exposure", displayEnabled(controls.enable_auto_exposure)],
    ["AE priority", displayEnabled(controls.auto_exposure_priority)],
    ["Exposure", displayCameraValue(controls.exposure, " µs")],
    ["Gain", displayCameraValue(controls.gain)],
    ["Auto white balance", displayEnabled(controls.enable_auto_white_balance)],
    ["White balance", displayCameraValue(controls.white_balance, " K")],
  ];
  return rows.filter(([, value]) => value !== "—");
}

function ReplayTransport({
  playing,
  onToggle,
  currentFrame,
  timestamp
}: {
  playing: boolean;
  onToggle: () => void;
  currentFrame: number;
  timestamp?: number;
}) {
  return (
    <div className="replay-local-transport">
      <button onClick={onToggle} type="button">{playing ? "Pause" : "Play"}</button>
      <div className="inspector-readout">
        <span>frame</span>
        <strong>{currentFrame}</strong>
        <span>ts</span>
        <strong>{timestamp?.toFixed(3) ?? "—"} s</strong>
      </div>
    </div>
  );
}

function touchScaleMax(timeline: ReplayTimeline | null): TouchScale {
  const samples: TouchPadFrame[] = [];
  for (const entry of timeline?.frames ?? []) {
    for (const sample of Object.values(entry.touch ?? {})) {
      if (sample) {
        samples.push(sample);
      }
    }
  }
  return touchScaleFromSamples(samples);
}

function touchPanelSummary(frame: ReplayTimelineFrame | undefined): string {
  const samples = Object.values(frame?.touch ?? {}).filter((sample): sample is TouchPadFrame => Boolean(sample));
  const firstWithData = samples.find((sample) => sample.fz.length > 0);
  const layout = firstWithData ? touchLayoutForCount(firstWithData.fz.length) : null;
  const mode = samples.some(touchSampleHasShear) ? "fz + fx/fy shear" : "fz pseudo color";
  return layout ? `${mode} · ${layout.label}` : mode;
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
  scale
}: {
  title: string;
  sample?: TouchPadFrame;
  scale: TouchScale;
}) {
  const values = sample?.fz ?? [];
  const hasData = values.length > 0;
  const localMax = hasData ? touchSampleLocalMax(sample) : 0;
  const activePoints = sample?.activePoints ?? touchSampleActivePoints(sample);
  const layout = hasData ? touchLayoutForCount(values.length) : null;
  const hasShear = touchSampleHasShear(sample);

  return (
    <div className="touch-map">
      <div className="touch-map-heading">
        <strong>{title}</strong>
        <span>max {localMax.toFixed(1)} · active {activePoints}</span>
      </div>
      {hasData ? (
        <TouchHeatmapGrid sample={sample} scale={scale} ariaLabel={title} />
      ) : (
        <div className="touch-empty">no touch sample</div>
      )}
      <div className="touch-map-footer">
        <span>ts {sample?.timestamp ?? "—"}</span>
        <span>{layout?.label ?? "—"}{hasShear ? " · shear hue" : ""}</span>
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

function ensureForceVector(force: ReplayTimelineFrame["forceVector"]): ForceVector | null {
  if (!force || force.x == null || force.y == null || force.z == null) {
    return null;
  }
  const magnitude = force.magnitude ?? Math.hypot(force.x, force.y, force.z);
  if (![force.x, force.y, force.z, magnitude].every(Number.isFinite)) {
    return null;
  }
  return { x: force.x, y: force.y, z: force.z, magnitude };
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
  fallbackFps,
  revision = 0,
  mujocoMode,
  onMujocoModeChange,
  onRunMujoco,
  onApproveMujoco,
  replayStatus,
  busy,
  mujocoRefreshKey = "",
  cubeSelection = true
}: {
  api: DataCollectionGuiApi;
  datasetPath: string;
  episode: number;
  fallbackFps: number;
  // Changes when the dataset content behind an unchanged (datasetPath, episode)
  // selection is mutated (e.g. deleting an episode renumbers the survivors into
  // the same slot). Part of the fetch effect deps so the timeline refetches.
  revision?: number;
  mujocoMode: MujocoCubeMode;
  onMujocoModeChange: (mode: MujocoCubeMode) => void;
  onRunMujoco: (mode: MujocoCubeMode) => void;
  onApproveMujoco: (mode: MujocoCubeMode) => void;
  replayStatus: ReplayStatus;
  busy: boolean;
  mujocoRefreshKey?: string;
  // Whether this rig tracks AprilTag cubes. The workstation does not: it replays the arm's own
  // recorded EE command stream, the gateway ignores the cube mode for it, and the saved-report
  // approval those controls drive reads a `mujoco_preview.<cube>` file it never writes.
  cubeSelection?: boolean;
}) {
  const [timeline, setTimeline] = useState<ReplayTimeline | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [mujocoPreview, setMujocoPreview] = useState<MujocoPreview | null>(null);
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
  }, [api, datasetPath, episode, revision]);

  const mujocoRunning = replayStatus.state === "sim_replay";

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    setMujocoPreview(null);
    if (!datasetPath) return;

    const loadPreview = () => {
      api.fetchMujocoPreview(datasetPath, episode, mujocoMode).then((result) => {
        if (!cancelled) setMujocoPreview(result);
      });
    };

    loadPreview();
    if (mujocoRunning) {
      timer = window.setInterval(loadPreview, 300);
    }
    return () => {
      cancelled = true;
      if (timer !== null) window.clearInterval(timer);
    };
  }, [api, datasetPath, episode, mujocoMode, mujocoRefreshKey, mujocoRunning]);

  const fps = timeline?.fps ?? fallbackFps;
  const replayActive = mujocoRunning || replayStatus.state === "replaying";
  const canRunMujoco =
    replayStatus.dataStatus === "loaded" &&
    (replayStatus.recordedFrames ?? replayStatus.totalFrames) > 0;

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
  const firstTimelineTime = timeline?.frames?.[0]?.timestamp ?? 0;
  const cameraVideoOffsetsS: Record<string, number> = timeline?.cameraVideoOffsetsS ?? {};
  const timelineTimeToVideoTime = useCallback((key: string, timelineTimeS: number): number => {
    return toVideoTime(cameraVideoOffsetsS, key, timelineTimeS, videoWarmupS);
  }, [cameraVideoOffsetsS, videoWarmupS]);
  const videoTimeToTimelineTime = useCallback((key: string, videoTimeS: number): number => {
    return toTimelineTime(cameraVideoOffsetsS, key, videoTimeS, videoWarmupS);
  }, [cameraVideoOffsetsS, videoWarmupS]);
  const syncVideoToTimelineTime = useCallback((
    key: string,
    video: HTMLVideoElement,
    timelineTimeS: number,
    toleranceS: number,
  ) => {
    const target = timelineTimeToVideoTime(key, timelineTimeS);
    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : Number.POSITIVE_INFINITY;
    const clamped = Math.max(0, Math.min(target, duration));
    if (Math.abs(video.currentTime - clamped) <= toleranceS) {
      return;
    }
    try {
      video.currentTime = clamped;
    } catch {
      // ignore: browsers may throw if metadata is not yet loaded
    }
  }, [timelineTimeToVideoTime]);

  const handleVideoMetadataLoaded = useCallback(
    (key: string, event: React.SyntheticEvent<HTMLVideoElement>) => {
      const target = event.currentTarget;
      const dur = target.duration;
      if (!Number.isFinite(dur) || dur <= 0) return;
      const frames = Math.max(1, Math.round(dur * Math.max(fps, 1)));
      setVideoFrameCounts((prev) => (prev[key] === frames ? prev : { ...prev, [key]: frames }));
      if (!playing && timeline) {
        const timelineTimeS = timeline.frames?.[currentFrame]?.timestamp ?? currentFrame / Math.max(fps, 1);
        syncVideoToTimelineTime(key, target, timelineTimeS, 0.0);
      }
    },
    [currentFrame, fps, playing, syncVideoToTimelineTime, timeline],
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
      const videos = Object.entries(videoRefs.current)
        .filter((entry): entry is [string, HTMLVideoElement] => entry[1] != null);
      const liveVideos = videos.filter(([, video]) => !video.ended && Number.isFinite(video.currentTime));
      const candidates = liveVideos.length > 0
        ? liveVideos
        : videos.filter(([, video]) => Number.isFinite(video.currentTime));
      const master = candidates.reduce<[string, HTMLVideoElement] | null>(
        (best, entry) => {
          if (best == null) {
            return entry;
          }
          const entryTime = videoTimeToTimelineTime(entry[0], entry[1].currentTime);
          const bestTime = videoTimeToTimelineTime(best[0], best[1].currentTime);
          return entryTime > bestTime ? entry : best;
        },
        null,
      );
      if (master && Number.isFinite(master[1].currentTime)) {
        const t = videoTimeToTimelineTime(master[0], master[1].currentTime);
        const frame = Math.min(totalFrames - 1, Math.max(0, Math.round(t * fps)));
        setCurrentFrame(frame);
        const syncToleranceS = Math.max(0.006, 0.5 / Math.max(fps, 1));
        videos.forEach(([key, video]) => {
          if (key !== master[0]) {
            syncVideoToTimelineTime(key, video, t, syncToleranceS);
          }
        });
        const allEnded = videos.length > 0 && videos.every(([, video]) => video.ended);
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
  }, [playing, fps, totalFrames, syncVideoToTimelineTime, videoTimeToTimelineTime]);

  useEffect(() => {
    if (!timeline) {
      return;
    }
    const t = (timeline.frames?.[currentFrame]?.timestamp ?? currentFrame / Math.max(fps, 1));
    if (!playing) {
      const seekToleranceS = Math.max(0.002, 0.25 / Math.max(fps, 1));
      Object.entries(videoRefs.current).forEach(([key, video]) => {
        if (video) {
          syncVideoToTimelineTime(key, video, t, seekToleranceS);
        }
      });
    }
  }, [timeline, currentFrame, fps, playing, syncVideoToTimelineTime]);

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
  const forceVector = ensureForceVector(frame?.forceVector);
  const touchMax = useMemo(() => touchScaleMax(timeline), [timeline]);
  const touchSummary = useMemo(() => touchPanelSummary(frame), [frame]);
  // Decided from the episode, not from the current frame: a rig without Paxini pads (the FR3
  // workstation runs a Pika gripper) otherwise gets a panel named after a sensor it does not
  // have, showing two fabricated empty pads for the whole recording.
  const hasTouchData = useMemo(
    () => (timeline?.frames ?? []).some((entry) => Object.values(entry.touch ?? {}).some(Boolean)),
    [timeline]
  );
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
    return cubePoseNames.flatMap((name) => cubePoseDims.map((dim) => cubePoseSeriesName(name, dim)));
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
      Object.entries(videoRefs.current).forEach(([key, video]) => {
        if (video) {
          syncVideoToTimelineTime(key, video, firstTimelineTime, 0.0);
        }
      });
    }
    setPlaying(true);
  }, [playing, currentFrame, totalFrames, firstTimelineTime, syncVideoToTimelineTime]);

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
              src={api.videoUrl(timeline.datasetRoot, key, timeline.episode)}
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
      {timeline.cameraControls && Object.keys(timeline.cameraControls.cameras).length ? (
        <section className="panel camera-controls-panel">
          <div className="panel-heading">
            <h2>Camera capture settings</h2>
            <span>{timeline.cameraControls.captured_at} · {timeline.cameraControls.backend ?? "recording"}</span>
          </div>
          <p className="panel-note">{timeline.cameraControls.source ?? "Settings captured when this recording session connected."}</p>
          <div className="camera-controls-grid">
            {Object.entries(timeline.cameraControls.cameras).map(([name, entry]) => (
              <article className="camera-controls-card" key={name}>
                <h3>{shortCameraName(name)} <small>{entry.type ?? "camera"} · {entry.status ?? "recorded"}</small></h3>
                {entry.message ? <p className="panel-note">{entry.message}</p> : null}
                <dl>
                  {cameraControlRows(entry).map(([label, value]) => (
                    <div key={label}><dt>{label}</dt><dd>{value}</dd></div>
                  ))}
                </dl>
              </article>
            ))}
          </div>
        </section>
      ) : null}
      {hasTouchData ? (
        <section className="panel touch-panel">
          <div className="panel-heading">
            <h2>Paxini touch</h2>
            <span>{touchSummary}</span>
          </div>
          <div className="touch-heatmaps">
            {touchEntries(frame).map(([key, sample]) => (
              <TouchHeatmap key={key} title={key} sample={sample} scale={touchMax} />
            ))}
          </div>
          <div className="touch-legend" aria-hidden="true">
            <span>0</span>
            <div />
            <span>{touchMax.normalMax.toFixed(1)}</span>
          </div>
        </section>
      ) : null}
      <section className="panel pose-panel">
        <div className="panel-heading">
          <h2>End-effector pose</h2>
          <span>
            drag to orbit · wheel to zoom{cubePoseNames.length ? ` · cubes ${cubePoseNames.join(", ")}` : ""}
          </span>
        </div>
        <ReplayTransport
          playing={playing}
          onToggle={togglePlay}
          currentFrame={currentFrame}
          timestamp={frame?.timestamp}
        />
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
            <span>
              F [<strong>{forceVector?.x.toFixed(3) ?? "—"}</strong>, <strong>{forceVector?.y.toFixed(3) ?? "—"}</strong>, <strong>{forceVector?.z.toFixed(3) ?? "—"}</strong>] N |F| <strong>{forceVector?.magnitude?.toFixed(3) ?? "—"}</strong>
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
          forceVector={forceVector}
          extraTrajectories={cubeTrajectories}
          currentExtraPoses={currentCubePoses}
        />
      </section>
      <section className="panel pose-panel mujoco-panel">
        <div className="panel-heading">
          <h2>MuJoCo EE trajectory replay</h2>
          <span>selected dataset · episode {episode} · Three.js/WebGL live view</span>
        </div>
        <div className="mujoco-local-controls">
          {cubeSelection ? (
            <div className="mujoco-mode-picker" role="group" aria-label="MuJoCo cube trajectory">
              {(["left", "right", "both"] as MujocoCubeMode[]).map((mode) => (
                <button
                  key={mode}
                  className={mujocoMode === mode ? "active" : ""}
                  disabled={busy || replayActive}
                  onClick={() => onMujocoModeChange(mode)}
                  type="button"
                >
                  {mode === "both" ? "Both cubes" : `${mode[0].toUpperCase()}${mode.slice(1)} cube`}
                </button>
              ))}
            </div>
          ) : null}
          <button
            className="mujoco-run-button"
            disabled={busy || replayActive || !canRunMujoco}
            onClick={() => onRunMujoco(mujocoMode)}
            type="button"
          >
            {mujocoRunning ? "Running MuJoCo…" : cubeSelection ? `Run MuJoCo · ${mujocoMode}` : "Run MuJoCo"}
          </button>
          {/* Approving a *saved* report is a cube-rig affordance: it re-reads
              `mujoco_preview.<cube>.episode_N.json` beside the cube CSVs it was scored against.
              A workstation run has no such file -- it reports its verdict on the
              `mujoco_replay_result=` line and the gateway settles pass/fail when the process
              exits -- so the button could only ever error here. */}
          {cubeSelection ? (
            <button
              className="mujoco-pass-button"
              disabled={busy || replayActive || !mujocoPreview}
              onClick={() => onApproveMujoco(mujocoMode)}
              type="button"
            >
              {replayStatus.mujocoValidation?.status === "passed" &&
              replayStatus.mujocoValidation?.isCurrentForSelection &&
              replayStatus.mujocoValidation?.cubeMode === mujocoMode
                ? "MuJoCo passed"
                : "Pass MuJoCo check"}
            </button>
          ) : null}
        </div>
        <p className="panel-note">
          {!cubeSelection
            ? "MuJoCo computes validation metrics and streams body poses through the gateway; this browser view renders them live with Three.js/WebGL."
            : mujocoMode === "both"
              ? "Two identical FR3 models are shown in parallel with 0.90 m between bases; both trajectories remain in their own robot-base coordinates."
              : `One FR3 follows state_action.${mujocoMode}.csv from the selected dataset and episode, then the browser renders the qpos report.`}
        </p>
        <ReplayTransport
          playing={playing}
          onToggle={togglePlay}
          currentFrame={currentFrame}
          timestamp={frame?.timestamp}
        />
        <MujocoReplayViewer preview={mujocoPreview} currentFrame={currentFrame} />
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
