export type ServiceState = "offline" | "idle" | "running" | "warning" | "error";

export type DeploymentProfile = {
  profile: "thor" | "workstation";
  label: string;
  capabilities: string[];
  defaultRoute: string;
};

export type DeviceKind = "camera" | "tactile" | "handheld_gripper" | "box_collection" | "robot" | "teleoperator";

export type DeviceStatus = {
  id: string;
  kind: DeviceKind;
  label: string;
  state: ServiceState;
  fps: number;
  latencyMs: number;
  detail: string;
  config?: Record<string, unknown>;
};

export type BoxPreviewPayload = {
  active: boolean;
  deviceId: string;
  updatedAt?: number;
  staleS?: number | null;
  receivedAtS?: number | null;
  receivedWallTimeS?: number | null;
  sensor?: Record<string, unknown> | null;
  sensors?: Record<string, Record<string, unknown>>;
  status?: Record<string, unknown>;
};

export type BoxCaliLogLine = {
  ts: number;
  line: string;
  done: boolean;
};

export type BoxCaliLog = {
  running: boolean;
  lines: BoxCaliLogLine[];
};

export type GatewayStatus = {
  configPath: string;
  pid: number | null;
  state: "mock" | "online" | "offline";
  processElapsedS: number | null;
  datasetsRoot?: string;
};

export type HardwareSyncStatus = {
  enabled: boolean;
  fps: number;
  trigMode: number;
  pwmChip: string;
  pwmId: number;
};

export type CameraDefaults = {
  codec: string;
  bitrateKbps: number;
  width: number;
  height: number;
  pipeline: string;
  exposureUs: number;
  gain: number;
  iframeInterval: number;
  container: string;
};

export type ConfigSummary = {
  configPath: string;
  repoId: string;
  root: string;
  fps: number;
  episodeTimeS: number;
  targetFrames: number;
  numEpisodes: number | "unlimited";
  video: boolean;
  streamingEncoding: boolean;
  vcodec: string;
  softSync: boolean;
  rerun: {
    displayData: boolean;
    savePath: string;
  };
  recorderScript?: string;
  rigType?: "gmsl2" | "handheld";
  hardwareSync?: HardwareSyncStatus;
  cameraDefaults?: CameraDefaults;
};

export type RecordingStatus = {
  state: "idle" | "connecting" | "armed" | "recording" | "review" | "saving" | "discarding" | "error";
  datasetRoot: string;
  repoId: string;
  episodeIndex: number;
  savedEpisodes: number;
  frameIndex: number;
  targetFrames: number;
  queueDepth: number;
  message: string;
  pid?: number | null;
  lastOutput?: string;
  // Backend-side ring buffer of recent recorder stdout lines. Bounded at
  // the gateway (default 300). The frontend renders these directly so
  // rapid bursts (Phase 1 spawn × 11, parallel retry, etc.) don't get
  // collapsed into the last line that happened to land at poll time.
  recentOutput?: string[];
  // Workstation profile: which robot the recorder is driving.
  backend?: RecordingBackend;
  // Verdict of the per-episode capture-timestamp audit. "unknown" until the first episode is
  // saved; "unavailable" when the audit itself could not run.
  syncStatus?: "unknown" | "pass" | "fail" | "unavailable";
  syncSummary?: string;
  syncReportPath?: string;
  syncWarnings?: string[];
};

export type RecordingBackend = "real" | "sim";

export type TeleopCameraView = {
  id: string;
  label: string;
  source: string;
  fps: number;
  deviceId?: string;
};

export type TeleopStatus = {
  state: "idle" | "running" | "error" | "starting" | "stopped";
  backend: "mujoco" | "real";
  inputDevice: "spacemouse" | string;
  robotModel: string;
  urdfPath: string;
  simXmlPath: string;
  targetFrameName: string;
  pid?: number | null;
  message: string;
  lastOutput?: string;
  command?: string[];
  realRobotReady?: boolean;
  cameraViews?: TeleopCameraView[];
};

export type ReplayStatus = {
  state: "idle" | "preflight" | "armed" | "dry_run" | "sim_replay" | "replaying" | "paused" | "aborted" | "complete";
  dataset: string;
  episode: number;
  frameIndex: number;
  totalFrames: number;
  fps: number;
  trackingErrorMm: number;
  safety: "locked" | "ready" | "active" | "fault";
  message: string;
  datasetRoot?: string;
  datasetKind?: "recorded" | "exported";
  sourcePath?: string;
  dataStatus?: "loaded" | "missing" | "unfinalized" | "unreadable" | "empty";
  trajectoryKind?: "pose" | "gripper_width" | "none";
  totalEpisodes?: number;
  episodeOptions?: number[];
  recordedFrames?: number;
  diagnostics?: string[];
  pid?: number | null;
  lastOutput?: string;
  mujocoCubeMode?: MujocoCubeMode;
  mujocoValidation?: MujocoValidation;
  realCubeMode?: RealCubeMode;
  realRobotIp?: string;
  realEndEffectorMode?: RealEndEffectorMode;
  mujocoOverrideAccepted?: boolean;
  realReplayLog?: string[];
  // Bumped when the dataset content changes under an unchanged (root, episode)
  // selection (e.g. after deleting an episode); the inspector refetches on it.
  revision?: number;
};

export type MujocoValidation = {
  status: "not_run" | "running" | "passed" | "failed";
  datasetRoot: string;
  episode: number;
  fps: number;
  exitCode: number | null;
  completedFrames: number;
  totalFrames: number;
  avgPositionErrorMm: number | null;
  maxPositionErrorMm: number | null;
  avgRotationErrorDeg: number | null;
  maxRotationErrorDeg: number | null;
  maxPositionThresholdMm: number;
  maxRotationThresholdDeg: number;
  hasStructuredResult?: boolean;
  trajectoryContract?: {
    status?: "passed" | "failed";
    frames?: number;
    checks?: Array<Record<string, unknown>>;
    failures?: string[];
  };
  isCurrentForSelection?: boolean;
  message: string;
  updatedAt: string;
  cubeMode?: MujocoCubeMode;
};

export type MujocoCubeMode = "left" | "right" | "both";
export type RealCubeMode = Exclude<MujocoCubeMode, "both">;
export type RealEndEffectorMode = "pika_gripper_ee" | "corenetic_gripper_ee" | "fr3_ee";

export type RealSensePreviewStatus = {
  available: boolean | null;
  running: boolean;
  serial?: string;
  width?: number;
  height?: number;
  fps?: number;
  error?: string;
  updated_at?: number;
};

export type MujocoPreviewFrame = {
  frame_index: number;
  joints_rad: number[];
  target_position_m: [number, number, number];
  target_quaternion_xyzw?: [number, number, number, number];
  mujoco_position_m: [number, number, number];
};

export type MujocoPreviewRobot = {
  cube: "left" | "right";
  episode_index: number;
  base_offset_m: [number, number, number];
  frames: MujocoPreviewFrame[];
  metrics: {
    avg_position_error_mm: number;
    max_position_error_mm: number;
    avg_rotation_error_deg: number;
    max_rotation_error_deg: number;
  };
};

export type MujocoPreview = {
  schema_version: number;
  dataset_root: string;
  cube_mode: MujocoCubeMode;
  episode_index: number;
  fps: number;
  robot_spacing_m: number;
  native_video_path?: string;
  robots: Partial<Record<"left" | "right", MujocoPreviewRobot>>;
};

export type AnnotationOutcome = "unreviewed" | "success" | "failure" | "partial";

export type AnnotationQuality = "unreviewed" | "good" | "needs_review" | "bad";

export type ReviewStatus = "pending" | "approved" | "rejected";

export type SubtaskSegment = {
  id: string;
  startFrame: number;
  endFrame: number;
  description: string;
};

export type EpisodeAnnotation = {
  datasetRoot: string;
  episode: number;
  taskPrompt: string;
  outcome: AnnotationOutcome;
  quality: AnnotationQuality;
  includeInTraining: boolean;
  tags: string[];
  notes: string;
  annotator: string;
  updatedAt: string;
  source: "dataset" | "manual" | "default";
  segments: SubtaskSegment[];
  reviewStatus: ReviewStatus;
  reviewComment: string;
};

export type TaskStatus = "pending" | "in_progress" | "completed" | "paused";

export type CollectionTask = {
  id: string;
  name: string;
  description: string;
  targetEpisodes: number;
  completedEpisodes: number;
  status: TaskStatus;
  assignee: string;
  datasetRepoId: string;
  tags: string[];
  createdAt: string;
  updatedAt: string;
};

export type RecordedDataset = {
  path: string;
  name: string;
  datasetKind?: "recorded" | "exported";
  updatedAt: string;
  updatedAtMs: number;
  totalEpisodes: number;
  totalFrames: number;
  dataStatus: "loaded" | "missing" | "unfinalized" | "unreadable" | "empty";
  sourcePath: string;
  isLatest: boolean;
};

export type TrajectoryPoint = {
  frame: number;
  x: number;
  y: number;
  z: number;
  gripperWidthMm: number;
  skewMs: number;
  event?: "save" | "discard" | "gap" | "timeout";
};

export type EventLogItem = {
  id: string;
  time: string;
  level: "info" | "warn" | "error";
  message: string;
};

export type DatasetExportStatus = {
  state: "idle" | "exporting" | "complete" | "error";
  target: "lerobot_v3";
  datasetRoot: string;
  outputPath: string;
  selectedEpisodes: number;
  totalFrames: number;
  includeRaw: boolean;
  includeDebug: boolean;
  includeTraining: boolean;
  message: string;
  manifest: string[];
  taskId?: string;
};

export type ProcessingStatus =
  | "pose_missing"
  | "queued"
  | "running"
  | "pose_ready"
  | "qc_pass"
  | "qc_failed"
  | "error";

export type OnlineSyncEpisodeSummary = {
  episode: number;
  present: boolean;
  ok: boolean;
  actualFrames: number | null;
  frameCountByCamera: Record<string, number>;
  maxSofDeltaMs: number | null;
  failure: string;
};

export type OnlineSyncSummary = {
  status: "pass" | "fail" | "missing";
  message: string;
  present: number;
  missing: number;
  ok: number;
  failed: number;
  totalEpisodes: number;
  actualFrames: number;
  maxSofDeltaMs: number | null;
  frameCountMismatch: number;
  failureReasons: string[];
  episodes: OnlineSyncEpisodeSummary[];
};

export type CalibrationCamera = {
  id: string;
  reprojectionMm: number;
  baselineMm: number;
  status: "pass" | "warn" | "fail";
};

export type CalibrationStatus = {
  state: "idle" | "running" | "complete" | "failed";
  pattern: string;
  lastRunAt: string;
  message: string;
  cameras: CalibrationCamera[];
  outputPath: string;
};

export type EePose = {
  x: number;
  y: number;
  z: number;
  qx: number;
  qy: number;
  qz: number;
  qw: number;
  gripper?: number | null;
};

export type TouchPadFrame = {
  timestamp?: number;
  tRelS?: number;
  fx?: number[];
  fy?: number[];
  fz: number[];
  maxFz?: number;
  activePoints?: number;
};

export type ForceVector = {
  x: number;
  y: number;
  z: number;
  magnitude?: number;
};

export type ReplayTimelineFrame = {
  frame: number;
  timestamp: number;
  state: number[];
  action: number[];
  eePose?: Partial<EePose>;
  touch?: Record<string, TouchPadFrame | undefined>;
  forceVector?: ForceVector;
  cubePoses?: Record<string, Partial<EePose>>;
  videoOverlays?: Record<string, CubeVideoOverlay[]>;
};

export type CubeVideoOverlay = {
  cubeName: string;
  color: string;
  corners: Array<[number, number] | null>;
  axes: {
    origin: [number, number] | null;
    x: [number, number] | null;
    y: [number, number] | null;
    z: [number, number] | null;
  };
  label: [number, number] | null;
  detected: number;
  numMarkers: number;
  rmsePx: number | null;
  usedForFusion: boolean;
};

export type ReplayTimeline = {
  datasetRoot: string;
  datasetKind?: "recorded" | "exported";
  name: string;
  episode: number;
  totalFrames: number;
  fps: number;
  stateNames: string[];
  actionNames: string[];
  cubePoseNames?: string[];
  cameraKeys: string[];
  videoTemplate: string;
  videoChunkIndex: number;
  videoFileIndex: number;
  frames: ReplayTimelineFrame[];
  sourcePath: string;
  videoWarmupS?: number;
  cameraVideoOffsetsS?: Record<string, number>;
  error?: string;
};

export type ProcessingItem = {
  path: string;
  name: string;
  status: ProcessingStatus;
  trajectoryVersion: string | null;
  qcSummary: string;
  message: string;
  updatedAt: string;
  totalEpisodes: number;
  totalFrames: number;
  validFramesPct: number | null;
  logTail: string[];
  onlineSync?: OnlineSyncSummary | null;
  qcChecks?: Array<{
    name: string;
    status: "pass" | "warn" | "fail";
    message: string;
    details?: Record<string, unknown>;
  }>;
  ikEvaluation?: {
    status: "pass" | "warn" | "fail" | "skipped";
    cubes: Array<{
      cube: string;
      status: "pass" | "warn" | "fail" | "skipped";
      message: string;
      reachableRatio?: number;
      numTargets?: number;
      numUnreachableTargets?: number;
      numUnreachableTrajectories?: number;
      reachableEpisodeIndices?: number[];
      unreachableEpisodeIndices?: number[];
      plotAvailable?: boolean;
      episodes?: Array<{
        episodeIndex: number;
        status: "reachable" | "unreachable";
        label: string;
        numTargets: number;
        numReachable: number;
        numUnreachable: number;
        reachableRatio: number;
        unreachableDurationS: number;
        maxConsecutiveUnreachableTimesteps: number;
        maxPositionErrorMm: number;
        maxOrientationErrorDeg: number;
      }>;
    }>;
    message: string;
  } | null;
};
