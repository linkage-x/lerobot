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
  rigType?: "gmsl2" | "handheld" | "fr3";
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

export type MarkerTcpSample = {
  id: string;
  side: "left" | "right" | string;
  condition: string;
  source: "recording" | "static_transform" | string;
  status: "pending" | "recording" | "saved" | "discarded" | "registered" | string;
  datasetRoot: string;
  episodeIndex: number;
  staticTransformPath: string;
  note: string;
  createdAt: string;
};

export type MarkerTcpSession = {
  active: boolean;
  sessionName: string;
  sessionRoot: string;
  stage: "idle" | "capture" | "reporting" | "done" | "failed" | string;
  samples: MarkerTcpSample[];
  pendingSampleId: string;
  message: string;
  reportPath: string;
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

/** The SpaceMouse gain the teleoperator applies to one axis of the 6D command. */
export type TeleopGainField =
  | "translation_scale"
  | "rotation_scale"
  | "scale_x"
  | "scale_y"
  | "scale_z"
  | "scale_wx"
  | "scale_wy"
  | "scale_wz";

/**
 * `null` means "not set" and is not the same as `0`: an unset per-axis gain falls back to the
 * matching global gain, while a `0` disables that axis outright.
 */
export type TeleopGainValues = Partial<Record<TeleopGainField, number | null>>;

export type TeleopGains = {
  /** What the next session will use: config defaults with any operator override applied. */
  values: TeleopGainValues;
  /** What the recorder YAML asks for; the Reset button returns here. */
  configDefaults: TeleopGainValues;
  /** The sim teleop script's own flag defaults, which the YAML does not feed. */
  simDefaults: TeleopGainValues;
  /**
   * Per-axis factors the teleoperator applies to a *global* gain (SpaceMouseTeleopConfig's
   * TRANSLATION_/ROTATION_AXIS_CALIBRATION). An unset z therefore runs at 59% of
   * `translation_scale`, and an explicitly set axis skips the factor entirely.
   */
  axisCalibration: Partial<Record<TeleopGainField, number>>;
  overridden: TeleopGainField[];
  absMax: number;
};

export type ReplayStatus = {
  state: "idle" | "preflight" | "armed" | "sim_replay" | "replaying" | "paused" | "aborted" | "complete";
  dataset: string;
  episode: number;
  frameIndex: number;
  totalFrames: number;
  fps: number;
  trackingErrorMm: number;
  safety: "locked" | "ready" | "active" | "fault";
  message: string;
  datasetRoot?: string;
  datasetKind?: DatasetKind;
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
  // The tool frame the gateway will build the replay command against, from
  // `robot.target_frame_name`. Shown rather than assumed: replaying a dataset recorded in
  // the other tool frame is a silent 411 mm offset, not an error.
  targetFrameName?: string;
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

export type RealSensePreviewCameraStatus = {
  cameraKey: string;
  configKey?: string;
  available: boolean | null;
  running: boolean;
  serial?: string;
  width?: number;
  height?: number;
  fps?: number;
  error?: string;
  updated_at?: number;
};

export type RealSensePreviewStatus = {
  available: boolean | null;
  running: boolean;
  serial?: string;
  width?: number;
  height?: number;
  fps?: number;
  error?: string;
  updated_at?: number;
  cameras?: RealSensePreviewCameraStatus[];
};

export type MujocoPreviewBodyPose = {
  position_m: [number, number, number];
  quaternion_xyzw: [number, number, number, number];
};

export type MujocoPreviewFrame = {
  frame_index: number;
  qpos?: number[];
  joints_rad?: number[];
  gripper?: number;
  target_position_m: [number, number, number];
  target_quaternion_xyzw?: [number, number, number, number];
  mujoco_position_m?: [number, number, number];
  actual_position_m?: [number, number, number];
  actual_quaternion_xyzw?: [number, number, number, number];
  body_poses?: Record<string, MujocoPreviewBodyPose>;
  target_frame_name?: string;
  position_error_mm?: number;
  rotation_error_deg?: number;
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
  frames?: MujocoPreviewFrame[];
  streaming?: boolean;
  stream_frame_count?: number;
  frame_source?: string;
  target_frame_name?: string;
  action_source?: string;
  model?: {
    renderer?: "three-webgl" | string;
    kinematics_path?: string;
  };
  robots: Partial<Record<"left" | "right", MujocoPreviewRobot>>;
};

/** One step of a running rollout, as the runtime publishes it. Shares the replay viewer's frame
 *  shape so the same Three.js component draws both; `source` is the addition, and it is the
 *  point: it says whether the policy or the operator produced the command behind this pose. */
export type RolloutLiveFrame = MujocoPreviewFrame & {
  source?: string;
  status?: string;
  rollout_index?: number;
};

export type RolloutLiveFrames = {
  /** Sequence of the newest frame the gateway holds. Never restarts, so "nothing new" and
   *  "you fell behind" are different answers rather than the same one. */
  seq: number;
  frames: RolloutLiveFrame[];
  rolloutIndex: number;
  running: boolean;
  /** The buffer rolled past what this client last saw; what follows is not a continuation. */
  dropped: boolean;
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

export type DatasetKind = "recorded" | "exported" | "training_view";

export type DatasetCameraFeature = {
  key: string;
  width: number;
  height: number;
};

/** What the Camera Crop picker needs to address one still frame of a recording. */
export type DatasetFramePreview = {
  path: string;
  name: string;
  /** Capture rate, used to turn a frame index into a seek timestamp. 0 when unknown. */
  fps: number;
  cameras: DatasetCameraFeature[];
  /** Per-episode frame counts. v3 packs several episodes into one video, so the count is
   * the episode's own length, not the file's. */
  episodes: { episode: number; frames: number }[];
};

export type RecordedDataset = {
  path: string;
  name: string;
  datasetKind?: DatasetKind;
  updatedAt: string;
  updatedAtMs: number;
  totalEpisodes: number;
  totalFrames: number;
  /** Capture rate. 0 when the dataset has no readable info.json. */
  fps?: number;
  dataStatus: "loaded" | "missing" | "unfinalized" | "unreadable" | "empty";
  sourcePath: string;
  isLatest: boolean;
  cameraFeatures?: DatasetCameraFeature[];
  /** Episodes marked includeInTraining=false in Episode Replay; the view build drops these. */
  excludedEpisodes?: number[];
  /** The language instruction in this dataset's tasks.parquet -- what a VLA would train on. */
  taskPrompt?: string;
  /** Training views only: the recording this view re-expresses, and in which action contract. */
  viewOf?: string;
  viewOfName?: string;
  actionContract?: string;
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
  /** "lerobot_v3" for a Thor consolidation; the action contract for a workstation training view. */
  target: string;
  datasetRoot: string;
  /** Every recording in the running build; `datasetRoot` is the first of them. */
  datasetRoots?: string[];
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
  // QC ran and raised warnings but no failures. Distinct from pose_ready ("QC has not run") so
  // that a warning is visible instead of quietly removing the dataset from Dataset Export.
  | "qc_warn"
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
  // Bundle-adjustment reprojection residual, in pixels. Previously a fabricated
  // millimetre figure from the mock; the real solve reports pixels, and there is
  // no honest conversion without a range.
  reprojectionPx: number;
  status: "pass" | "warn" | "fail";
};

export type CalibrationStatus = {
  state: "idle" | "running" | "complete" | "failed";
  pattern: string;
  lastRunAt: string;
  message: string;
  cameras: CalibrationCamera[];
  outputPath: string;
  // Which calibration runs production is pointed at, read from the tracking
  // config rather than tracked separately so the two cannot drift.
  intrinsicsRun?: string;
  extrinsicsRun?: string;
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

export type CameraControlsMetadata = {
  schema_version: number;
  captured_at: string;
  backend?: string;
  source?: string;
  cameras: Record<string, {
    type?: string;
    status?: string;
    message?: string;
    requested?: Record<string, unknown>;
    effective?: {
      device?: Record<string, unknown>;
      stream?: Record<string, unknown>;
      controls?: Record<string, unknown>;
      unsupported_controls?: string[];
    };
  }>;
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
  datasetKind?: DatasetKind;
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
  cameraControls?: CameraControlsMetadata | null;
  error?: string;
};

/** Verdict of meta/fr3_sync_report.json, split the way the report judges it. */
export type TimestampSyncSummary = {
  status: string;
  clockSemantics: string;
  totalFrames: number;
  budgetsMs: { within_group?: number | null; residual?: number | null; bias?: number | null };
  groupSkewP95Ms: number | null;
  groupSkewOverBudgetFrames: number;
  residualSkewP95Ms: number | null;
  // null when no sensor rate was available to derive the budget from: measured, not judged.
  residualSkewOverBudgetFrames: number | null;
  gridLagOverBudgetFrames: number;
  // The all-device spread. Reported because it is real, excluded from the verdict because on
  // this rig it is dominated by the cameras' constant offset from the arm read.
  rawSkewP95Ms: number;
  biasMs: Record<string, number>;
  failures: string[];
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
  timestampSync?: TimestampSyncSummary | null;
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

// Camera self-check: did a fixed camera move since it was calibrated?
//
// Reported per camera as a view shift in pixels against a baseline frame stored
// at calibration time. The criterion is change from that baseline, never how far
// the cameras disagree with each other -- inter-camera disagreement is dominated
// by target geometry (a marker size declared 5.8 cm instead of 5.6 accounted for
// 23.5 mm of it), which no recalibration would fix.
export type RigCheckVerdict = "ok" | "suspect" | "moved" | "unknown";

// "partial" means every camera that could be checked was fine but some could
// not be checked at all; it is deliberately not folded into "ok".
export type RigCheckOverall = RigCheckVerdict | "partial" | "inconclusive";

export type RigCheckCamera = {
  status?: "measured" | "unknown";
  verdict: RigCheckVerdict;
  reason?: string;
  shift_px_median?: number;
  shift_px_p95?: number;
  inliers?: number;
  inlier_ratio?: number;
  inlier_coverage?: number;
  equivalent_rotation_deg?: number;
  equivalent_error_mm_at_working_distance?: number;
};

export type RigCheckBaseline = {
  exists: boolean;
  captured_at?: string;
  cameras?: string[];
  intrinsics_run?: string;
  extrinsics_run?: string;
};

export type RigCheckReport = {
  generated_utc: string;
  overall: RigCheckOverall;
  guidance: string;
  moved_cameras: string[];
  unchecked_cameras?: string[];
  thresholds_px: { warn: number; fail: number };
  cameras: Record<string, RigCheckCamera>;
  baseline?: RigCheckBaseline;
};

export type RigCheckResponse = {
  ok: boolean;
  error?: string;
  hint?: string;
  report: RigCheckReport | null;
  baseline?: RigCheckBaseline;
};

// Guided calibration: per-camera intrinsics sweeps, then one shared extrinsics
// sweep. They cannot be merged -- intrinsics are constrained by how much of one
// camera's frame the board reaches, extrinsics by how often several cameras see
// it at the same instant.
export type CalibrationStepKind = "intrinsics" | "extrinsics";

export type CalibrationSessionStep = {
  kind: CalibrationStepKind;
  camera: string;
  status: "pending" | "recording" | "captured" | "skipped";
  episodeIndex: number;
  note: string;
};

export type CalibrationSession = {
  active: boolean;
  stage: "idle" | "capture" | "ready" | "solving" | "done" | "failed";
  datasetName: string;
  datasetRoot: string;
  currentIndex: number;
  message: string;
  recorderState: string;
  steps: CalibrationSessionStep[];
};


// ------------------------------------------------------------------ training ---

export type TrainingHost = {
  id: string;
  label: string;
  kind: "local" | "remote";
  sshTarget: string;
  repoDir: string;
  pythonPath: string;
};

export type TrainingGpu = {
  index: number | null;
  name: string;
  memoryTotalMb: number | null;
  memoryUsedMb: number | null;
  utilizationPct: number | null;
  temperatureC: number | null;
  driverVersion: string;
};

export type TrainingMachine = {
  ok: boolean;
  error?: string;
  detail?: string[];
  hostname?: string;
  platform?: string;
  cpuCount?: number | null;
  repoRoot?: string;
  repoRootExists?: boolean;
  python?: { version: string; executable: string };
  gpus?: TrainingGpu[];
  gpuError?: string;
  torch?: {
    installed: boolean;
    version?: string;
    cudaAvailable?: boolean;
    cudaVersion?: string | null;
    deviceCount?: number;
    bf16Supported?: boolean;
    error?: string;
  };
  disk?: { path?: string; totalGb?: number; freeGb?: number; error?: string };
  modules?: Record<string, boolean>;
  moduleVersions?: Record<string, string>;
  /** `extras`: the pyproject extras that install what `missing` names, so the page can offer
   *  the install rather than only reporting the gap. Empty means "no extra needed" -- torch is
   *  a base dependency, so a bare `uv sync` is what fixes act on a machine without it. */
  policies?: Record<string, { trainable: boolean; missing: string[]; extras?: string[] }>;
  /** Training options that are not policies -- LoRA needs peft whatever policy is picked. */
  features?: Record<string, { available: boolean; missing: string[]; extras?: string[] }>;
  /** Whether the install can be run from here at all, answered about *that* machine. */
  installer?: {
    canInstall: boolean;
    reason: string;
    uvPath?: string;
    uvVersion?: string;
    venvPath?: string;
    venvExists?: boolean;
    /** The same button builds and extends; only the size and the wording differ. */
    willCreateEnvironment?: boolean;
    scriptPresent?: boolean;
  };
};

/** A `uv sync` started from the Training page, as the log modal reads it. */
export type DependencyInstall = {
  state: "idle" | "running" | "complete" | "error";
  hostId: string;
  hostLabel: string;
  extras: string[];
  command: string;
  message: string;
  pid: number | null;
  startedAt: string;
  finishedAt: string;
  logPath: string;
  lastLines: string[];
};

export type TrainingWandbStatus = {
  configured: boolean;
  keySuffix: string;
  hostId: string;
};

export type TrainingView = {
  name: string;
  root: string;
  repoId: string;
  episodes: number;
  frames: number;
  fps: number;
  actionMode: string;
  cameras: string[];
  sourceFps: Record<string, number>;
  frameStride: Record<string, number>;
  /** What the build used, from meta/il_view_manifest.json: the only record of these settings,
   *  since the crop is baked into the view's video and the page that drew it keeps nothing. */
  cameraCrops: Record<string, number[]>;
  sourceRoots: string[];
  excludedEpisodes: Record<string, number[]>;
  /** Views are rebuilt in place when a task gains a session, so the root path alone no longer
   *  identifies the frames. These say which build is currently on disk. */
  buildId: string;
  sourceDigest: string;
  modifiedAt: string;
};

/** One dataset going into a policy-ready DAgger merge, as the merge script reports it.
 *  `qc_status` keeps the script's own spelling: these objects are `asdict(SourceSummary)`
 *  forwarded verbatim, and renaming a key here would only hide which side produced it. */
export type DaggerMergeSource = {
  role: "base" | "dagger" | string;
  root: string;
  episodes: number;
  frames: number;
  fps: number;
  tasks: string[];
  qc_status?: string;
};

export type DaggerMergeCheckItem = {
  name: string;
  status: "pass" | "warn" | "fail" | string;
  message: string;
};

/** The answer to "can these be merged", from `--check-only`. The gateway adds the resolved
 *  paths it would use, so what the UI shows is what a merge would actually run on rather than
 *  what was typed into the form. */
export type DaggerMergeCheck = {
  ok: boolean;
  error?: string;
  summary?: string;
  actionMode?: string;
  fps?: number;
  baseView?: string;
  daggerRoots?: string[];
  baseEpisodes?: number[];
  outputRoot?: string;
  outputName?: string;
  repoId?: string;
  totalEpisodes?: number;
  totalFrames?: number;
  sources?: DaggerMergeSource[];
  checks?: DaggerMergeCheckItem[];
  /** Last lines of a merge command that printed no JSON at all -- a crash, not a refusal. */
  detail?: string[];
};

export type DaggerMergeRequest = {
  baseView: string;
  daggerRoots: string[];
  /** Base-view episodes to keep. Empty means every episode; a subset is how a holdout stays
   *  out of the trained set. */
  baseEpisodes: number[];
  outputName: string;
  overwrite: boolean;
  copyVideos: boolean;
};

export type TrainingRun = {
  state: "idle" | "syncing" | "starting" | "running" | "complete" | "error" | "stopped";
  hostId: string;
  hostLabel: string;
  viewName: string;
  viewRoot: string;
  policy: string;
  jobName: string;
  outputDir: string;
  step: number;
  totalSteps: number;
  loss: number | null;
  message: string;
  pid: number | null;
  startedAt: string;
  finishedAt: string;
  logPath: string;
  wandbUrl: string;
  wandbEnabled: boolean;
  lastLines: string[];
};

export type TrainingStartRequest = {
  hostId: string;
  viewName: string;
  policy: string;
  jobName: string;
  steps: number;
  batchSize: number;
  numWorkers: number;
  saveFreq: number;
  logFreq: number;
  device: string;
  useAmp: boolean;
  policyConfig: string;
  /** HF repo id or local checkpoint dir to finetune from. Empty trains from scratch. */
  pretrainedPath: string;
  /** Continue a saved optimizer/trainer state instead of starting a fresh run from weights. */
  resumeTraining: boolean;
  /** Checkpoint directory or its pretrained_model child used with resumeTraining. */
  resumeCheckpoint: string;
  /** Freeze the base weights and train a PEFT adapter instead. Needs pretrainedPath. */
  loraEnabled: boolean;
  loraR: number;
  /** Scaling numerator; the adapter's strength is alpha / r. 0 tracks the rank (scaling 1.0). */
  loraAlpha: number;
  /** Empty leaves the policy's own default target set alone, which for pi0.5 is the tuned one. */
  loraTargetModules: string;
  wandbEnabled: boolean;
  wandbProject: string;
  wandbEntity: string;
};

/** The knobs one past run can hand to the next. Every field is optional: a run recorded
 *  before a knob existed simply lacks it, and the form then keeps its current value. */
export type TrainingHistoryParams = Partial<
  Pick<
    TrainingStartRequest,
    | "policy"
    | "steps"
    | "batchSize"
    | "numWorkers"
    | "saveFreq"
    | "logFreq"
    | "device"
    | "useAmp"
    | "policyConfig"
    | "pretrainedPath"
    | "resumeTraining"
    | "resumeCheckpoint"
    | "loraEnabled"
    | "loraR"
    | "loraAlpha"
    | "loraTargetModules"
    | "wandbEnabled"
    | "wandbProject"
    | "wandbEntity"
  >
>;

/** One previously started run, newest first, as offered by /api/training/history. */
export type TrainingHistoryEntry = {
  jobName: string;
  startedAt: string;
  /** The view it trained on. Shown, never copied: the point is usually to retrain new frames. */
  viewName: string;
  hostLabel: string;
  policy: string;
  steps: number;
  params: TrainingHistoryParams;
};

// ------------------------------------------------------- checkpoints & rollout ---

/** One disagreement between a checkpoint and the rig it would drive. */
export type ContractIssue = {
  level: "ok" | "warn" | "block";
  field: string;
  message: string;
};

/** The dataset facts recovered from the view a checkpoint names. */
export type CheckpointView = {
  root?: string;
  exists?: boolean;
  fps?: number;
  episodes?: number;
  frames?: number;
  cameras?: string[];
  actionMode?: string;
  stateKeys?: string[];
  /** True when the view was found by name in this repo rather than at the path the
   *  checkpoint records — what a fetch from a training machine with a different layout
   *  produces. */
  relocated?: boolean;
};

/** The rollout settings recorded in the checkpoint's generated inference config. */
export type CheckpointContract = {
  robotIp?: string;
  targetFrameName?: string;
  gripperBackend?: string;
  gripperPort?: string;
  cameraConfig?: string;
  cameraKeys?: string[];
  policy?: string;
  safety?: {
    firstFrameMaxPosDeltaMm?: number | null;
    firstFrameMaxRotDeltaDeg?: number | null;
    maxStepPosDeltaMm?: number | null;
    maxLeashPosDeltaMm?: number | null;
    maxLeashRotDeltaDeg?: number | null;
    maxStepRotDeltaDeg?: number | null;
  };
};

export type CheckpointOutcomes = {
  success: number;
  failure: number;
  aborted: number;
  total: number;
};

export type Checkpoint = {
  id: string;
  jobName: string;
  stepLabel: string;
  step: number;
  isLast: boolean;
  /** Set when this entry is a symlink; names the numbered step whose bytes it shares. */
  aliasOf: string;
  path: string;
  pretrainedPath: string;
  policyType: string;
  chunkSize?: number | null;
  nActionSteps?: number | null;
  cameras: string[];
  totalSteps: number;
  datasetRepoId: string;
  /** Where the view actually is on this machine. */
  datasetRoot: string;
  /** The absolute path baked into the checkpoint by the machine that trained it. */
  recordedDatasetRoot: string;
  sizeBytes: number;
  modifiedAt: number;
  view: CheckpointView;
  contract: CheckpointContract;
  inferenceConfigPath: string;
  issues: ContractIssue[];
  verdict: "ok" | "warn" | "block";
  outcomes: CheckpointOutcomes | null;
  hostId: string;
  hostLabel: string;
  wandbProject?: string;
  wandbRunId?: string;
};

/** What the rig is configured as today, for comparison against a checkpoint. */
export type RigContract = {
  robotIp: string;
  targetFrameName: string;
  cameraKeys: string[];
  cameraConfigPath: string;
};

export type CheckpointListing = {
  ok: boolean;
  error?: string;
  detail?: string[];
  host: TrainingHost;
  rig: RigContract;
  checkpoints: Checkpoint[];
};

export type RolloutMode = {
  id: string;
  label: string;
  description: string;
  movesArm: boolean;
  interactive: boolean;
  /** Whether the launcher forwards the DAgger flags to this mode. Narrower than `interactive`:
   *  the runtime refuses takeover without interactive rollouts, so only `real` and `real_debug`
   *  carry it. */
  takeover?: boolean;
};

export type RolloutRun = {
  state:
    | "idle"
    | "starting"
    | "waiting"
    | "homing"
    | "resetting"
    | "rolling"
    | "complete"
    | "error"
    | "stopped";
  mode: string;
  checkpointId: string;
  checkpointPath: string;
  policy: string;
  datasetRoot: string;
  targetFrameName: string;
  robotIp: string;
  cameraKeys: string[];
  interactive: boolean;
  movesArm: boolean;
  /** Whether a SpaceMouse was opened for this rollout, i.e. whether the operator can take the
   *  arm over by moving it. Read off the takeover key the runtime binds, which it refuses to
   *  bind without a device. Reported by the runtime rather than derived from the mode. */
  takeoverAvailable?: boolean;
  /** `yes` once the runtime's takeover pre-flight has printed. It is the only value a real
   *  rollout can reach -- an undated SpaceMouse driver is refused before the arm is built -- so
   *  this is the operator's confirmation that the check ran, not a warning light. */
  daggerReportTimestamps?: string;
  /** Seconds of a still device before the policy takes the arm back. 0 means only the hold latch
   *  moves the arm between the two drivers. */
  daggerReleaseAfterS?: number | null;
  /** Where corrections are being written; empty when the operator chose to steer without
   *  recording. */
  daggerDatasetPath?: string;
  /** Correction episodes written so far this session, summed across rollouts. */
  daggerEpisodes?: number;
  /** Frames dropped past the buffer cap. Non-zero means a correction was longer than the buffer
   *  and the dataset is missing its end. */
  daggerDroppedFrames?: number;
  step: number;
  maxSteps: number;
  commandStatus: string;
  clampedSteps: number;
  leashedSteps: number;
  rolloutIndex: number;
  lastRolloutStatus: string;
  /** Whether the arm is at the pose the demonstrations started from. False from the moment a
   *  rollout begins until somebody homes it: the launcher homes once, before the runtime. */
  armAtStart: boolean;
  pid: number | null;
  message: string;
  startedAt: string;
  finishedAt: string;
  logPath: string;
  /** This launch's trace directory, one per launch. The runtime restarts its rollout numbering
   *  at 1 on every start, so a shared directory would overwrite the previous batch. */
  tracePath: string;
  previewDir: string;
  lastLines: string[];
  /** Non-zero when a finished rollout is waiting for the operator to grade it. */
  pendingOutcomeFor: number;
  /** Where the last finished rollout put the gripper. Empty until one has finished. */
  lastRolloutGeometry?: RolloutGeometry;
  /** Whether a human drove part of the last finished rollout. Empty until one has finished,
   *  and on a runtime too old to report it. */
  lastRolloutIntervention?: RolloutIntervention;
};

/** Whose hand drove the rollout that is about to be graded.
 *
 *  `intervened` false is a measurement: the runtime reported a summary and it contained no
 *  expert spans. The whole record being absent is the other case -- nobody counted -- and the
 *  page must not turn that into "the policy did it alone".
 */
export type RolloutIntervention = {
  intervened?: boolean;
  /** Steps the operator drove, out of the rollout's total. Zero when `intervened` is false. */
  expertSteps?: number;
};

/** The landing points of one rollout, measured by the runtime from its own per-step trace.
 *
 *  Every field is optional because a rollout that never closed its gripper has an approach
 *  point and no grasp point, and one that closed and never reopened has no release point. A
 *  missing field means the event did not happen, which is itself a result -- it is what
 *  separates "reached for it and did not grip" from "gripped and dropped it".
 */
export type RolloutGeometry = {
  graspXyz?: [number, number, number];
  releaseXyz?: [number, number, number];
  approachXyz?: [number, number, number];
  apexZ?: number;
  liftM?: number;
  descentM?: number;
  samples?: number;
  heldSteps?: number;
  closed?: boolean;
  /** Who was driving at the instant of each point. Absent on rollouts recorded before the
   *  runtime attributed them — which is not the same as "the policy". Rollout-level
   *  `intervened` cannot answer this: it says a human was somewhere in the rollout, not
   *  whether they were in *this* event. */
  graspBy?: EventDriver;
  releaseBy?: EventDriver;
  approachBy?: EventDriver;
};

/** The two things that can drive the arm during a rollout. */
export type EventDriver = "policy" | "expert";

/** One demonstration's grasp and release, reduced by the same rule the runtime applies live. */
export type DemoLandingPoint = {
  episode: number;
  graspXyz: [number, number, number];
  releaseXyz: [number, number, number];
  liftM: number;
  descentM: number;
};

export type RolloutLandmarks = {
  datasetRoot?: string;
  /** Measured as the mean of every demonstration's release point, not configured. */
  hole?: [number, number];
  /** The same mean release point with its height: where the demonstrations left the peg, and
   *  therefore where a scene reset has to reach to pick it back up. */
  placeXyz?: [number, number, number];
  points?: DemoLandingPoint[];
  graspRadiusM?: { min: number; max: number; mean: number };
};

/** The base-frame rectangle a map is drawing, and therefore the one its backdrop must cover. */
export type TableWindow = { minX: number; maxX: number; minY: number; maxY: number };

/** One correspondence in a camera's table calibration: where the tool was, where it appeared.
 *
 *  `u`/`v` are pixels in the camera's own still and `x`/`y` are metres in the robot base frame.
 *  The base half is measured by the robot, never typed: it comes from the runtime's report of
 *  the point it actually reached.
 */
export type TablePlanePoint = { u: number; v: number; x: number; y: number };

/** How one camera's pixels map onto the table plane, and how well.
 *
 *  `calibrated` is the only thing a map should branch on. Points can exist without a fit (fewer
 *  than four), and a fit can exist and be bad, which is what `maxResidualMm` is for: it is the
 *  distance between where the robot was and where the fit says the click was, in the units the
 *  error budget is written in.
 */
export type TableAlignment = {
  ok?: boolean;
  cameraKey: string;
  planeZ: number;
  imageWidth: number;
  imageHeight: number;
  points: TablePlanePoint[];
  imageToBase: number[][] | null;
  residualsMm: number[];
  maxResidualMm: number;
  /** Why enough points still produced no mapping — three of them in a line, most often. */
  fitError: string;
  calibrated: boolean;
  minPoints: number;
  recommendedPoints: number;
  updatedAt: string;
  probeFrame?: {
    requestId: string;
    xyz: number[];
    at: number;
    pendingRequestId: string;
    pendingXyz: number[];
    /** A still exists and belongs to the probe that was asked for. */
    ready: boolean;
    /** The arm is still on its way to the point this still is supposed to show. */
    moving: boolean;
  };
};

export type RolloutStatusPayload = {
  ok: boolean;
  rollout: RolloutRun;
  modes: RolloutMode[];
  rig: RigContract;
  trainingBusy: boolean;
};

/** One task's grading ladder, as the gateway serves it from `tools/fr3/task_ladders/`.
 *  The page never defines stages of its own: the menu and the validation have to be the
 *  same list, or the operator can pick a stage the gateway then rejects. */
export type TaskLadderStage = {
  id: string;
  ordinal: number;
  /** The vocabulary's word for this link, shared across tasks. */
  label: string;
  /** What the link means in general. */
  criterion: string;
  /** What it looks like in *this* task -- the sentence the operator matches against. */
  instance: string;
};

export type TaskLadderBlocker = { id: string; label: string; instance: string };

export type TaskLadder = {
  task: string;
  label: string;
  /** The stage that counts as success. Everything below it is a shortfall. */
  terminal: number;
  stages: TaskLadderStage[];
  blockers: TaskLadderBlocker[];
};

export type RolloutOutcomeEntry = {
  recordedAt: string;
  checkpointId: string;
  outcome: "success" | "failure" | "aborted";
  mode: string;
  steps: number;
  note: string;
  logPath: string;
  rolloutIndex?: number;
  /** Absent on rollouts recorded before the runtime reported landing points. */
  geometry?: RolloutGeometry;
  /** Whether a human drove part of this rollout, and for how many steps. Absent on records
   *  written before the runtime reported it — which is not the same as false. An assisted
   *  rollout's outcome describes the operator, not the checkpoint. */
  intervened?: boolean;
  expertSteps?: number;
  /** The ladder this rollout was graded against. Absent on rollouts graded before ladders. */
  taskLadder?: string;
  /** How far along that task's precondition chain it got. Ordinal — never average these. */
  stage?: number;
  stageId?: string;
  /** The stage that counts as success. Stored per record so an old grade still renders right. */
  terminalStage?: number;
  blocker?: string;
  /** Whether the attempt was inside what the demonstrations cover. */
  inDistribution?: boolean;
};

export type RolloutRtcMode = "auto" | "enabled" | "disabled";

export type RolloutRtcSchedule = "EXP" | "LINEAR" | "ONES" | "ZEROS";

export type RolloutRuntimeOptions = {
  /** Empty lets the runtime recover the single task prompt recorded in the dataset view. */
  taskPrompt?: string;
  /** auto enables RTC for flow policies such as pi0.5 and keeps it off for ACT. */
  rtcMode?: RolloutRtcMode;
  rtcExecutionHorizon?: number;
  rtcMaxGuidanceWeight?: number;
  rtcPrefixAttentionSchedule?: RolloutRtcSchedule;
  rtcReplanQueueSize?: number;
  /** null or undefined leaves the runtime to estimate the delay from measured inference time. */
  rtcInferenceDelaySteps?: number | null;
  /** null or undefined disables extra command EMA smoothing. */
  commandEmaAlpha?: number | null;
  /** Open a SpaceMouse and let the operator take the arm mid-rollout. Only `real` and
   *  `real_debug` accept it; anything else is refused with a reason rather than dropped. */
  daggerTakeover?: boolean;
  /** Whether those corrections become training data. False is the shakedown case -- feeling out
   *  the handoff on the real arm without adding half-meant corrections to a dataset. */
  daggerRecord?: boolean;
  /** Blank derives one per checkpoint, so an afternoon's corrections accumulate in one dataset
   *  instead of scattering across launches. */
  daggerDatasetRoot?: string;
  /** null or undefined leaves the runtime's 1 s handback. 0 turns automatic handback off. */
  daggerReleaseAfterS?: number | null;
};

/** The previous rollout's settings, as offered by /api/rollout/last-params.
 *
 *  Deliberately excludes both safety gates (`confirmMotion`, `overrideContract`): a remembered
 *  "yes" to arm motion is a gate that answers itself. The checkpoint is not a gate -- it is
 *  re-selected, and then re-checked against the rig like any other pick. */
export type RolloutLastParams = {
  mode?: string;
  /** Pre-selects the picker once the listing containing it has loaded. Absent, or naming a
   *  checkpoint that has since been deleted, leaves the page with nothing selected. */
  checkpointId?: string;
  maxSteps?: number;
  moveToStart?: boolean;
  runtimeOptions?: RolloutRuntimeOptions;
};


export type SceneResetStroke = {
  x: number;
  y: number;
  radiusM: number;
};

export type SceneResetRequest = {
  pickXyz: [number, number, number];
  targetZ: number;
  liftM: number;
  approachClearanceM?: number;
  openGripper?: number;
  closedGripper?: number;
  gripperTolerance?: number;
  graspSettleS?: number;
  returnToStart?: boolean;
  mask: { strokes: SceneResetStroke[] };
};

export type SceneResetResult = {
  ok: boolean;
  error?: string;
  sceneReset?: SceneResetRequest & { targetXyz?: [number, number, number] };
  rollout?: RolloutRun;
};

export type RolloutStartRequest = {
  mode: string;
  checkpointId: string;
  confirmMotion?: boolean;
  overrideContract?: boolean;
  maxSteps?: number;
  moveToStart?: boolean;
  runtimeOptions?: RolloutRuntimeOptions;
};
