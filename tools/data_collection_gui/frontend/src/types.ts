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

export type MarkerTcpSample = {
  id: string;
  side: "left" | "right" | string;
  boxId?: string;
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
  solvePath?: string;
  solveSummaryPath?: string;
  pivotReportPath?: string;
  trackingRunPath?: string;
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

export type DatasetKind = "recorded" | "exported" | "training_view";

export type RecordedDataset = {
  path: string;
  name: string;
  datasetKind?: DatasetKind;
  updatedAt: string;
  updatedAtMs: number;
  totalEpisodes: number;
  totalFrames: number;
  dataStatus: "loaded" | "missing" | "unfinalized" | "unreadable" | "empty";
  sourcePath: string;
  isLatest: boolean;
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
  // Bundle-adjustment reprojection residual, in pixels. Previously a fabricated
  // millimetre figure from the mock; the real solve reports pixels, and there is
  // no honest conversion without a range.
  reprojectionPx: number;
  status: "pass" | "warn" | "fail";
  /** Fraction of the frame radius the board reached, when intrinsics were re-fitted. */
  coverage?: number;
  /** Why the intrinsics for this camera are suspect, if they are. */
  intrinsicsNote?: string;
};

/** How far a running solve has got. Absent on gateways older than this field. */
export type CalibrationProgress = {
  /** 1-based; 0 when nothing is running. */
  stepIndex: number;
  stepCount: number;
  label: string;
  done: number;
  /** 0 means this step reports no unit of its own, so the bar cannot advance. */
  total: number;
  /** Overall, 0..1, weighted across the steps. */
  fraction: number;
  detail: string;
  startedAt: number;
  /** Computed on the gateway: the rig's clock is not the browser's. */
  elapsedS: number;
  /** 0 = no basis to extrapolate from yet. */
  etaS: number;
};

/** The capture the next solve will read, plus everything else it could read. */
export type CalibrationSolve = {
  datasetRoot: string;
  datasetName: string;
  episodes: number;
  /** Who chose it: an explicit pick, the guided session, or the fallback scan. */
  source: "manual" | "session" | "auto" | "missing" | "none" | string;
  candidates: { path: string; name: string; episodes: number; updatedAt: string }[];
  /** The capture intrinsics would be re-fitted from; empty when none is chosen. */
  intrinsicsDatasetRoot?: string;
  intrinsicsDatasetName?: string;
  intrinsicsEpisodes?: number;
  /** The production intrinsics run reused when they are not re-fitted. */
  intrinsicsRun?: string;
};

export type CalibrationStatus = {
  state: "idle" | "running" | "complete" | "failed";
  pattern: string;
  lastRunAt: string;
  message: string;
  cameras: CalibrationCamera[];
  outputPath: string;
  progress?: CalibrationProgress;
  solve?: CalibrationSolve;
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
  /**
   * Pad geometry this frame came from ("m2020", "paxini_l5325", ...). The BOX
   * SDK carries every pad in one fixed 239-slot array, so array length alone
   * cannot identify the pad; the gateway resolves and sends it explicitly.
   */
  model?: string;
  points?: number;
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
  markerTcpCalibrationPath?: string;
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
  // Machine-readable "why it could not be judged", so the summary can name the
  // real cause instead of guessing one from the overall verdict.
  cause?: string;
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
  // Cameras present now that the baseline never saw: nothing to compare, so
  // they carry no verdict -- but leaving them off the report entirely would let
  // a rig whose camera set changed read as fully checked.
  cameras_without_baseline?: string[];
  // Attached by the gateway, not the analysis: cameras whose frame could not be
  // grabbed at all. Without it "no current frame" has no explanation.
  failed_captures?: { camera: string; reason: string }[];
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

// --- canonical world frame (roadmap 2.4) ------------------------------------
//
// The world is not re-derived from each calibration; it is frozen once, and
// every later solve is registered back onto it by the cameras whose mutual
// geometry did not change. `world_frame_id` is the contract: two recordings are
// comparable in absolute terms only when they carry the same one.
export type WorldContinuityState = "CONTINUOUS" | "RECONNECTED" | "BROKEN";

export type WorldPairDelta = {
  a: string;
  b: string;
  translation_mm: number;
  rotation_deg: number;
  consistent: boolean;
};

export type WorldConsensus = {
  stable_cameras: string[];
  moved_cameras: string[];
  new_cameras: string[];
  missing_cameras: string[];
  // Two equal-size clusters mean "half the rig moved together" is
  // indistinguishable from "half of it stayed" -- a human has to say which.
  ambiguous: boolean;
  alternative_clusters: string[][];
  thresholds: { translation_mm: number; rotation_deg: number };
  pairs: WorldPairDelta[];
};

export type WorldAlignment = {
  cameras_used: string[];
  translation_residual_mm: Record<string, number>;
  rotation_residual_deg: Record<string, number>;
  rms_translation_mm: number;
  max_translation_mm: number;
  rms_rotation_deg: number;
  max_rotation_deg: number;
  sigma_world_translation_mm: number;
  sigma_world_rotation_deg: number;
  method?: string;
};

// The one motion camera consensus cannot see: the whole rig carried as one
// leaves every pairwise transform unchanged. Reported as an explicit blind spot
// when no independent datum was observed, never as silence.
export type WorldCommonMode = {
  observable: boolean;
  note?: string;
  translation_mm?: number;
  rotation_deg?: number;
  drifted?: boolean;
};

export type WorldRegistration = {
  generated_utc: string;
  world_continuity_state: WorldContinuityState;
  world_frame_id: string;
  parent_world_frame_id?: string | null;
  reference_world_frame_id: string;
  calibration_id: string;
  reason: string;
  guidance: string;
  auto_declarable: boolean;
  committed: boolean;
  min_stable_cameras: number;
  consensus: WorldConsensus;
  alignment: WorldAlignment | null;
  common_mode: WorldCommonMode;
  session?: { source: string; gauge: string; solver: string };
};

export type WorldReferenceSummary = {
  exists: boolean;
  world_frame_id?: string;
  created_utc?: string;
  calibration_id?: string;
  definition?: string;
  cameras?: string[];
  revisions?: {
    utc?: string;
    reason?: string;
    state?: string;
    cameras_replaced?: string[];
    stable_cameras?: string[];
  }[];
};

// Which evidence chose the cameras that define the frame. The self-check
// resolves ~1.7 mm at 1 m and the geometric consensus about a centimetre, so
// when the finer measurement is available it is the one that should decide —
// but which one was used must never be invisible.
export type WorldStableSource = {
  origin: "rig_check" | "operator" | "geometry";
  cameras?: string[];
  moved?: string[];
  generatedUtc?: string;
  rigCheckOverall?: string;
  reason?: string;
};

export type WorldFrameResponse = {
  ok: boolean;
  error?: string;
  output?: string;
  reference: WorldReferenceSummary;
  registration: WorldRegistration | null;
  stableSource?: WorldStableSource;
  graph: { worlds: number; edges: number; nodes: { world_frame_id: string; parent_world_frame_id?: string | null }[] };
  currentBundle?: string;
  extrinsicsRun?: string;
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
  /** Seconds each sweep records before the recorder ends and saves it. */
  episodeTimeS: number;
  recorderState: string;
  steps: CalibrationSessionStep[];
};

// --- production intrinsics coverage -----------------------------------------
//
// How much of each camera's frame radius the calibration board actually reached.
// It is reported separately from any reprojection score because the two cannot
// substitute for each other: held-out RMSE is measured where the board went, so
// a lens whose outer ring was never sampled scores exactly as well as one that
// was covered everywhere -- the distortion model just extrapolates out there
// with no data to contradict it.
export type IntrinsicsCoverageCamera = {
  camera: string;
  serial?: string;
  model?: string;
  // Absent when the intrinsics did not come from a metrology self-calibration
  // (a vendor file carries no such record). Absent is not "fine".
  coverage?: number | null;
  // Degrees between the model's radial fold and its own frame corner. null
  // means it never folds; <= 0 means part of the image has no unique ray.
  foldMarginDeg?: number | null;
  foldsInsideFrame?: boolean;
  framesUsed?: number;
  heldoutRmsePx?: number | null;
};

export type IntrinsicsCoverageResponse = {
  ok: boolean;
  error?: string;
  run: string;
  source?: string;
  coverageTarget: number;
  foldMarginWarnDeg: number;
  cameras: IntrinsicsCoverageCamera[];
};
