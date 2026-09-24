export type ServiceState = "offline" | "idle" | "running" | "warning" | "error";

export type DeploymentProfile = {
  profile: "thor" | "workstation";
  label: string;
  capabilities: string[];
  defaultRoute: string;
};

export type DeviceKind = "camera" | "tactile" | "handheld_gripper" | "box_collection" | "robot" | "teleoperator" | "laser_tracker";

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
  // Whether this Connect asked for the laser tracker. A per-session choice, not
  // a property of the rig: the instrument is shared and admits one client.
  laserTracker?: boolean;
  laserTrackerState?: string;
  laserTrackerDetail?: string;
  // Model / serial / firmware as the instrument reported them.
  laserTrackerDevice?: string;
  // Whether the beam is on the SMR *and* the session homed. Start Episode is gated on it.
  laserTrackerReady?: boolean;
  // Whether this tracker session homed. Without it every range inherits a stale
  // reference (W2, 2026-09-21: locked and green throughout, 337-440 mm off).
  laserTrackerHomed?: boolean;
  /** Homed, but the beam broke since: the current lock has no absolute range. */
  laserTrackerBeamBroken?: boolean;
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
  /** This Connect had the laser tracker on: the sample doubles as an E1p capture. */
  laserTracker?: boolean;
  trackerSessionId?: string;
};

/** The last E1p run from the marker->TCP panel, as the gateway holds it. */
export type MarkerTcpTrackerCheck = {
  ok?: boolean;
  returncode?: number;
  error?: string;
  summary?: string;
  boxId?: string;
  cube?: string;
  condition?: string;
  mountId?: string;
  samples?: number;
  notes?: string[];
  stationPath?: string;
  mountFitPath?: string;
  markerTcpPath?: string;
  reportPath?: string;
  createdAt?: string;
  report?: TrackerPivotReport | null;
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
  trackerCheck?: MarkerTcpTrackerCheck;
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

// "calibration" is a board sweep recorded by the guided wizard, not a
// demonstration: it never counts towards a task's episode budget and is never
// merged into a v3 export.
export type DatasetKind = "recorded" | "exported" | "training_view" | "calibration";

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
  /** Whether re-fitting from that capture could survive its own export. */
  intrinsicsPreflight?: IntrinsicsPreflight;
};

/** Which cameras a re-fit would have to produce a lens for, which of them
 * production has no lens for today, and which of production's lenses this
 * capture does not re-fit. The last group is carried into the new run by the
 * exporter, so it is a note; the middle one still blocks, because a camera in
 * the capture that fails its fit takes the export down at the last step and
 * there is nothing in production to put in its place. */
export type IntrinsicsPreflight = {
  cameras: string[];
  production: string[];
  uncalibrated: string[];
  /** Of those, the ones an earlier solve already fitted from this unchanged
   * capture with a lens the exporter takes. They no longer block. */
  proven?: string[];
  provenReport?: string;
  /** Fitted from this capture by an earlier solve, but a lens the exporter
   * refuses (folds in the frame). Any one of them fails the whole export. */
  refusedFit?: string[];
  /** Kept from the production run because this capture never swept them. */
  carriedForward?: string[];
  blocking: boolean;
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
  /**
   * The last run the gateway *knows about*: the tracking config's value at
   * startup, overwritten in memory by whatever a solve produces.
   *
   * The comment that used to sit here said these were read from the tracking
   * config "so the two cannot drift". They do drift, and that is the whole
   * point of `production` below: nothing writes the run names back to the
   * config, so a finished solve leaves these two naming a calibration that
   * production will not load.
   */
  intrinsicsRun?: string;
  extrinsicsRun?: string;
  /** What the tracking config says right now — what production actually loads. */
  production?: CalibrationProduction;
  /** Present only when the last solve and the production pointer disagree. */
  pointerMismatch?: CalibrationPointerMismatch;
  /**
   * The mandatory review: what promoting the newest run would change. Absent
   * when production already loads the newest run, which is the ordinary case.
   */
  promotion?: CalibrationPromotionReview;
};

/** Per-camera difference between two extrinsics runs, in gauge-free terms. */
export type PromotionCameraRow = {
  camera: string;
  medianBaselineShiftMm: number;
  maxBaselineShiftMm: number;
  medianRotationDeg: number;
  maxRotationDeg: number;
};

export type PromotionWorld = {
  worldFrameId: string;
  referenceWorldFrameId: string;
  continuityState: string;
  reason: string;
  stableCameras: string[];
};

export type PromotionBlocker = { kind: string; message: string; target?: string };

export type ExtrinsicsComparison = {
  ok: boolean;
  error?: string;
  live: string;
  candidate: string;
  cameras?: PromotionCameraRow[];
  addedCameras?: string[];
  removedCameras?: string[];
  pairCount?: number;
  medianBaselineShiftMm?: number;
  medianRotationDeg?: number;
  worstPair?: { a: string; b: string; liveMm: number; candidateMm: number; shiftMm: number; rotationDeg: number };
  liveWorld?: PromotionWorld;
  candidateWorld?: PromotionWorld;
  /** Shown, never ranked on: this number picked the wrong run in August. */
  liveRmsePx?: number | null;
  candidateRmsePx?: number | null;
};

export type IntrinsicsComparison = {
  ok: boolean;
  error?: string;
  live: string;
  candidate: string;
  cameras?: string[];
  model?: string;
  mixedModels?: string[];
  trackerModel?: string;
  addedCameras?: string[];
  removedCameras?: string[];
};

export type CalibrationPromotionReview = {
  candidates: { intrinsics?: string; extrinsics?: string };
  configPath: string;
  extrinsics?: ExtrinsicsComparison;
  extrinsicsBlockers?: PromotionBlocker[];
  intrinsics?: IntrinsicsComparison;
  intrinsicsBlockers?: PromotionBlocker[];
};

export type CalibrationProduction = {
  configPath: string;
  intrinsicsRun: string;
  extrinsicsRun: string;
  /** Non-empty when the config could not be read or parsed. */
  error: string;
};

export type CalibrationPointerMismatch = {
  fields: { kind: "intrinsics" | "extrinsics"; label: string; solved: string; production: string }[];
  configPath: string;
  message: string;
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
  kind?: "cube" | "hybrid_carrier";
  polygons?: Array<{
    role: "anchor" | "facet";
    label: string;
    color: string;
    points: Array<[number, number] | null>;
  }>;
  markerIds?: number[];
  numEdgeSamples?: number;
  message?: string;
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

export type TrackingTarget = "april_cube" | "hybrid_carrier_v1";

export type TrackingDetectionSummary = {
  target: TrackingTarget;
  label: string;
  totalViews: number;
  detectedViews: number;
  detectionRatePct: number;
  overlayAvailable: boolean;
  perCamera: Array<{
    camera: string;
    streamKey: string;
    totalViews: number;
    detectedViews: number;
    detectionRatePct: number;
    medianAnchors: number | null;
    medianRmsePx: number | null;
    medianEdgeSamples: number | null;
  }>;
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
  trackingTarget?: TrackingTarget;
  detectionSummary?: TrackingDetectionSummary | null;
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

// --- hand-eye (AX = XB), the rotation half of marker rig -> TCP -------------
// Mirrors metrology/cli/hand_eye_calibration.py. `verdict.status` is the field
// to branch on, not `ok`: "not_observable", "mis_associated" and
// "no_uncertainty_estimate" are all successful *runs* that refuse to produce a
// constant, and each needs a different thing from the operator.
export type HandEyeVerdictStatus =
  | "ok"
  | "not_observable"
  | "mis_associated"
  | "insufficient_motions"
  | "solved_but_out_of_budget"
  | "no_uncertainty_estimate";

export type HandEyeReport = {
  input?: { path?: string; sha256?: string; num_poses?: number; pose_names?: string[] };
  motions?: {
    num_poses?: number;
    num_motions?: number;
    num_candidate_pairs?: number;
    pairing?: string;
    dropped?: Record<string, number>;
    worst_angle_disagreement_deg?: number;
    motion_angle_deg?: { p05?: number; p50?: number; p95?: number };
  };
  observability?: {
    ok?: boolean;
    rotation_axis_rank_ratio?: number;
    translation_rank_ratio?: number;
    reasons?: string[];
  };
  solution?: {
    T_flange_rig?: number[][];
    rotation_vector_deg?: number[];
    translation_mm?: number[];
    num_motions_used?: number;
    num_motions_rejected?: number;
    residual_rotation_deg?: { p50: number; p95: number; max: number };
    residual_translation_mm?: { p50: number; p95: number; max: number };
  };
  holdout?: {
    num_folds?: number;
    note?: string;
    solution_shift_rotation_deg?: { p50: number; p95: number; max: number };
  };
  bootstrap?: { num_resamples?: number; sigma_deg?: number | null; note?: string };
  rotation_sigma_deg?: number | null;
  lever_equivalent_mm?: number;
  lever_mm?: number;
  against_budget?: {
    target_deg: number;
    acceptable_deg: number;
    meets_target: boolean;
    meets_acceptable: boolean;
    replaces_declared_deg: number;
    replaces_declared_mm: number;
  };
  T_box_rig?: { status?: string; why?: string; note?: string; matrix?: number[][] };
  production?: {
    wired_in?: boolean;
    note?: string;
    marker_rig_to_tcp_patch?: { rotation_source: string; rotation_sigma_deg: number } | null;
  };
  verdict?: { status?: HandEyeVerdictStatus; why?: string; note?: string };
  validated?: boolean;
};

export type HandEyeSolveResponse = {
  ok: boolean;
  returncode?: number;
  error?: string;
  verdict?: { status?: HandEyeVerdictStatus; why?: string; note?: string };
  report: HandEyeReport | null;
  reportPath?: string;
  stdout?: string;
  stderr?: string;
};

export type HandEyePlanRow = {
  num_poses: number;
  num_trials: number;
  rotation_error_p50_deg?: number;
  rotation_error_p95_deg?: number;
  lever_equivalent_p95_mm?: number;
  meets_target?: boolean;
  meets_acceptable?: boolean;
};

export type HandEyePlanResponse = {
  ok: boolean;
  error?: string;
  plan: {
    assumptions?: Record<string, number>;
    lever_mm?: number;
    rows?: HandEyePlanRow[];
  } | null;
  planPath?: string;
  stdout?: string;
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

/**
 * One episode's comparison against the laser tracker, as the offline CLI left
 * it. The gateway reads the artifact and does not compute, so everything here
 * is a fact about a file on disk rather than about the current session.
 */
export type TrackerAlignmentSeries = {
  /** Seconds from the first paired frame, not from episode start. */
  t_rel_s: number[];
  /** Sidecar frame index, so the plot can align to the replay timeline. */
  frame_index: number[];
  residual_mm: number[];
  speed_m_s: number[];
  /** Camera prediction of the SMR centre, metres, in the camera world frame. */
  camera_xyz_m: Array<[number, number, number]>;
  /** Tracker measurement of the same point, mapped into the same frame. */
  tracker_xyz_m: Array<[number, number, number]>;
};

export type TrackerAlignmentSummary = {
  n_paired: number;
  n_camera_frames: number;
  coverage: number;
  residual_mm: Record<string, number | null>;
  strata: Record<string, Record<string, number | null>>;
  clock: Record<string, number>;
  registration: {
    source: string;
    certifies_space: boolean;
    rms_mm: number | null;
    scale_diagnostic: number | null;
    /**
     * Error modes the registration -- and the lever arm fitted beside it --
     * removed before the residual was computed, so the residual is silent about
     * them however small it is.
     *
     * Independence and absorption are different questions. `certifies_space`
     * says the transform came from data other than this trajectory; this says
     * which constant biases that independent fit still soaked up. A lever arm
     * fitted from parked poses is independent *and* absorbs both body-frame
     * constants, which together are the marker-to-TCP constant.
     */
    absorbed_modes?: string[];
  };
  lever_arm_mm: number;
  /**
   * Millimetres of residual one degree of orientation error would produce.
   * Zero means the residual says nothing at all about rotation, which is a
   * property of where the SMR sits, not of how good the pipeline is.
   */
  rotation_sensitivity_mm_per_deg: number;
  interp_error_mm_bound: number;
  /** Positive = the tracker leads the cameras. Null = the motion could not answer. */
  time_crosscheck_s: number | null;
  certifies_space: boolean;
};

export type TrackerAlignment =
  | { ok: true; available: false; reason: string }
  | {
      ok: true;
      available: true;
      artifact: string;
      generatedUtc: string;
      episode: number;
      target: string;
      session: Record<string, unknown>;
      summary: TrackerAlignmentSummary;
      certifiesSpace: boolean;
      registrationSource: string;
      coverage: number;
      series: TrackerAlignmentSeries;
      /** Loss-of-lock windows, seconds relative to the first paired frame. */
      dropoutsRelS: Array<[number, number]>;
      leverArmM: [number, number, number];
      minCoverage: number;
      /** Provenance of the fitted lever arm, when one was used. */
      mountFit: {
        artifact?: string | null;
        mount_id?: string | null;
        session_id?: string | null;
        station_artifact?: string | null;
        n_poses?: number | null;
        holdout_rms_mm?: number | null;
        per_pose_c_spread_mm?: number | null;
        sigma?: { c_sigma_norm_mm?: number | null } | null;
        observability?: { fixed_attitude?: boolean; rotation_span_deg?: number } | null;
      } | null;
    };

// --- Laser tracker station + SMR lever arm (metrology.tracker_mount_fit) ---
//
// The shapes mirror the solver's artifacts rather than flattening them, because
// what the panel has to render is *which* claim a fit is entitled to make, and
// that lives in the nested observability / sigma / attitude blocks.

export type TrackerMountObservability = {
  n_poses: number;
  station_frozen: boolean;
  ok: boolean;
  rotation_span_deg: number;
  fixed_attitude: boolean;
  c_gain_min: number;
  c_gain_max: number;
  c_gain_min_equiv_deg: number;
  c_sigma_amplification: number;
  planarity: number | null;
  extent_m: number | null;
  reasons: string[];
};

export type TrackerMountSigma = {
  num_resamples: number;
  c_sigma_mm: number[] | null;
  c_sigma_norm_mm: number | null;
  rotation_sigma_deg: number | null;
  translation_sigma_mm: number | null;
  note: string;
};

/** Structure of the per-pose lever arms in attitude -- the mount-rigidity check. */
export type TrackerAttitudeDependence = {
  n_poses: number;
  explained_frac: number;
  null_explained_frac: number;
  slope_mm_per_deg: number;
  rotation_span_deg: number;
  structured: boolean;
};

export type TrackerStationSession = {
  session_id: string;
  mount_id: string;
  c_m: number[];
  lever_arm_mm: number;
  attitude: TrackerAttitudeDependence | null;
};

export type TrackerStationReport = {
  T_world_tracker: number[][];
  world_frame_id: string;
  tracker_station_id: string;
  sessions: TrackerStationSession[];
  n_poses_total: number;
  iterations: number;
  rms_mm: number;
  registration_rms_mm: number;
  leave_one_out_max_mm: number;
  attitude_structured: boolean | null;
  scale_diagnostic: number;
  scale_error_ppm: number;
  scale_applied: boolean;
  observability: TrackerMountObservability;
  sigma: TrackerMountSigma;
  certifies_marker_to_tcp: boolean;
};

export type TrackerMountReport = {
  mount_id: string;
  session_id: string;
  c_m: number[];
  lever_arm_mm: number;
  rotation_sensitivity_mm_per_deg: number;
  T_world_tracker: number[][];
  station_fitted: boolean;
  n_poses: number;
  rms_mm: number;
  max_mm: number;
  holdout_rms_mm: number | null;
  per_pose_c_spread_mm: number;
  attitude: TrackerAttitudeDependence | null;
  observability: TrackerMountObservability;
  sigma: TrackerMountSigma;
  absorbed_modes: string[];
  certifies: boolean;
  certifies_marker_to_tcp: boolean;
};

export type TrackerMountArtifact = {
  path: string;
  name: string;
  modifiedUnixS: number;
  report: TrackerStationReport | TrackerMountReport;
};

export type TrackerMountListResponse = {
  ok: boolean;
  root?: string;
  stations?: TrackerMountArtifact[];
  mounts?: TrackerMountArtifact[];
  /** E1p artifacts that carry the SMR->TCP vector a TCP comparison needs. */
  pivots?: TrackerMountArtifact[];
  error?: string;
};

export type TrackerMountCaptureRow = {
  session: string;
  dataset: string;
  episode: string;
  mountId: string;
  sessionId?: string;
};

export type TrackerMountSolveResponse = {
  ok: boolean;
  returncode?: number;
  kind?: "station" | "lever_arm" | "pivot";
  report?: TrackerStationReport | TrackerMountReport | TrackerPivotReport | null;
  reportPath?: string;
  stationPath?: string;
  markerTcpPath?: string;
  /** Segments left out (no dwell / no pivot data), listed whether or not the fit then solved. */
  skipped?: { episode: number; why: string }[];
  summary?: string;
  stdout?: string;
  stderr?: string;
  error?: string;
};

// --- GT comparison: camera trajectory vs tracker (validate_against_tracker) ---

export type TrackerErrorStats = {
  count: number;
  mean_mm?: number;
  rms_mm?: number;
  p50_mm?: number;
  p95_mm?: number;
  max_mm?: number;
};

export type TrackerValidateSummary = {
  n_paired: number;
  n_camera_frames: number;
  coverage: number;
  residual_mm: TrackerErrorStats;
  /** Split by speed, because that is the axis a timing error lives on. */
  strata: Record<string, TrackerErrorStats>;
  clock: Record<string, unknown>;
  registration: { source?: string; absorbed_modes?: string[]; certifies_space?: boolean };
  lever_arm_mm: number[];
  rotation_sensitivity_mm_per_deg: number;
  interp_error_mm_bound: number;
  time_crosscheck_s: number | null;
  certifies_space: boolean;
};

export type TrackerValidateReport = {
  dataset: string;
  episode: number;
  target: string;
  sidecar: string;
  summary: TrackerValidateSummary;
  min_coverage: number;
  lever_arm_m: number[];
  mount_fit: Record<string, unknown> | null;
  /** "tcp": camera TCP - tracker TCP (c_TCP error included); "smr_centre": the lever arm absorbed the constants. */
  compared_point?: "tcp" | "smr_centre";
  tcp_from?: Record<string, unknown> | null;
  /** "argus_sidecar_sof_plus_exposure" or "dataset_nfps_grid_episode_local". */
  camera_time_base: string;
};

export type TrackerValidateResponse = {
  ok: boolean;
  returncode?: number;
  kind?: "validate";
  report?: TrackerValidateReport | null;
  reportPath?: string;
  summary?: string;
  episodeDir?: string;
  exposureFraction?: number;
  stderr?: string;
  error?: string;
};

// --- One-click tracker-mount capture: record -> discover -> solve -----------

/**
 * Who owns the recorder while parked poses are collected.
 *
 * Minted and kept by the gateway, not by this page: the session name used to
 * live in React state, so a reload renamed the run in flight and orphaned every
 * dwell already on disk, and Live Record had no way to know a mount capture was
 * under way. Both pages read this one object instead.
 */
export type TrackerMountSession = {
  active: boolean;
  /** idle | capture | landed | failed. `landed` is the only solvable one. */
  stage: string;
  sessionName: string;
  captureRoot: string;
  trackerSessionId: string;
  landedPath: string;
  /** Presses. Diverges from `dwellsOnDisk` when a take was discarded. */
  dwellsStarted: number;
  /** Episode directories actually written -- what the solve will read. */
  dwellsOnDisk: number;
  message: string;
  startedAt: string;
  recorderState: string;
  episodeInFlight: boolean;
  /** "dwell" (station + lever arm) or "pivot" (E1p, TCP pinned in the socket). */
  kind?: "dwell" | "pivot";
  /** Segments saved before they could hold a dwell; each needs re-recording. */
  shortSegments?: number;
  lastSegmentSeconds?: number;
  segmentMinSeconds?: number;
  segmentSuggestedSeconds?: number;
};

/** E1p: production's TCP against the pivot socket the tracker finds. */
export type TrackerPivotReport = {
  n_poses: number;
  cube?: string;
  mount_id?: string;
  static_tcp_error_mm: { p95: number; rms: number; max: number; per_pose: number[] };
  tcp_budget_mm: number;
  static_p95_within_budget: boolean;
  c_tcp_production_mm: number[];
  c_tcp_measured_mm: number[];
  c_tcp_error_mm: number[];
  c_tcp_error_norm_mm: number;
  d_cube_mm: number[];
  split: null | {
    cube_frame_constant_mm: number[];
    world_frame_constant_mm: number[];
    pose_dependent_rms_mm: number;
    gain_min: number;
  };
  sphere: null | {
    radius_mm: number;
    rms_mm: number;
    gain_min: number;
    center_sigma_norm_mm: number | null;
    center_sigma_weak_mm: number;
    /** Continuous pivots only: what the sphere was fitted from. */
    continuous?: {
      n_points: number;
      n_points_seated: number;
      lifted_fraction: number | null;
      n_direction_cells: number;
      max_radial_mm: number;
      radial_sigma_mm: number | null;
    };
  };
  radius_check_mm: null | {
    sphere_radius_mm: number;
    socket_to_smr_from_camera_mm: number;
    difference_mm: number;
  };
  certifies: boolean;
  certify_reasons: string[];
  cannot_see: string[];
  /** SMR -> TCP in the cube frame, TCP end from the tracker; only with a lever-arm fit. */
  smr_to_tcp_cube_mm?: number[];
  smr_to_tcp_norm_mm?: number;
  /** "tcp" when graded on production's own TCP labels, "cube" on cube poses. */
  pose_frame?: string;
  bundle_calibration_id?: string | null;
  /** Legacy dwell-based pivots: samples without a pause; listed, not fatal. */
  episodes_without_dwells?: { dataset: string; episode: number; why: string }[];
  /** Continuous pivots: samples with no solved frame or no beam; listed, not fatal. */
  episodes_skipped?: { dataset: string; episode: number; why: string }[];
  /** Continuous pivots: every seated frame compared, pooled into attitude cells. */
  sampling?: TrackerPivotSampling;
};

export type TrackerPivotSampling = {
  mode: "continuous";
  n_frames: number;
  n_frames_seated: number;
  n_attitudes: number;
  attitude_cell_deg: number;
  per_frame_error_mm: null | { n: number; p95: number; rms: number; max: number };
  by_smr_speed: { smr_speed_mm_s: [number, number | null]; n: number; p95?: number; rms?: number; max?: number }[];
  lift_sensitivity: null | { median: number; p10: number };
};

export type TrackerMountSessionResponse = {
  ok: boolean;
  error?: string;
  session?: TrackerMountSession;
};

export type TrackerMountCapture = {
  dataset: string;
  datasetName: string;
  episode: number;
  episodeDir: string;
  sessionId: string;
  sessionPath: string;
  /** The session seals and lands at Disconnect, not at the end of an episode. */
  landed: boolean;
  poseLabel: string;
  purpose: string;
  /** smr_parked_pose_dwell | tcp_pivot_sweep (tcp_pivot_dwell before 2026-09-22); empty on older captures. */
  protocol?: string;
  segmentSeconds?: number;
  beamValidFraction: number;
  streamAdvanced: boolean;
  trackerError: string;
  modifiedUnixS: number;
};

export type TrackerMountCaptureListResponse = {
  ok: boolean;
  episodes?: TrackerMountCapture[];
  error?: string;
};

export type TrackerMountCaptureDeleteResponse = {
  ok: boolean;
  deleted?: number;
  removedDirs?: string[];
  /** Tracker streams no surviving episode referred to any more. */
  removedStreams?: string[];
  error?: string;
};

export type TrackerMountRecordResponse = {
  ok: boolean;
  captureRoot?: string;
  episodeIndex?: number;
  seconds?: number;
  error?: string;
};

export type TrackerMountChainResponse = {
  ok: boolean;
  /** Diagnostic mode graded with a fit that ran but does not certify. */
  uncertifiedFit?: boolean;
  fit?: TrackerMountSolveResponse | null;
  validate?: TrackerValidateResponse | null;
  error?: string;
};
