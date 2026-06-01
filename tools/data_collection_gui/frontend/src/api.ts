import { handheldConfigSummary, initialDevices } from "./defaultHandheldConfig";
import type {
  CollectionTask,
  EpisodeAnnotation,
  CalibrationCamera,
  CalibrationStatus,
  ConfigSummary,
  DatasetExportStatus,
  DeviceStatus,
  BoxPreviewPayload,
  EventLogItem,
  GatewayStatus,
  ProcessingItem,
  ProcessingStatus,
  RecordedDataset,
  RecordingStatus,
  ReplayStatus,
  ReplayTimeline,
  TrajectoryPoint
} from "./types";

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

const defaultMujocoValidation = (datasetRoot: string, episode: number, fps: number): ReplayStatus["mujocoValidation"] => ({
  status: "not_run",
  datasetRoot: "",
  episode,
  fps,
  exitCode: null,
  completedFrames: 0,
  totalFrames: 0,
  avgPositionErrorMm: null,
  maxPositionErrorMm: null,
  avgRotationErrorDeg: null,
  maxRotationErrorDeg: null,
  maxPositionThresholdMm: 20,
  maxRotationThresholdDeg: 15,
  hasStructuredResult: false,
  trajectoryContract: {},
  isCurrentForSelection: false,
  message: datasetRoot ? "Run MuJoCo replay before real-robot replay." : "Select a recorded dataset before validation.",
  updatedAt: ""
});

export type GuiSnapshot = {
  gateway: GatewayStatus;
  configSummary: ConfigSummary;
  devices: DeviceStatus[];
  recording: RecordingStatus;
  replay: ReplayStatus;
  annotation: EpisodeAnnotation;
  calibration: CalibrationStatus;
  datasetExport: DatasetExportStatus;
  recordedDatasets: RecordedDataset[];
  processing: ProcessingItem[];
  trajectory: TrajectoryPoint[];
  events: EventLogItem[];
  tasks: CollectionTask[];
  notice?: string;
};

export class DataCollectionGuiApi {
  private readonly apiBase = import.meta.env.VITE_GUI_API_BASE ?? "";
  private usingRemote = false;
  private snapshot: GuiSnapshot = {
    gateway: {
      configPath: handheldConfigSummary.configPath,
      pid: null,
      state: "mock",
      processElapsedS: null,
      datasetsRoot: "outputs/datasets"
    },
    configSummary: {
      configPath: handheldConfigSummary.configPath,
      repoId: handheldConfigSummary.repoId,
      root: handheldConfigSummary.root,
      fps: handheldConfigSummary.fps,
      episodeTimeS: handheldConfigSummary.episodeTimeS,
      targetFrames: handheldConfigSummary.targetFrames,
      numEpisodes: "unlimited",
      video: handheldConfigSummary.video,
      streamingEncoding: handheldConfigSummary.streamingEncoding,
      vcodec: handheldConfigSummary.vcodec,
      softSync: handheldConfigSummary.softSync,
      rerun: handheldConfigSummary.rerun,
      recorderScript: handheldConfigSummary.recorderScript,
      rigType: handheldConfigSummary.rigType,
      hardwareSync: handheldConfigSummary.hardwareSync,
      cameraDefaults: handheldConfigSummary.cameraDefaults
    },
    devices: initialDevices,
    recording: {
      state: "idle",
      datasetRoot: handheldConfigSummary.root,
      repoId: handheldConfigSummary.repoId,
      episodeIndex: 0,
      savedEpisodes: 0,
      frameIndex: 0,
      targetFrames: handheldConfigSummary.targetFrames,
      queueDepth: 0,
      message: "Ready to launch handheld recorder"
    },
    replay: {
      state: "idle",
      dataset: handheldConfigSummary.repoId,
      episode: 0,
      frameIndex: 0,
      totalFrames: handheldConfigSummary.targetFrames,
      fps: handheldConfigSummary.fps,
      trackingErrorMm: 0,
      safety: "locked",
      message: "Start the local gateway to load recorded trajectory data",
      datasetRoot: handheldConfigSummary.root,
      sourcePath: "",
      dataStatus: "missing",
      trajectoryKind: "none",
      totalEpisodes: 0,
      episodeOptions: [],
      recordedFrames: 0,
      diagnostics: [],
      pid: null,
      lastOutput: "",
      mujocoValidation: defaultMujocoValidation(handheldConfigSummary.root, 0, handheldConfigSummary.fps)
    },
    annotation: {
      datasetRoot: handheldConfigSummary.root,
      episode: 0,
      taskPrompt: handheldConfigSummary.repoId,
      outcome: "unreviewed",
      quality: "unreviewed",
      includeInTraining: true,
      tags: [],
      notes: "",
      annotator: "",
      updatedAt: "",
      source: "default",
      segments: [],
      reviewStatus: "pending",
      reviewComment: ""
    },
    calibration: {
      state: "idle",
      pattern: "ChArUco 5x7 (mock)",
      lastRunAt: "",
      message: "Run calibration to refresh extrinsics",
      cameras: [],
      outputPath: ""
    },
    datasetExport: {
      state: "idle",
      target: "lerobot_v3",
      datasetRoot: handheldConfigSummary.root,
      outputPath: `${handheldConfigSummary.root}/exports/lerobot_v3`,
      selectedEpisodes: 0,
      totalFrames: 0,
      includeRaw: true,
      includeDebug: true,
      includeTraining: true,
      message: "Prepare an export after recording or loading an episode",
      manifest: [
        "raw/videos/*.mp4",
        "raw/events.jsonl",
        "debug/session.mcap",
        "debug/rerun.rrd",
        "training/lerobot_v3"
      ]
    },
    recordedDatasets: [],
    processing: [],
    trajectory: [],
    tasks: [],
    events: [
      {
        id: "boot",
        time: new Date().toLocaleTimeString(),
        level: "info",
        message: "GUI initialized with mock gateway adapter"
      }
    ]
  };

  async getSnapshot(): Promise<GuiSnapshot> {
    const remote = await this.getRemoteSnapshot();
    if (remote) {
      return remote;
    }
    await wait(120);
    this.snapshot = this.withFrontendFallbacks(this.snapshot);
    return structuredClone(this.snapshot);
  }

  async connectRecording(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/handheld/record/connect");
    if (remote) {
      return remote;
    }
    await wait(220);
    this.snapshot.recording = {
      ...this.snapshot.recording,
      state: "armed",
      pid: 4242,
      frameIndex: 0,
      queueDepth: 0,
      message: "Devices connected; ready to start episode"
    };
    this.snapshot.devices = this.snapshot.devices.map((device) => ({ ...device, state: "running" }));
    this.log("info", "Handheld devices connected");
    return this.getSnapshot();
  }

  async startRecording(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/handheld/record/start");
    if (remote) {
      return remote;
    }
    await wait(180);
    this.snapshot.recording = {
      ...this.snapshot.recording,
      state: "recording",
      frameIndex: 0,
      queueDepth: 1,
      message: `Recording episode via ${handheldConfigSummary.configPath}`
    };
    this.snapshot.devices = this.snapshot.devices.map((device) => ({ ...device, state: "running" }));
    this.log("info", "Episode recording started");
    return this.getSnapshot();
  }

  async prepareDatasetExport(target: DatasetExportStatus["target"]): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/dataset/export/prepare?target=${target}`);
    if (remote) {
      return this.withFrontendFallbacks(remote);
    }
    await wait(180);
    const totalFrames = this.snapshot.replay.recordedFrames || this.snapshot.trajectory.length;
    const selectedEpisodes = Math.max(this.snapshot.replay.totalEpisodes ?? 0, this.snapshot.recording.savedEpisodes);
    this.snapshot.datasetExport = {
      ...this.snapshot.datasetExport,
      state: "ready",
      target,
      datasetRoot: this.snapshot.replay.datasetRoot || this.snapshot.recording.datasetRoot,
      outputPath: `${this.snapshot.replay.datasetRoot || this.snapshot.recording.datasetRoot}/exports/${target}`,
      selectedEpisodes,
      totalFrames,
      message: `Export plan ready for ${target}`,
      manifest: this.exportManifest(target)
    };
    this.log("info", `Dataset export prepared: ${target}`);
    return this.getSnapshot();
  }

  async startDatasetExport(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/dataset/export/start");
    if (remote) {
      return this.withFrontendFallbacks(remote);
    }
    await wait(260);
    this.snapshot.datasetExport = {
      ...this.snapshot.datasetExport,
      state: "complete",
      message: `Export complete at ${this.snapshot.datasetExport.outputPath}`
    };
    this.log("info", `Dataset export complete: ${this.snapshot.datasetExport.outputPath}`);
    return this.getSnapshot();
  }

  async stopRecording(action: "save" | "discard" | "exit"): Promise<GuiSnapshot> {
    const endpoint =
      action === "save"
        ? "/api/handheld/record/stop-save"
        : action === "discard"
          ? "/api/handheld/record/stop-discard"
          : "/api/handheld/record/exit";
    const remote = await this.postRemoteSnapshot(endpoint);
    if (remote) {
      return remote;
    }
    await wait(180);
    const saved = action === "save" ? this.snapshot.recording.savedEpisodes + 1 : this.snapshot.recording.savedEpisodes;
    const exiting = action === "exit";
    this.snapshot.recording = {
      ...this.snapshot.recording,
      state: exiting ? "idle" : "armed",
      savedEpisodes: saved,
      episodeIndex: saved,
      frameIndex: 0,
      queueDepth: 0,
      pid: exiting ? null : this.snapshot.recording.pid,
      message: action === "save" ? "Episode saved; ready for next episode" : action === "discard" ? "Episode discarded; ready for next episode" : "Session stopped"
    };
    this.snapshot.devices = this.snapshot.devices.map((device) => ({ ...device, state: exiting ? "idle" : "running" }));
    this.log(action === "save" ? "info" : "warn", `Recording command: ${action}`);
    return this.getSnapshot();
  }

  async preflightReplay(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/replay/preflight");
    if (remote) {
      return remote;
    }
    await wait(220);
    this.snapshot.replay = {
      ...this.snapshot.replay,
      state: "aborted",
      safety: "fault",
      message: "Gateway unavailable; preflight cannot be mocked for real-robot safety"
    };
    this.log("error", "Replay preflight blocked because gateway is unavailable");
    return this.getSnapshot();
  }

  async selectRecordedDataset(path: string): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/replay/select-dataset?path=${encodeURIComponent(path)}`);
    if (remote) {
      return remote;
    }
    await wait(140);
    const selected = this.snapshot.recordedDatasets.find((dataset) => dataset.path === path);
    this.snapshot.replay = {
      ...this.snapshot.replay,
      state: "idle",
      safety: "locked",
      episode: 0,
      frameIndex: 0,
      trackingErrorMm: 0,
      dataset: path,
      datasetRoot: path,
      sourcePath: selected?.sourcePath ?? "",
      dataStatus: selected?.dataStatus ?? "missing",
      totalEpisodes: selected?.totalEpisodes ?? 0,
      episodeOptions: Array.from({ length: selected?.totalEpisodes ?? 0 }, (_item, index) => index),
      recordedFrames: selected?.totalFrames ?? 0,
      totalFrames: selected?.totalFrames || this.snapshot.replay.totalFrames,
      message: selected ? `Selected recorded dataset: ${selected.name}` : `Selected recorded dataset: ${path}`,
      mujocoValidation: defaultMujocoValidation(path, 0, this.snapshot.replay.fps)
    };
    this.log("info", `Selected replay dataset: ${path}`);
    return this.getSnapshot();
  }

  async selectReplayEpisode(episode: number): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/replay/select-episode?episode=${episode}`);
    if (remote) {
      return remote;
    }
    await wait(120);
    this.snapshot.replay = {
      ...this.snapshot.replay,
      state: "idle",
      safety: "locked",
      episode,
      frameIndex: 0,
      trackingErrorMm: 0,
      message: `Selected episode ${episode}`,
      mujocoValidation: defaultMujocoValidation(this.snapshot.replay.datasetRoot ?? this.snapshot.replay.dataset, episode, this.snapshot.replay.fps)
    };
    this.log("info", `Selected replay episode: ${episode}`);
    return this.getSnapshot();
  }

  async startReplay(realRobot: boolean): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(realRobot ? "/api/replay/start-real" : "/api/replay/start");
    if (remote) {
      return remote;
    }
    await wait(220);
    this.snapshot.replay = {
      ...this.snapshot.replay,
      state: "aborted",
      safety: "fault",
      frameIndex: 0,
      trackingErrorMm: 0,
      message: `Gateway unavailable; ${realRobot ? "real-robot replay" : "dry-run replay"} cannot be mocked for safety`
    };
    this.log("error", `${realRobot ? "Real robot replay" : "Dry-run replay"} blocked because gateway is unavailable`);
    return this.getSnapshot();
  }

  async startMujocoReplay(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/replay/start-mujoco");
    if (remote) {
      return remote;
    }
    await wait(220);
    this.snapshot.replay = {
      ...this.snapshot.replay,
      state: "aborted",
      safety: "fault",
      frameIndex: 0,
      trackingErrorMm: 0,
      pid: null,
      message: "Gateway unavailable; MuJoCo validation cannot be mocked for real-robot safety",
      mujocoValidation: defaultMujocoValidation(
        this.snapshot.replay.datasetRoot ?? this.snapshot.replay.dataset,
        this.snapshot.replay.episode,
        this.snapshot.replay.fps
      )
    };
    this.log("error", "MuJoCo validation blocked because gateway is unavailable");
    return this.getSnapshot();
  }

  async saveEpisodeAnnotation(annotation: EpisodeAnnotation): Promise<GuiSnapshot> {
    const remote = await this.postRemoteJsonSnapshot("/api/annotation/save", annotation);
    if (remote) {
      return remote;
    }
    await wait(160);
    this.snapshot.annotation = {
      ...annotation,
      tags: this.normalizeTags(annotation.tags),
      updatedAt: new Date().toISOString(),
      source: "manual"
    };
    this.log("info", `Annotation saved: episode ${annotation.episode}`);
    return this.getSnapshot();
  }

  async runCalibration(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/calibration/run");
    if (remote) {
      return remote;
    }
    await wait(450);
    const cameras: CalibrationCamera[] = this.snapshot.devices
      .filter((device) => device.kind === "camera")
      .map((device, index) => {
        const reprojection = Math.round((0.7 + ((index + 1) * 0.15) % 1.0) * 1000) / 1000;
        const baseline = Math.round((90 + index * 110) * 10) / 10;
        const status: CalibrationCamera["status"] = reprojection < 1.2 ? "pass" : reprojection < 1.8 ? "warn" : "fail";
        return { id: device.id, reprojectionMm: reprojection, baselineMm: baseline, status };
      });
    const failed = cameras.some((entry) => entry.status === "fail");
    this.snapshot.calibration = {
      state: cameras.length === 0 ? "failed" : failed ? "failed" : "complete",
      pattern: this.snapshot.calibration.pattern,
      lastRunAt: new Date().toISOString(),
      message:
        cameras.length === 0
          ? "No cameras configured in handheld config"
          : failed
            ? "Mock calibration finished with at least one fail"
            : `Mock calibration completed for ${cameras.length} cameras`,
      cameras,
      outputPath: `outputs/calibration/mock_${Date.now()}.json`
    };
    this.log(failed || cameras.length === 0 ? "warn" : "info", `Mock calibration: ${this.snapshot.calibration.message}`);
    return this.getSnapshot();
  }

  async setDatasetsRoot(path: string): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/processing/datasets-root?path=${encodeURIComponent(path)}`);
    if (remote) {
      if (remote.notice) {
        window.alert(remote.notice);
      }
      return remote;
    }
    await wait(120);
    this.snapshot.gateway = { ...this.snapshot.gateway, datasetsRoot: path };
    this.log("info", `Mock: datasets root set to ${path}`);
    return this.getSnapshot();
  }

  async queueTrajGen(path: string): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/processing/traj-gen?path=${encodeURIComponent(path)}`);
    if (remote) {
      return remote;
    }
    await wait(140);
    const message = "Gateway unavailable: cannot start EE trajectory generation.";
    window.alert(message);
    this.snapshot.processing = this.snapshot.processing.map((item) =>
      item.path === path
        ? {
            ...item,
            status: "pose_missing" as ProcessingStatus,
            message,
            logTail: [...item.logTail, `[traj-gen] ${message}`]
          }
        : item
    );
    this.log("warn", `Traj-gen unavailable: ${path}`);
    return this.getSnapshot();
  }

  async runQc(path: string): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/processing/qc?path=${encodeURIComponent(path)}`);
    if (remote) {
      return remote;
    }
    await wait(160);
    this.snapshot.processing = this.snapshot.processing.map((item) =>
      item.path === path
        ? {
            ...item,
            status: "qc_pass" as ProcessingStatus,
            qcSummary: "schema + sync + completeness pass",
            validFramesPct: item.validFramesPct ?? 96.4,
            message: "QC passed; available for export",
            logTail: [...item.logTail, "[qc] all checks passed"]
          }
        : item
    );
    this.log("info", `QC passed: ${path}`);
    return this.getSnapshot();
  }

  videoUrl(datasetPath: string, cameraKey: string): string {
    const params = new URLSearchParams({ path: datasetPath, key: cameraKey });
    return `${this.apiBase}/api/replay/video?${params.toString()}`;
  }

  cameraPreviewUrl(deviceId: string): string {
    const params = new URLSearchParams({ key: deviceId });
    return `${this.apiBase}/api/device-preview/camera.mjpeg?${params.toString()}`;
  }

  async fetchBoxPreview(deviceId: string): Promise<BoxPreviewPayload | null> {
    try {
      const params = new URLSearchParams({ device: deviceId });
      const response = await fetch(`${this.apiBase}/api/device-preview/box?${params.toString()}`, {
        headers: { Accept: "application/json" }
      });
      if (!response.ok) {
        return null;
      }
      return (await response.json()) as BoxPreviewPayload;
    } catch {
      return null;
    }
  }

  async fetchReplayTimeline(datasetPath: string, episode?: number): Promise<ReplayTimeline | null> {
    try {
      const params = new URLSearchParams({ path: datasetPath });
      if (episode != null) {
        params.set("episode", String(episode));
      }
      const response = await fetch(`${this.apiBase}/api/replay/timeline?${params.toString()}`, {
        headers: { Accept: "application/json" }
      });
      if (!response.ok) {
        return null;
      }
      return (await response.json()) as ReplayTimeline;
    } catch {
      return null;
    }
  }

  async abortReplay(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/replay/abort");
    if (remote) {
      return remote;
    }
    await wait(140);
    this.snapshot.replay = {
      ...this.snapshot.replay,
      state: "aborted",
      safety: "locked",
      message: "Replay aborted; robot command stream stopped"
    };
    this.log("warn", "Replay aborted");
    return this.getSnapshot();
  }

  async createTask(task: Partial<CollectionTask>): Promise<GuiSnapshot> {
    const remote = await this.postRemoteJsonSnapshot("/api/tasks/create", task);
    if (remote) {
      return remote;
    }
    await wait(140);
    const newTask: CollectionTask = {
      id: `task-${Date.now()}`,
      name: task.name ?? "",
      description: task.description ?? "",
      targetEpisodes: task.targetEpisodes ?? 0,
      completedEpisodes: 0,
      status: "pending",
      assignee: task.assignee ?? "",
      datasetRepoId: task.datasetRepoId ?? "",
      tags: task.tags ?? [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    this.snapshot.tasks = [...this.snapshot.tasks, newTask];
    this.log("info", `Created task: ${newTask.name}`);
    return this.getSnapshot();
  }

  async updateTask(task: Partial<CollectionTask>): Promise<GuiSnapshot> {
    const remote = await this.postRemoteJsonSnapshot("/api/tasks/update", task);
    if (remote) {
      return remote;
    }
    await wait(120);
    this.snapshot.tasks = this.snapshot.tasks.map((t) =>
      t.id === task.id ? { ...t, ...task, updatedAt: new Date().toISOString() } : t
    );
    this.log("info", `Updated task: ${task.name ?? task.id}`);
    return this.getSnapshot();
  }

  async deleteTask(taskId: string): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/tasks/delete?id=${encodeURIComponent(taskId)}`);
    if (remote) {
      return remote;
    }
    await wait(120);
    this.snapshot.tasks = this.snapshot.tasks.filter((t) => t.id !== taskId);
    this.log("info", `Deleted task: ${taskId}`);
    return this.getSnapshot();
  }

  tick(): GuiSnapshot {
    if (this.usingRemote) {
      return structuredClone(this.snapshot);
    }

    if (this.snapshot.recording.state === "recording") {
      const nextFrame = Math.min(this.snapshot.recording.frameIndex + 3, this.snapshot.recording.targetFrames);
      this.snapshot.recording = {
        ...this.snapshot.recording,
        frameIndex: nextFrame,
        queueDepth: (this.snapshot.recording.queueDepth + 1) % 7,
        state: nextFrame >= this.snapshot.recording.targetFrames ? "review" : "recording",
        message: nextFrame >= this.snapshot.recording.targetFrames ? "Episode reached target duration; save or discard" : "Capturing frames"
      };
    }

    if (this.snapshot.replay.state === "dry_run" || this.snapshot.replay.state === "sim_replay" || this.snapshot.replay.state === "replaying") {
      const nextFrame = Math.min(this.snapshot.replay.frameIndex + 2, this.snapshot.replay.totalFrames);
      this.snapshot.replay = {
        ...this.snapshot.replay,
        frameIndex: nextFrame,
        trackingErrorMm: this.snapshot.replay.state === "replaying" ? 2 + Math.sin(nextFrame / 18) * 1.2 : 0,
        state: nextFrame >= this.snapshot.replay.totalFrames ? "complete" : this.snapshot.replay.state,
        safety: nextFrame >= this.snapshot.replay.totalFrames ? "locked" : this.snapshot.replay.safety,
        pid: nextFrame >= this.snapshot.replay.totalFrames ? null : this.snapshot.replay.pid,
        message: nextFrame >= this.snapshot.replay.totalFrames ? "Replay complete" : this.snapshot.replay.message
      };
    }

    return structuredClone(this.snapshot);
  }

  private log(level: EventLogItem["level"], message: string) {
    this.snapshot.events = [
      {
        id: `${Date.now()}-${message}`,
        time: new Date().toLocaleTimeString(),
        level,
        message
      },
      ...this.snapshot.events
    ].slice(0, 12);
  }

  private async getRemoteSnapshot(): Promise<GuiSnapshot | null> {
    try {
      const response = await fetch(`${this.apiBase}/api/snapshot`, {
        headers: { Accept: "application/json" }
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const snapshot = (await response.json()) as GuiSnapshot;
      this.snapshot = this.withFrontendFallbacks(snapshot);
      this.usingRemote = true;
      return structuredClone(this.snapshot);
    } catch {
      this.usingRemote = false;
      this.log("warn", "Gateway unavailable; using mock adapter");
      return null;
    }
  }

  private async postRemoteSnapshot(endpoint: string): Promise<GuiSnapshot | null> {
    try {
      const response = await fetch(`${this.apiBase}${endpoint}`, {
        method: "POST",
        headers: { Accept: "application/json" }
      });
      if (!response.ok) {
        const message = await this.remoteErrorMessage(response);
        this.applyRemoteCommandError(endpoint, message);
        this.usingRemote = true;
        return structuredClone(this.snapshot);
      }
      const snapshot = (await response.json()) as GuiSnapshot;
      this.snapshot = this.withFrontendFallbacks(snapshot);
      this.usingRemote = true;
      return structuredClone(this.snapshot);
    } catch {
      this.usingRemote = false;
      this.log("warn", "Gateway command failed; using mock adapter");
      return null;
    }
  }

  private async postRemoteJsonSnapshot(endpoint: string, payload: unknown): Promise<GuiSnapshot | null> {
    try {
      const response = await fetch(`${this.apiBase}${endpoint}`, {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        const message = await this.remoteErrorMessage(response);
        this.applyRemoteCommandError(endpoint, message);
        this.usingRemote = true;
        return structuredClone(this.snapshot);
      }
      const snapshot = (await response.json()) as GuiSnapshot;
      this.snapshot = this.withFrontendFallbacks(snapshot);
      this.usingRemote = true;
      return structuredClone(this.snapshot);
    } catch {
      this.usingRemote = false;
      this.log("warn", "Gateway command failed; using mock adapter");
      return null;
    }
  }

  private async remoteErrorMessage(response: Response): Promise<string> {
    try {
      const payload = await response.json();
      if (payload && typeof payload.error === "string") {
        return payload.error;
      }
    } catch {
      // Fall back to HTTP status text below.
    }
    return `HTTP ${response.status}`;
  }

  private applyRemoteCommandError(endpoint: string, message: string) {
    const command = endpoint.split("/").filter(Boolean).at(-1) ?? "command";
    if (endpoint.includes("/handheld/record/")) {
      this.snapshot.recording = {
        ...this.snapshot.recording,
        state: "error",
        message: `Recording ${command} failed: ${message}`
      };
    } else if (endpoint.includes("/replay/")) {
      this.snapshot.replay = {
        ...this.snapshot.replay,
        state: "aborted",
        safety: "fault",
        message: `Replay ${command} failed: ${message}`
      };
    } else if (endpoint.includes("/dataset/export/")) {
      this.snapshot.datasetExport = {
        ...this.snapshot.datasetExport,
        state: "error",
        message: `Export ${command} failed: ${message}`
      };
    } else if (endpoint.includes("/processing/datasets-root")) {
      window.alert(`Datasets Root save failed: ${message}`);
    } else if (endpoint.includes("/processing/traj-gen")) {
      const fallbackMessage = "Generate EE Trajectory failed.";
      const displayMessage = message || fallbackMessage;
      const targetPath = new URLSearchParams(endpoint.split("?")[1] ?? "").get("path");
      window.alert(displayMessage);
      this.snapshot.processing = this.snapshot.processing.map((item) =>
        !targetPath || item.path === targetPath
          ? {
              ...item,
              status: "pose_missing" as ProcessingStatus,
              message: displayMessage,
              logTail: [...item.logTail, `[traj-gen] ${displayMessage}`]
            }
          : item
      );
    }
    this.log("error", `${command} failed: ${message}`);
  }

  private withFrontendFallbacks(snapshot: GuiSnapshot): GuiSnapshot {
    const recordedDatasets =
      snapshot.recordedDatasets?.length || snapshot.gateway.state !== "mock"
        ? snapshot.recordedDatasets ?? []
        : this.mockRecordedDatasets(snapshot);
    const processing = snapshot.processing ?? this.snapshot.processing ?? [];
    const activeAnnotationPath = snapshot.replay.datasetRoot || snapshot.replay.dataset || snapshot.recording.datasetRoot;
    const annotation =
      snapshot.annotation?.datasetRoot === activeAnnotationPath && snapshot.annotation.episode === snapshot.replay.episode
        ? snapshot.annotation
        : this.mockAnnotation(snapshot, false);
    const annotationWithDefaults: typeof annotation = {
      ...annotation,
      segments: annotation.segments ?? [],
      reviewStatus: annotation.reviewStatus ?? "pending",
      reviewComment: annotation.reviewComment ?? ""
    };
    return {
      ...snapshot,
      replay: {
        ...snapshot.replay,
        mujocoValidation:
          snapshot.replay.mujocoValidation ??
          defaultMujocoValidation(snapshot.replay.datasetRoot || snapshot.replay.dataset, snapshot.replay.episode, snapshot.replay.fps)
      },
      recordedDatasets,
      processing,
      annotation: annotationWithDefaults,
      tasks: snapshot.tasks ?? this.snapshot.tasks ?? [],
      calibration: snapshot.calibration ?? this.snapshot.calibration,
      datasetExport: snapshot.datasetExport ?? {
        ...this.snapshot.datasetExport,
        datasetRoot: snapshot.replay.datasetRoot || snapshot.recording.datasetRoot,
        outputPath: `${snapshot.replay.datasetRoot || snapshot.recording.datasetRoot}/exports/${this.snapshot.datasetExport.target}`,
        selectedEpisodes: Math.max(snapshot.replay.totalEpisodes ?? 0, snapshot.recording.savedEpisodes),
        totalFrames: snapshot.replay.recordedFrames || snapshot.trajectory.length,
        message: "Export planning is available in the frontend mock adapter"
      }
    };
  }

  private mockAnnotation(snapshot: GuiSnapshot, reuseExisting = true): EpisodeAnnotation {
    const datasetRoot = snapshot.replay.datasetRoot || snapshot.replay.dataset || snapshot.recording.datasetRoot;
    const existing = this.snapshot.annotation;
    if (reuseExisting && existing?.datasetRoot === datasetRoot && existing.episode === snapshot.replay.episode) {
      return existing;
    }
    return {
      datasetRoot,
      episode: snapshot.replay.episode ?? 0,
      taskPrompt: snapshot.configSummary.repoId,
      outcome: "unreviewed",
      quality: "unreviewed",
      includeInTraining: true,
      tags: [],
      notes: "",
      annotator: "",
      updatedAt: "",
      source: "default",
      segments: [],
      reviewStatus: "pending",
      reviewComment: ""
    };
  }

  private normalizeTags(tags: string[]): string[] {
    return tags.map((tag) => tag.trim()).filter(Boolean).slice(0, 12);
  }

  private mockRecordedDatasets(snapshot: GuiSnapshot): RecordedDataset[] {
    const root = snapshot.replay.datasetRoot || snapshot.recording.datasetRoot;
    if (!root) {
      return [];
    }
    return [
      {
        path: root,
        name: root.split("/").filter(Boolean).at(-1) ?? root,
        updatedAt: new Date().toLocaleString(),
        updatedAtMs: Date.now(),
        totalEpisodes: snapshot.replay.totalEpisodes ?? snapshot.recording.savedEpisodes,
        totalFrames: snapshot.replay.recordedFrames ?? snapshot.trajectory.length,
        dataStatus: snapshot.replay.dataStatus ?? (snapshot.trajectory.length ? "loaded" : "missing"),
        sourcePath: snapshot.replay.sourcePath ?? "",
        isLatest: true
      }
    ];
  }

  private exportManifest(target: DatasetExportStatus["target"]): string[] {
    if (target === "mcap") {
      return ["debug/session.mcap", "debug/timeline_index.json", "debug/rerun.rrd"];
    }
    if (target === "parquet") {
      return ["training/data/chunk-*/episode_*.parquet", "training/meta/info.json", "training/meta/stats.json"];
    }
    return [
      "training/lerobot_v3/data/chunk-*/episode_*.parquet",
      "training/lerobot_v3/videos/chunk-*/*.mp4",
      "training/lerobot_v3/meta/info.json",
      "training/lerobot_v3/meta/episodes.jsonl"
    ];
  }
}
