// Calls of the Training page (gateway training_runs.py). Separate from api.ts: the page polls its own
// status instead of growing the snapshot every page receives.
import { gatewayErrorMessage } from "../shared/httpError";

const base = import.meta.env.VITE_GUI_API_BASE ?? "";

export type TrainingHost = { id: string; label: string; kind: "local" | "remote"; sshTarget: string; repoDir: string; python: string };

export type TrainingRun = {
  job: string;
  hostId: string;
  hostLabel: string;
  kind: "local" | "remote";
  state: "syncing" | "starting" | "running" | "done" | "failed" | "stopped" | "lost";
  message: string;
  pid: number | null;
  runDir: string;
  localRunDir: string;
  startedAt: string;
  finishedAt: string;
  step: number;
  totalSteps: number;
  loss: number | null;
  lastCheckpoint: string;
  reloadCheck: { ok: boolean; max_abs_diff?: number; error?: string } | null;
  params: Record<string, unknown>;
  exports: string[];
  resumes: string[];
  fetchedAt: string;
};

export type TrainingStatus = {
  hosts: TrainingHost[];
  runs: TrainingRun[];
  profile: string;
  sandbox: string | null;
  recorderConnected: boolean;
};

export type TrainingExport = {
  name: string;
  path: string;
  episodes: number;
  frames: number;
  fps: number;
  hands: string[];
  cameras: string[];
  qcGated: boolean;
  worldFrameId: string;
  labelsFingerprint: string;
  usable: boolean;
  problems: string[];
  modified: string;
};

export type ExportCheck = {
  ok: boolean;
  hand: string;
  problems: string[];
  warnings: string[];
  commonCameras: Record<string, [number, number]>;
  boxes: string[];
};

export type Checkpoint = {
  step: number | string;
  path: string;
  hasContract: boolean;
  reloadCheck: { ok: boolean; max_abs_diff?: number; error?: string } | null;
  exports: string[];
  code: Record<string, unknown> | null;
};

export type ProbeResult = {
  ok: boolean;
  ready?: boolean;
  error?: string;
  python?: string;
  torch?: string;
  cuda?: boolean;
  gpu?: string;
  gpu_mem_gb?: number;
  lerobot?: boolean;
  torchcodec?: boolean;
  av?: boolean;
  ffmpeg?: boolean;
};

async function call<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { Accept: "application/json", ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
    body: body === undefined ? undefined : JSON.stringify(body)
  });
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) throw new Error(gatewayErrorMessage(response.status, payload));
  return payload as T;
}

export const trainingApi = {
  status: () => call<TrainingStatus>("/api/training/status"),
  exports: () => call<{ exports: TrainingExport[] }>("/api/training/exports").then((r) => r.exports),
  check: (exports: string[], hand: string) =>
    call<{ check: ExportCheck }>("/api/training/check", { exports, hand }).then((r) => r.check),
  crop: (exportName: string, camera: string, hand: string) =>
    call<{ crop: [number, number, number, number]; note: string; points: number }>(
      `/api/training/crop?${new URLSearchParams({ export: exportName, camera, hand })}`
    ),
  frameUrl: (exportName: string, camera: string, crop: number[] | null, at: number) =>
    `${base}/api/training/frame.jpg?${new URLSearchParams({
      export: exportName,
      camera,
      at: String(at),
      ...(crop ? { crop: crop.join(",") } : {})
    })}`,
  probe: (hostId: string) => call<ProbeResult>("/api/training/probe", { hostId }),
  addHost: (host: { label: string; sshTarget: string; repoDir: string; python: string }) =>
    call<{ host: TrainingHost }>("/api/training/hosts/add", host),
  removeHost: (hostId: string) => call("/api/training/hosts/remove", { hostId }),
  start: (body: Record<string, unknown>) => call<{ run: TrainingRun }>("/api/training/start", body),
  stop: (job: string) => call<{ message: string }>("/api/training/stop", { job }),
  resume: (job: string, steps: number | null) => call<{ run: TrainingRun }>("/api/training/resume", { job, steps }),
  fetchRun: (job: string) => call<{ run: TrainingRun }>("/api/training/fetch", { job }),
  remove: (job: string) => call("/api/training/delete", { job }),
  log: (job: string) => call<{ lines: string[] }>(`/api/training/log?job=${encodeURIComponent(job)}`).then((r) => r.lines),
  checkpoints: (job: string) =>
    call<{ checkpoints: Checkpoint[] }>(`/api/training/checkpoints?job=${encodeURIComponent(job)}`).then(
      (r) => r.checkpoints
    )
};
