// Calls of the Rollout page (gateway rollout_runs.py).
import { gatewayErrorMessage } from "../shared/httpError";

const base = import.meta.env.VITE_GUI_API_BASE ?? "";

export type Stage = { id: string; label: string; status: string };

export type RolloutJob = {
  id: string;
  kind: "offline" | "preview" | "replay" | "execute";
  checkpoint: string;
  state: "running" | "done" | "failed" | "stopped";
  message: string;
  startedAt: string;
  finishedAt: string;
  out: string;
  params: Record<string, unknown>;
  last: Record<string, unknown>;
};

export type RecorderReadiness = {
  state: string;
  sensors: { id: string; kind: string; state: string }[];
  ready: boolean;
  pending: boolean;
  failed: boolean;
};

export type RolloutStatus = {
  stages: Stage[];
  jobs: RolloutJob[];
  profile: string;
  sandbox: string | null;
  recorderConnected: boolean;
  // connected for inference: the only cameras the recorder opened ([] = a full session)
  recorderCameras?: string[];
  // pending = the recorder is still bringing its cameras and BOXes up (it has not said "Episode N ready")
  recorder?: RecorderReadiness;
  executeStages?: Record<string, string>;
};

export type RolloutCheckpoint = {
  path: string;
  run: string;
  step: number | string;
  hand: string;
  cameras: string[];
  wrench: boolean;
  exports: string[];
  reloadCheck: { ok: boolean; max_abs_diff?: number } | null;
  hasWorkspace: boolean;
  saved: string;
};

export type ResultRow = {
  path: string;
  kind: string;
  created: string;
  checkpoint: string;
  export?: string;
  frames?: number;
  trainedOnThisExport?: boolean;
  posMm?: number;
  rotDeg?: number;
  stillMm?: number;
  viewVsOnlineMm?: number;
  arm?: string;
  steps?: number;
  replayCheck?: { ok: boolean; max_abs_diff: number; steps: number } | null;
  planProblem?: string;
  stage?: number | null;
  result?: { completed: boolean; stopped: string; chunks: number; duration_s: number } | null;
};

export type ErrorSummary = { mean: number; p95: number; by_step: (number | null)[] };

export type OfflineFrame = {
  episode: number;
  frame: number;
  tcp: number[];
  pred: number[][];
  truth: (number | null)[][];
  pred_width: number[];
  truth_width: number[];
  pos_err_mm_end: number | null;
};

export type OfflineResult = {
  kind: "offline";
  created: string;
  checkpoint: string;
  export: string;
  trained_on_this_export: boolean;
  frames: number;
  stride: number;
  episodes: number[];
  error: { pos_mm: ErrorSummary; rot_deg: ErrorSummary; width_mm: ErrorSummary };
  not_moving_baseline_pos_mm: ErrorSummary;
  view_vs_online_path_mm: { max: number; mean: number; note: string };
  series: OfflineFrame[];
};

export type PreviewLatest = {
  step?: number;
  time?: string;
  width?: number;
  wrench?: number[] | null;
  latency_ms?: number;
  targets_tcp_mm?: number[][];
  widths?: number[];
  cluster?: { publish_seq: number; max_delta_ns: number; recording: boolean };
  plan?: {
    ok: boolean;
    reasons: string[];
    tcp_base: number[];
    min_env_gap_mm: number | null;
    ik_ok: boolean[];
    inside_workspace: boolean[] | null;
  } | null;
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

const q = (params: Record<string, string>) => new URLSearchParams(params).toString();

export const rolloutApi = {
  status: () => call<RolloutStatus>("/api/rollout/status"),
  checkpoints: () => call<{ checkpoints: RolloutCheckpoint[] }>("/api/rollout/checkpoints").then((r) => r.checkpoints),
  results: () => call<{ results: ResultRow[] }>("/api/rollout/results").then((r) => r.results),
  result: (path: string) => call<OfflineResult>(`/api/rollout/result?${q({ path })}`),
  log: (job: string) => call<{ lines: string[] }>(`/api/rollout/log?${q({ job })}`).then((r) => r.lines),
  latest: (record: string) => call<PreviewLatest>(`/api/rollout/preview-latest?${q({ record })}`),
  previewImageUrl: (record: string, tick: number) => `${base}/api/rollout/preview.jpg?${q({ record, t: String(tick) })}`,
  offline: (checkpoint: string, exportName: string, episodes: number[], stride: number) =>
    call<{ job: RolloutJob }>("/api/rollout/offline", { checkpoint, export: exportName, episodes, stride }),
  preview: (checkpoint: string, arm: string, rate: number, readJoints: boolean) =>
    call<{ job: RolloutJob }>("/api/rollout/preview", { checkpoint, arm, rate, readJoints }),
  replay: (record: string) => call<{ job: RolloutJob }>("/api/rollout/replay", { record }),
  execute: (checkpoint: string, arm: string, stage: number, attended: boolean) =>
    call<{ job: RolloutJob }>("/api/rollout/execute", { checkpoint, arm, stage, attended }),
  stop: (job: string) => call("/api/rollout/stop", { job }),
  // Connect the recorder for inference: only these cameras, the main frame bus at 30 Hz, no episodes.
  connectForInference: (cameras: string[]) =>
    call(`/api/handheld/record/connect?${q({ cameras: cameras.join(",") })}`, {}),
  disconnect: () => call("/api/handheld/record/exit", {})
};
