// What keeps the Rollout page's Execute button disabled (the gateway re-checks every one of these). With
// none left, the button opens the dialog "机器人会动，手放急停"; only its confirm sends attended=true.
import type { RolloutCheckpoint, RolloutStatus } from "./rolloutApi";

// The gateway refuses a live job when the recorder was connected for inference without one of the
// checkpoint's cameras (rollout_runs.frame_source); "" when nothing stands in the way.
export function cameraBlocker(status: RolloutStatus | null, checkpoint: RolloutCheckpoint | undefined): string {
  const open = status?.recorderCameras ?? [];
  if (!checkpoint || open.length === 0) return "";
  const missing = checkpoint.cameras.filter((c) => !open.includes(c));
  return missing.length
    ? `the recorder is connected for inference with ${open.join(", ")} only; this checkpoint reads ${missing.join(", ")}`
    : "";
}

// While the recorder is still connecting, nothing may start or stop it from this page: the Disconnect
// button waits until every camera and BOX it opened streams (or one has failed, so it can be torn down).
// A connect stuck past the gateway's limit with nothing failed may be cancelled, naming what never came up.
export function recorderConnecting(status: RolloutStatus | null): {
  pending: boolean;
  stuck: boolean;
  label: string;
  waiting: string[];
} {
  const r = status?.recorder;
  if (!status?.recorderConnected || !r?.pending) return { pending: false, stuck: false, label: "", waiting: [] };
  const ready = r.sensors.filter((s) => s.state === "running").length;
  const waiting = r.sensors.filter((s) => s.state !== "running").map((s) => s.id);
  const label = r.sensors.length ? `Connecting… ${ready}/${r.sensors.length} sensors ready` : "Connecting…";
  return { pending: true, stuck: Boolean(r.stuck), label, waiting };
}

// Once the recorder is up, the sensors it opened that do not stream: a BOX it did not discover ("Box devices:
// (none)") or that went stale, a camera that failed. The recorder still says "Episode N ready" without its
// BOX, so the page must say so out loud: the policy reads the arm gripper BOX's width. noBox = no BOX row
// at all. Empty while not connected or still connecting (recorderConnecting says that).
export function recorderNotStreaming(status: RolloutStatus | null): {
  boxes: string[];
  cameras: string[];
  noBox: boolean;
} {
  const r = status?.recorder;
  if (!status?.recorderConnected || !r || r.pending) return { boxes: [], cameras: [], noBox: false };
  const down = (kind: string) => r.sensors.filter((s) => s.kind === kind && s.state !== "running").map((s) => s.id);
  return {
    boxes: down("box_collection"),
    cameras: down("camera"),
    noBox: !r.sensors.some((s) => s.kind === "box_collection")
  };
}

export function executeBlockers(args: {
  status: RolloutStatus | null;
  checkpoint: RolloutCheckpoint | undefined;
  previewReplayed: boolean;
}): string[] {
  const { status, checkpoint, previewReplayed } = args;
  const out: string[] = [];
  if (!status) return ["loading"];
  if (status.sandbox) out.push("a sandbox has no arm");
  if (status.profile !== "thor") out.push("this gateway is not a Thor");
  if (!checkpoint) out.push("pick a checkpoint");
  else if (!checkpoint.hasWorkspace) out.push("the checkpoint carries no training workspace");
  if (!status.recorderConnected) out.push("connect the recorder");
  else if (recorderConnecting(status).pending) out.push("the recorder is still connecting");
  const cameras = cameraBlocker(status, checkpoint);
  if (cameras) out.push(cameras);
  if (status.jobs.some((j) => j.state === "running")) out.push("another rollout job is running");
  if (!previewReplayed) out.push("no identical Replay check of a preview of this checkpoint yet");
  return out;
}
