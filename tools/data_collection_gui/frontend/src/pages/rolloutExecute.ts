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
export function recorderConnecting(status: RolloutStatus | null): { pending: boolean; label: string } {
  const r = status?.recorder;
  if (!status?.recorderConnected || !r?.pending) return { pending: false, label: "" };
  const ready = r.sensors.filter((s) => s.state === "running").length;
  return {
    pending: true,
    label: r.sensors.length ? `Connecting… ${ready}/${r.sensors.length} sensors ready` : "Connecting…"
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
