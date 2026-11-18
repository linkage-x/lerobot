// What keeps the Rollout page's Execute button disabled (the gateway re-checks every one of these).
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

export function executeBlockers(args: {
  status: RolloutStatus | null;
  checkpoint: RolloutCheckpoint | undefined;
  arm: string;
  attended: boolean;
  previewReplayed: boolean;
}): string[] {
  const { status, checkpoint, arm, attended, previewReplayed } = args;
  const out: string[] = [];
  if (!status) return ["loading"];
  if (status.sandbox) out.push("a sandbox has no arm");
  if (status.profile !== "thor") out.push("this gateway is not a Thor");
  if (!checkpoint) out.push("pick a checkpoint");
  else if (!checkpoint.hasWorkspace) out.push("the checkpoint carries no training workspace");
  if (!status.recorderConnected) out.push("connect the recorder");
  const cameras = cameraBlocker(status, checkpoint);
  if (cameras) out.push(cameras);
  const lock = status.unlock?.[arm];
  if (!lock?.unlocked) out.push(`the ${arm} arm is locked${lock?.why ? `: ${lock.why}` : ""}`);
  if (status.jobs.some((j) => j.state === "running")) out.push("another rollout job is running");
  if (!previewReplayed) out.push("no identical Replay check of a preview of this checkpoint yet");
  if (!attended) out.push("confirm that you are at the e-stop");
  return out;
}
