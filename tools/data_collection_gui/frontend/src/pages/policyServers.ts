import type { PolicyServerEntry, PolicyServerListing } from "../types";

/** The empty choice: run the forward pass on the gateway's own machine. */
export const LOCAL_POLICY = "";

/** Whether `choice` names something this listing can run a policy on right now. */
export function choiceAvailable(listing: PolicyServerListing, choice: string): boolean {
  if (choice === LOCAL_POLICY) return listing.local.available;
  return listing.servers.some((entry) => entry.address === choice && entry.reachable);
}

/** What the picker starts on: the last rollout's choice while it is still usable, else this
 *  machine if it has a GPU, else the first server that answers. A robot host without a GPU (the
 *  NUC) therefore opens on its policy server instead of on a choice the gateway would refuse. */
export function defaultPolicyChoice(listing: PolicyServerListing, remembered?: string): string {
  if (remembered !== undefined && choiceAvailable(listing, remembered)) return remembered;
  if (listing.local.available) return LOCAL_POLICY;
  const reachable = listing.servers.find((entry) => entry.reachable);
  return reachable ? reachable.address : listing.servers[0]?.address ?? LOCAL_POLICY;
}

/** `outputs/train/<job>/checkpoints/<step>/...` names checkpoint `<job>/<step>`. */
export function holdsCheckpoint(entry: PolicyServerEntry, checkpointId: string): boolean {
  const loaded = entry.status?.loaded?.spec?.checkpoint;
  if (typeof loaded !== "string" || !checkpointId.includes("/")) return false;
  const [job, step] = checkpointId.split("/");
  return loaded.includes(`train/${job}/checkpoints/${step}/`) || loaded.endsWith(`train/${job}/checkpoints/${step}`);
}

function checkpointLabel(spec: Record<string, unknown> | null | undefined): string {
  const path = typeof spec?.checkpoint === "string" ? spec.checkpoint : "";
  const match = path.match(/train\/([^/]+)\/checkpoints\/([^/]+)/);
  return match ? `${match[1]}/${match[2]}` : path || "?";
}

function gigabytes(mb: number): string {
  return (mb / 1024).toFixed(1);
}

/** One line on what a server is doing, for the operator choosing it. */
export function describeServer(entry: PolicyServerEntry): string {
  if (!entry.reachable) return `unreachable — ${entry.error ?? "no answer"}`;
  const status = entry.status;
  if (!status) return "reachable";
  const parts: string[] = [];
  if (status.loading) parts.push(`loading ${checkpointLabel(status.loading)}`);
  else if (status.loaded) parts.push(`holds ${checkpointLabel(status.loaded.spec)}`);
  else parts.push("nothing loaded");
  if (status.holder) parts.push(`in use by ${status.holder}`);
  if (status.gpu) {
    const used = status.gpu.total_mb - status.gpu.free_mb;
    parts.push(`${status.gpu.name} ${gigabytes(used)}/${gigabytes(status.gpu.total_mb)} GB in use`);
  }
  return parts.join(" · ");
}

/** What pressing Start will cost on `entry` for `checkpointId`, or "" when it is ready. */
export function startNote(entry: PolicyServerEntry | undefined, checkpointId: string): string {
  if (!entry) return "";
  if (!entry.reachable) return `${entry.address} is not answering; a rollout started now fails at connect.`;
  if (entry.status?.holder) return `${entry.address} is serving another rollout; a second one is refused.`;
  if (checkpointId && !holdsCheckpoint(entry, checkpointId)) {
    return `Start loads this checkpoint on ${entry.address} first — about a minute for pi0.5.`;
  }
  return "";
}
