import type { GpuHostStatus, GpuService, GpuServiceOp } from "../types";

export type GpuAction = "start" | "stop" | "switch";

/** A button the panel offers for one service: enabled, or greyed with the reason why. */
export type GpuButton = { action: GpuAction; label: string; disabledReason: string };

function gigabytes(mb: number): string {
  return (mb / 1024).toFixed(1);
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "";
  const s = Math.max(0, Math.round(seconds));
  if (s < 90) return `${s} s`;
  const m = Math.round(s / 60);
  if (m < 90) return `${m} min`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h} h ${m % 60} min`;
  return `${Math.floor(h / 24)} d ${h % 24} h`;
}

export function freeMb(status: GpuHostStatus): number {
  return status.gpu.total_mb - status.gpu.used_mb;
}

/** One line on what a service is doing. */
export function describeService(svc: GpuService): string {
  const parts: string[] = [svc.state];
  if (svc.state !== "stopped") {
    if (svc.gpuMb) parts.push(`${gigabytes(svc.gpuMb)} GB`);
    const up = duration(svc.upS);
    if (up) parts.push(`up ${up}`);
  }
  if (svc.id === "policy_server" && svc.state === "running") {
    if (svc.loading) parts.push("loading a policy");
    else parts.push(svc.loaded ? `holds ${svc.loaded}` : "nothing loaded");
    if (svc.holder && svc.holder !== "warmup") parts.push(`serving ${svc.holder}`);
  }
  if (svc.state === "starting" && svc.port) parts.push(`waiting for :${svc.port}`);
  return parts.join(" · ");
}

/** The buttons for `svc`, with every refusal the agent would make shown before the click. */
export function buttonsFor(status: GpuHostStatus, svc: GpuService, rolloutLive: boolean): GpuButton[] {
  const blocked =
    rolloutLive
      ? "a rollout is running on this robot host"
      : status.op?.state === "running"
        ? `busy: ${status.op.action} ${status.op.service}`
        : "";
  const others = status.services.filter((other) => other.id !== svc.id && other.state !== "stopped");
  const servingRollout = (other: GpuService) => other.id === "policy_server" && !!other.holder && other.holder !== "warmup";
  if (svc.state !== "stopped") {
    return [
      {
        action: "stop",
        label: "Stop",
        disabledReason: blocked || (servingRollout(svc) ? `serving ${svc.holder}` : "")
      }
    ];
  }
  const training = status.training.length ? `training on this card (${status.training.join(", ")})` : "";
  const needMb = svc.gpuGb * 1024;
  const buttons: GpuButton[] = [];
  if (others.length) {
    const stuck = others.find(servingRollout);
    buttons.push({
      action: "switch",
      label: `Switch to (stops ${others.map((other) => other.id).join(", ")})`,
      disabledReason: blocked || training || (stuck ? `${stuck.id} is serving ${stuck.holder}` : "")
    });
  }
  buttons.push({
    action: "start",
    label: "Start",
    disabledReason:
      blocked ||
      training ||
      (freeMb(status) < needMb ? `needs ${svc.gpuGb.toFixed(0)} GB, ${gigabytes(freeMb(status))} GB free` : "")
  });
  return buttons;
}

/** What the last action did, for the line under the panel; "" when there was none. */
export function opNote(op: GpuServiceOp | null | undefined, now: number = Date.now() / 1000): string {
  if (!op) return "";
  const what = `${op.action} ${op.service}`;
  if (op.state === "running") return `${what}: ${op.message} (${duration(now - op.startedAt)})…`;
  const ago = op.finishedAt ? ` ${duration(now - op.finishedAt)} ago` : "";
  return `${what} ${op.state === "done" ? "done" : "FAILED"}${ago}: ${op.message}`;
}
