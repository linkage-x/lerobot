// Pure helpers of the Training page's GPU table (trainingGpus.test.ts).
import type { Gpu } from "./trainingApi";

/** A GPU someone is using: a process on it, or more than 1 GiB taken. */
export function gpuBusy(gpu: Gpu): boolean {
  return gpu.processes.length > 0 || (gpu.memUsedMiB ?? 0) > 1024;
}

export function gpuMemText(gpu: Gpu): string {
  if (gpu.memUsedMiB == null || gpu.memTotalMiB == null) return "—";
  return `${(gpu.memUsedMiB / 1024).toFixed(1)} / ${(gpu.memTotalMiB / 1024).toFixed(0)} GB`;
}

/** The GPUs a run should take on a host: the requested ones that exist, else the first free one. */
export function gpusFor(requested: number[], gpus: Gpu[]): number[] {
  const known = new Set(gpus.map((g) => g.index));
  const kept = requested.filter((g) => known.has(g));
  if (kept.length || gpus.length === 0) return gpus.length === 0 ? requested : kept;
  const free = gpus.find((g) => !gpuBusy(g));
  return free ? [free.index] : [];
}
