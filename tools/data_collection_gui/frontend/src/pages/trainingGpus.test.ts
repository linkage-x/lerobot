import { describe, expect, it } from "vitest";
import type { Gpu } from "./trainingApi";
import { gpuBusy, gpuMemText, gpusFor } from "./trainingGpus";

const gpu = (index: number, memUsedMiB: number, procs = 0): Gpu => ({
  index,
  name: "NVIDIA A100-SXM4-80GB",
  memUsedMiB,
  memTotalMiB: 81920,
  utilPct: 0,
  processes: Array.from({ length: procs }, (_, i) => ({ pid: 100 + i, user: "hph", memMiB: 500, command: "python" }))
});

describe("training GPUs", () => {
  it("calls a GPU busy when something runs on it or holds memory", () => {
    expect(gpuBusy(gpu(0, 0))).toBe(false);
    expect(gpuBusy(gpu(0, 0, 1))).toBe(true);
    expect(gpuBusy(gpu(0, 4096))).toBe(true);
    expect(gpuMemText(gpu(0, 2048))).toBe("2.0 / 80 GB");
  });

  it("keeps the requested GPUs a host has, else takes its first free one", () => {
    const gpus = [gpu(0, 9000, 1), gpu(1, 0), gpu(2, 0)];
    expect(gpusFor([0, 2, 7], gpus)).toEqual([0, 2]);
    expect(gpusFor([7], gpus)).toEqual([1]);
    expect(gpusFor([3], [])).toEqual([3]);
    expect(gpusFor([5], [gpu(0, 9000, 1)])).toEqual([]);
  });
});
